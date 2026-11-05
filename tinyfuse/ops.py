"""Operator registry.

Every op in the IR is described by an :class:`OpDef` that provides

* ``infer``    -- static shape inference ``(input_shapes, attrs) -> shape``
* ``evaluate`` -- a NumPy reference implementation ``(input_arrays, attrs) -> array``
* flags such as ``elementwise`` and ``commutative`` used by the passes.

The NumPy implementations are the *semantic definition* of each op: the
interpreter, the constant folder and the tests all use them, and the C code
generator must agree with them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

Shape = Tuple[int, ...]

GELU_C = 0.7978845608028654  # sqrt(2/pi)


class ShapeError(ValueError):
    """Raised when an op is applied to incompatible shapes/attributes."""


@dataclass(frozen=True)
class OpDef:
    name: str
    arity: Optional[int]  # None = variadic
    infer: Callable[[Sequence[Shape], dict], Shape]
    evaluate: Callable[[Sequence[np.ndarray], dict], np.ndarray]
    elementwise: bool = False
    commutative: bool = False
    # Ops whose result is a view of input 0 (no data movement needed).
    alias: bool = False
    attr_names: Tuple[str, ...] = field(default_factory=tuple)


OPS: Dict[str, OpDef] = {}


def register(op: OpDef) -> OpDef:
    if op.name in OPS:
        raise ValueError(f"duplicate op {op.name}")
    OPS[op.name] = op
    return op


def get_op(name: str) -> OpDef:
    try:
        return OPS[name]
    except KeyError:
        raise KeyError(f"unknown op '{name}'") from None


# --------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------

def broadcast_shapes(*shapes: Shape) -> Shape:
    """NumPy broadcasting rules, with a friendlier error."""
    ndim = max((len(s) for s in shapes), default=0)
    out = []
    for i in range(ndim):
        dim = 1
        for s in shapes:
            j = i - (ndim - len(s))
            if j < 0:
                continue
            d = s[j]
            if d == 1:
                continue
            if dim != 1 and dim != d:
                raise ShapeError(f"cannot broadcast shapes {list(map(list, shapes))}")
            dim = d
        out.append(dim)
    return tuple(out)


def broadcast_strides(in_shape: Shape, out_shape: Shape) -> Tuple[int, ...]:
    """Element strides of a contiguous ``in_shape`` tensor viewed as
    ``out_shape`` under broadcasting (0 for broadcast dimensions)."""
    nd = len(out_shape)
    pad = nd - len(in_shape)
    if pad < 0:
        raise ShapeError(f"{in_shape} has more dims than {out_shape}")
    strides = [0] * nd
    acc = 1
    for i in range(len(in_shape) - 1, -1, -1):
        d = in_shape[i]
        if d != 1:
            if d != out_shape[pad + i]:
                raise ShapeError(f"{in_shape} does not broadcast to {out_shape}")
            strides[pad + i] = acc
        acc *= d
    return tuple(strides)


def numel(shape: Shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _norm_axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ShapeError(f"axis {axis} out of range for rank {ndim}")
    return axis % ndim


# --------------------------------------------------------------------------
# Leaf ops
# --------------------------------------------------------------------------

def _no_eval(name):
    def ev(xs, attrs):
        raise RuntimeError(f"op '{name}' cannot be evaluated directly")
    return ev


register(OpDef("input", 0, lambda s, a: tuple(a["shape"]), _no_eval("input"),
               attr_names=("shape",)))
register(OpDef("const", 0, lambda s, a: tuple(a["value"].shape),
               lambda xs, a: a["value"], attr_names=("value",)))


# --------------------------------------------------------------------------
# Elementwise ops.  Scalar semantics live in SCALAR_FNS so that the fused
# elementwise interpreter and the C emitter share one table.
# --------------------------------------------------------------------------

def np_gelu(x):
    x = np.asarray(x)
    c = np.asarray(GELU_C, dtype=x.dtype)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


UNARY_FNS = {
    "relu": lambda x: np.maximum(x, 0),
    "gelu": np_gelu,
    "neg": lambda x: -x,
    "exp": np.exp,
    "tanh": np.tanh,
    "sigmoid": np_sigmoid,
}

BINARY_FNS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "maximum": np.maximum,
}

ELEMENTWISE_ARITY = {**{k: 1 for k in UNARY_FNS}, **{k: 2 for k in BINARY_FNS}}


def _ew_infer(shapes, attrs):
    return broadcast_shapes(*shapes)


def _f32(x):
    return np.asarray(x, dtype=np.float32)


for _name, _fn in UNARY_FNS.items():
    register(OpDef(_name, 1, _ew_infer,
                   (lambda fn: lambda xs, a: _f32(fn(xs[0])))(_fn), elementwise=True))

for _name, _fn in BINARY_FNS.items():
    register(OpDef(_name, 2, _ew_infer,
                   (lambda fn: lambda xs, a: _f32(fn(xs[0], xs[1])))(_fn), elementwise=True,
                   commutative=_name in ("add", "mul", "maximum")))


# --------------------------------------------------------------------------
# Matmul
# --------------------------------------------------------------------------

def matmul_shape(a: Shape, b: Shape) -> Shape:
    """Supported forms: [M,K]@[K,N], [*,M,K]@[K,N], [*,M,K]@[*,K,N]
    (identical batch dims; no batch broadcasting)."""
    if len(a) < 2 or len(b) < 2:
        raise ShapeError(f"matmul needs rank>=2 operands, got {a} @ {b}")
    if a[-1] != b[-2]:
        raise ShapeError(f"matmul inner dims differ: {a} @ {b}")
    if len(b) == 2:
        return tuple(a[:-1]) + (b[-1],)
    if tuple(a[:-2]) != tuple(b[:-2]):
        raise ShapeError(f"matmul batch dims must match exactly: {a} @ {b}")
    return tuple(a[:-1]) + (b[-1],)


register(OpDef("matmul", 2, lambda s, a: matmul_shape(s[0], s[1]),
               lambda xs, a: _f32(np.matmul(xs[0], xs[1]))))


# --------------------------------------------------------------------------
# Row-wise reductions (over the last axis only)
# --------------------------------------------------------------------------

def _softmax_infer(s, a):
    x = s[0]
    if len(x) == 0:
        raise ShapeError("softmax needs rank>=1")
    if _norm_axis(a.get("axis", -1), len(x)) != len(x) - 1:
        raise ShapeError("softmax only supports the last axis")
    return tuple(x)


def np_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


register(OpDef("softmax", 1, _softmax_infer,
               lambda xs, a: _f32(np_softmax(xs[0], -1)), attr_names=("axis",)))


def _layernorm_infer(s, a):
    x, g, b = s
    if len(x) == 0:
        raise ShapeError("layernorm needs rank>=1")
    d = x[-1]
    if tuple(g) != (d,) or tuple(b) != (d,):
        raise ShapeError(f"layernorm gamma/beta must have shape ({d},), got {g}, {b}")
    return tuple(x)


def np_layernorm(x, g, b, eps):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b


register(OpDef("layernorm", 3, _layernorm_infer,
               lambda xs, a: _f32(np_layernorm(xs[0], xs[1], xs[2], a["eps"])),
               attr_names=("eps",)))


# --------------------------------------------------------------------------
# Layout ops
# --------------------------------------------------------------------------

def _reshape_infer(s, a):
    x = s[0]
    shape = list(a["shape"])
    if shape.count(-1) > 1:
        raise ShapeError("reshape: at most one -1 allowed")
    known = numel([d for d in shape if d != -1])
    if -1 in shape:
        if known == 0 or numel(x) % known:
            raise ShapeError(f"cannot reshape {x} to {a['shape']}")
        shape[shape.index(-1)] = numel(x) // known
    if numel(shape) != numel(x):
        raise ShapeError(f"cannot reshape {x} to {a['shape']}")
    return tuple(shape)


register(OpDef("reshape", 1, _reshape_infer,
               lambda xs, a: xs[0].reshape(_reshape_infer([xs[0].shape], a)),
               alias=True, attr_names=("shape",)))


def _transpose_infer(s, a):
    x = s[0]
    perm = tuple(a["perm"])
    if sorted(perm) != list(range(len(x))):
        raise ShapeError(f"transpose: bad perm {perm} for rank {len(x)}")
    return tuple(x[p] for p in perm)


register(OpDef("transpose", 1, _transpose_infer,
               lambda xs, a: np.ascontiguousarray(np.transpose(xs[0], a["perm"])),
               attr_names=("perm",)))


# --------------------------------------------------------------------------
# Fused ops (created only by passes)
#
# An elementwise *program* is a tuple of instructions ``(op, operands)``.
# Operands are ("in", i)  -> external input i of the fused node
#             ("t", j)    -> result of instruction j
#             ("c", f)    -> a float literal
#             ("acc",)    -> the matmul accumulator (fused_matmul epilogues only)
# The result of the program is the last instruction.
# --------------------------------------------------------------------------

def eval_program(program, inputs, out_shape, acc=None):
    vals = []

    def get(o):
        kind = o[0]
        if kind == "in":
            return inputs[o[1]]
        if kind == "t":
            return vals[o[1]]
        if kind == "c":
            return np.float32(o[1])
        if kind == "acc":
            return acc
        raise ValueError(f"bad operand {o}")

    for op, operands in program:
        args = [get(o) for o in operands]
        if op in UNARY_FNS:
            vals.append(_f32(UNARY_FNS[op](*args)))
        else:
            vals.append(_f32(BINARY_FNS[op](*args)))
    return np.broadcast_to(vals[-1], out_shape).astype(np.float32)


def check_program(program, n_inputs, allow_acc=False):
    if not program:
        raise ShapeError("empty elementwise program")
    for j, (op, operands) in enumerate(program):
        if op not in ELEMENTWISE_ARITY:
            raise ShapeError(f"program uses non-elementwise op {op}")
        if len(operands) != ELEMENTWISE_ARITY[op]:
            raise ShapeError(f"program op {op} has {len(operands)} operands")
        for o in operands:
            if o[0] == "in" and not 0 <= o[1] < n_inputs:
                raise ShapeError(f"program references input {o[1]} of {n_inputs}")
            if o[0] == "t" and not 0 <= o[1] < j:
                raise ShapeError(f"program references future temp {o[1]} at {j}")
            if o[0] == "acc" and not allow_acc:
                raise ShapeError("('acc',) operand outside a matmul epilogue")
            if o[0] not in ("in", "t", "c", "acc"):
                raise ShapeError(f"bad operand {o}")


def _fused_ew_infer(s, a):
    check_program(a["program"], len(s))
    return broadcast_shapes(*s)


register(OpDef("fused_elementwise", None, _fused_ew_infer,
               lambda xs, a: eval_program(a["program"], xs, broadcast_shapes(*[x.shape for x in xs])),
               attr_names=("program",)))


def _fused_mm_infer(s, a):
    out = matmul_shape(s[0], s[1])
    ep = a.get("epilogue")
    if ep is not None:
        check_program(ep, len(s) - 2, allow_acc=True)
        for extra in s[2:]:
            if broadcast_shapes(out, extra) != out:
                raise ShapeError(f"epilogue input {extra} does not broadcast to {out}")
    elif len(s) != 2:
        raise ShapeError("fused_matmul without epilogue takes exactly 2 inputs")
    return out


def _fused_mm_eval(xs, a):
    acc = _f32(np.matmul(xs[0], xs[1]))
    ep = a.get("epilogue")
    if ep is None:
        return acc
    return eval_program(ep, xs[2:], acc.shape, acc=acc)


register(OpDef("fused_matmul", None, _fused_mm_infer, _fused_mm_eval,
               attr_names=("epilogue",)))
