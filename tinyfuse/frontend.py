"""Tracing frontend.

User code is ordinary Python that manipulates :class:`Tensor` objects.  While
:func:`trace` runs the function, every operation appends a node to the graph
being built, so control flow is resolved at trace time (as in ``torch.fx`` /
``jax.jit``): the result is a straight-line dataflow graph specialised to the
given input shapes.

Example::

    import tinyfuse as tf

    def mlp(x, w1, b1, w2, b2):
        return tf.relu(x @ w1 + b1) @ w2 + b2

    g = tf.trace(mlp, x=(32, 64), w1=(64, 128), b1=(128,), w2=(128, 10), b2=(10,))
"""

from __future__ import annotations

import contextlib
import inspect
import numbers
import threading
from typing import Callable, Optional

import numpy as np

from .ir import Graph, Node

_state = threading.local()


def _current_graph() -> Graph:
    g = getattr(_state, "graph", None)
    if g is None:
        raise RuntimeError("tinyfuse ops can only be used inside tinyfuse.trace()")
    return g


@contextlib.contextmanager
def _tracing(g: Graph):
    prev = getattr(_state, "graph", None)
    _state.graph = g
    try:
        yield g
    finally:
        _state.graph = prev


class Tensor:
    """A symbolic tensor: a handle to a node in the graph being traced."""

    __array_ufunc__ = None  # make `ndarray <op> Tensor` defer to Tensor

    def __init__(self, node: Node):
        self.node = node

    @property
    def shape(self):
        return self.node.shape

    @property
    def ndim(self):
        return len(self.node.shape)

    def __repr__(self):
        return f"Tensor({self.node.op}, shape={list(self.shape)})"

    # arithmetic -----------------------------------------------------------
    def __add__(self, o): return _binary("add", self, o)
    def __radd__(self, o): return _binary("add", o, self)
    def __sub__(self, o): return _binary("sub", self, o)
    def __rsub__(self, o): return _binary("sub", o, self)
    def __mul__(self, o): return _binary("mul", self, o)
    def __rmul__(self, o): return _binary("mul", o, self)
    def __truediv__(self, o): return _binary("div", self, o)
    def __rtruediv__(self, o): return _binary("div", o, self)
    def __neg__(self): return _unary("neg", self)
    def __matmul__(self, o): return matmul(self, o)
    def __rmatmul__(self, o): return matmul(o, self)

    # layout ---------------------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self, *perm):
        if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
            perm = perm[0]
        return transpose(self, perm or None)

    @property
    def T(self):
        return transpose(self, None)


def _as_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (numbers.Number, np.ndarray, np.generic, list, tuple)):
        return Tensor(_current_graph().add_const(x))
    raise TypeError(f"cannot convert {type(x).__name__} to a tinyfuse Tensor")


def constant(value) -> Tensor:
    """Embed a NumPy array / Python scalar in the graph as a constant."""
    return Tensor(_current_graph().add_const(value))


def _op(op: str, inputs, attrs: Optional[dict] = None) -> Tensor:
    ts = [_as_tensor(i) for i in inputs]
    return Tensor(_current_graph().add(op, [t.node for t in ts], attrs))


def _unary(op, x):
    return _op(op, [x])


def _binary(op, a, b):
    return _op(op, [a, b])


# public op API ------------------------------------------------------------

def add(a, b): return _binary("add", a, b)
def sub(a, b): return _binary("sub", a, b)
def mul(a, b): return _binary("mul", a, b)
def div(a, b): return _binary("div", a, b)
def maximum(a, b): return _binary("maximum", a, b)
def relu(x): return _unary("relu", x)
def gelu(x):
    """GELU, tanh approximation (== torch.nn.functional.gelu(x, approximate='tanh'))."""
    return _unary("gelu", x)
def neg(x): return _unary("neg", x)
def exp(x): return _unary("exp", x)
def tanh(x): return _unary("tanh", x)
def sigmoid(x): return _unary("sigmoid", x)
def sqrt(x): return _unary("sqrt", x)
def rsqrt(x): return _unary("rsqrt", x)
def matmul(a, b): return _op("matmul", [a, b])


def sum(x, axis: int = -1, keepdims: bool = True):  # noqa: A001 (mirrors numpy)
    """Sum over the last axis."""
    return _op("reduce_sum", [x], {"axis": axis, "keepdims": bool(keepdims)})


def mean(x, axis: int = -1, keepdims: bool = True):
    """Mean over the last axis."""
    return _op("reduce_mean", [x], {"axis": axis, "keepdims": bool(keepdims)})


def max(x, axis: int = -1, keepdims: bool = True):  # noqa: A001
    """Maximum over the last axis."""
    return _op("reduce_max", [x], {"axis": axis, "keepdims": bool(keepdims)})


def softmax(x, axis: int = -1):
    return _op("softmax", [x], {"axis": axis})


def layernorm(x, gamma, beta, eps: float = 1e-5):
    """Layer normalisation over the last axis (biased variance, like PyTorch)."""
    return _op("layernorm", [x, gamma, beta], {"eps": float(eps)})


def reshape(x, shape):
    return _op("reshape", [x], {"shape": tuple(int(d) for d in shape)})


def transpose(x, perm=None):
    x = _as_tensor(x)
    if perm is None:
        perm = tuple(reversed(range(x.ndim)))
    perm = tuple(int(p) % x.ndim for p in perm)
    return _op("transpose", [x], {"perm": perm})


# tracing ------------------------------------------------------------------

def trace(fn: Callable, *shapes, name: Optional[str] = None, **named_shapes) -> Graph:
    """Trace ``fn`` into a :class:`Graph`.

    Input shapes are given positionally (in parameter order) or by parameter
    name.  ``fn`` may return a single Tensor or a tuple/list of Tensors.
    """
    sig = inspect.signature(fn).parameters
    params = list(sig)
    specs = {}
    for p, s in zip(params, shapes):
        specs[p] = s
    for k, s in named_shapes.items():
        if k not in params:
            raise TypeError(f"{fn.__name__}() has no parameter '{k}'")
        if k in specs:
            raise TypeError(f"shape for '{k}' given twice")
        specs[k] = s
    if len(shapes) > len(params):
        raise TypeError("too many input shapes")
    # parameters with defaults and no shape keep their default (e.g. axis=-1)
    params = [p for p in params
              if p in specs or sig[p].default is inspect.Parameter.empty]
    missing = [p for p in params if p not in specs]
    if missing:
        raise TypeError(f"missing shapes for parameters {missing}")

    g = Graph(name or getattr(fn, "__name__", "graph"))
    with _tracing(g):
        args = [Tensor(g.add_input(tuple(specs[p]), p)) for p in params]
        out = fn(*args)
        outs = list(out) if isinstance(out, (tuple, list)) else [out]
        outs = [_as_tensor(o) for o in outs]
    g.outputs = [o.node for o in outs]
    return g
