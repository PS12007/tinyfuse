from __future__ import annotations

from typing import Dict

import numpy as np

from ..ir import Graph, Node
from ..ops import get_op

# Do not materialise folded constants bigger than this many elements.
MAX_FOLD_ELEMS = 1 << 24


def _const_scalar(n: Node):
    if n.op == "const" and n.attrs["value"].size == 1:
        return float(n.attrs["value"].reshape(()))
    return None


def _simplify(n: Node):
    """Algebraic identities.  Returns a replacement node or None.  Only
    rewrites that keep the output shape identical are applied."""
    op, ins = n.op, n.inputs
    if op in ("add", "sub", "mul", "div") and len(ins) == 2:
        a, b = ins
        cb, ca = _const_scalar(b), _const_scalar(a)
        if op in ("add", "sub") and cb == 0.0 and a.shape == n.shape:
            return a                       # x + 0, x - 0
        if op == "add" and ca == 0.0 and b.shape == n.shape:
            return b                       # 0 + x
        if op in ("mul", "div") and cb == 1.0 and a.shape == n.shape:
            return a                       # x * 1, x / 1
        if op == "mul" and ca == 1.0 and b.shape == n.shape:
            return b                       # 1 * x
    if op == "neg" and ins[0].op == "neg":
        return ins[0].inputs[0]            # --x
    if op == "reshape" and ins[0].shape == n.shape:
        return ins[0]                      # no-op reshape
    if op == "transpose" and tuple(n.attrs["perm"]) == tuple(range(len(n.shape))):
        return ins[0]                      # identity transpose
    return None


def _compose(n: Node):
    """Collapse layout chains: reshape(reshape(x)) and transpose(transpose(x)).
    Returns (op, inputs, attrs) for a new node, or None."""
    if n.op == "reshape" and n.inputs[0].op == "reshape":
        return ("reshape", [n.inputs[0].inputs[0]], {"shape": n.shape})
    if n.op == "transpose" and n.inputs[0].op == "transpose":
        inner = n.inputs[0].attrs["perm"]
        outer = n.attrs["perm"]
        perm = tuple(inner[p] for p in outer)
        return ("transpose", [n.inputs[0].inputs[0]], {"perm": perm})
    return None


def constant_folding(g: Graph) -> bool:
    """Evaluate nodes whose operands are all constants (using the NumPy
    reference semantics) and apply simple algebraic identities:
    ``x+0, x-0, x*1, x/1, 1*x, 0+x, --x``, no-op reshape/transpose, and
    reshape∘reshape / transpose∘transpose composition."""
    changed = False
    replace: Dict[int, Node] = {}

    def res(x: Node) -> Node:
        while x.id in replace:
            x = replace[x.id]
        return x

    new_nodes = []
    for n in g.nodes:
        n.inputs = [res(i) for i in n.inputs]
        if n.op in ("input", "const"):
            new_nodes.append(n)
            continue
        cur = n
        comp = _compose(cur)
        if comp is not None:
            op, ins, attrs = comp
            cur = Node(op, ins, attrs)
        if cur.inputs and all(i.op == "const" for i in cur.inputs) and \
                int(np.prod(cur.shape)) <= MAX_FOLD_ELEMS:
            val = get_op(cur.op).evaluate([i.attrs["value"] for i in cur.inputs], cur.attrs)
            cur = Node("const", [], {"value": _freeze_array(val)})
            assert cur.shape == n.shape
        else:
            s = _simplify(cur)
            if s is not None:
                replace[n.id] = s
                changed = True
                continue
        if cur is not n:
            replace[n.id] = cur
            changed = True
        new_nodes.append(cur)
    g.nodes = new_nodes
    g.outputs = [res(o) for o in g.outputs]
    return changed


def _freeze_array(v):
    arr = np.array(v, dtype=np.float32, order="C")
    arr.setflags(write=False)
    return arr
