from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np

from ..ir import Graph, Node
from ..ops import get_op


def _freeze(v):
    if isinstance(v, np.ndarray):
        return ("ndarray", v.shape, hashlib.sha1(v.tobytes()).hexdigest())
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


def node_key(n: Node, inputs) -> tuple:
    ids = [i.id for i in inputs]
    if get_op(n.op).commutative:
        ids = sorted(ids)
    return (n.op, tuple(ids), _freeze(n.attrs), n.shape)


def common_subexpression_elimination(g: Graph) -> bool:
    """Merge nodes that compute the same op on the same operands with the same
    attributes.  Constants are merged by value (shape + bytes), and operands of
    commutative ops are compared as a sorted multiset, so ``a+b`` and ``b+a``
    are merged.  Graph inputs are never merged."""
    seen: Dict[tuple, Node] = {}
    replace: Dict[int, Node] = {}
    new_nodes = []
    for n in g.nodes:
        n.inputs = [replace.get(i.id, i) for i in n.inputs]
        if n.op == "input":
            new_nodes.append(n)
            continue
        k = node_key(n, n.inputs)
        if k in seen:
            replace[n.id] = seen[k]
        else:
            seen[k] = n
            new_nodes.append(n)
    if not replace:
        return False
    g.nodes = new_nodes
    g.outputs = [replace.get(o.id, o) for o in g.outputs]
    return True
