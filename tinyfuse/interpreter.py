"""Reference interpreter: executes a graph node-by-node with the NumPy
implementations from :mod:`tinyfuse.ops`.  Used to check that passes preserve
semantics independently of the C backend."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .ir import Graph
from .ops import get_op


def run_graph(g: Graph, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
    if len(inputs) != len(g.inputs):
        raise TypeError(f"graph expects {len(g.inputs)} inputs, got {len(inputs)}")
    env: Dict[int, np.ndarray] = {}
    for node, val in zip(g.inputs, inputs):
        arr = np.asarray(val, dtype=np.float32)
        if arr.shape != node.shape:
            raise ValueError(f"input '{node.name}' expects shape {node.shape}, got {arr.shape}")
        env[node.id] = arr
    for n in g.nodes:
        if n.op == "input":
            continue
        out = get_op(n.op).evaluate([env[i.id] for i in n.inputs], n.attrs)
        out = np.asarray(out, dtype=np.float32)
        assert out.shape == n.shape, (n, out.shape)
        env[n.id] = out
    return [env[o.id] for o in g.outputs]
