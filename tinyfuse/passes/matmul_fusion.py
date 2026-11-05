"""Matmul + bias + activation (epilogue) fusion.

A ``matmul`` followed by a chain of elementwise consumers -- typically
``add(bias)`` then ``relu``/``gelu``, but any elementwise op is accepted,
e.g. a residual add or the ``* scale`` in attention -- is rewritten into a
single ``fused_matmul`` node whose ``epilogue`` program is applied to each
output tile while it is still hot in cache.

A consumer ``u`` of the current chain tail ``t`` is absorbed iff ``t`` is
not a graph output, ``u`` is ``t``'s only user, ``u`` is elementwise, and
``u`` has the matmul's output shape.  The extra operands of ``u`` (bias,
residual, ...) become additional inputs of the fused node; they may be any
shape that broadcasts to the output.
"""

from __future__ import annotations

from typing import Dict

from ..ir import Graph, Node
from .fusion import build_program, is_elementwise


def matmul_epilogue_fusion(g: Graph) -> bool:
    users = g.users()
    outs = g.output_ids()
    replace: Dict[int, Node] = {}   # last chain node -> fused node
    removed = set()
    for m in g.nodes:
        if m.op not in ("matmul", "fused_matmul") or m.id in removed:
            continue
        chain = []
        cur = m
        while cur.id not in outs:
            us = {u.id: u for u in users.get(cur.id, [])}
            if len(us) != 1:
                break
            u = next(iter(us.values()))
            if not is_elementwise(u) or u.shape != m.shape:
                break
            chain.append(u)
            cur = u
        if not chain:
            continue
        program, ext = build_program(chain, anchor=m)
        attrs = {k: m.attrs[k] for k in ("a_view", "b_view") if m.attrs.get(k) is not None}
        attrs["epilogue"] = program
        fused = Node("fused_matmul", [m.inputs[0], m.inputs[1]] + ext, attrs)
        assert fused.shape == chain[-1].shape
        replace[chain[-1].id] = fused
        removed.add(m.id)
        removed.update(c.id for c in chain[:-1])
    if not replace:
        return False
    new_nodes = []
    for n in g.nodes:
        if n.id in replace:
            new_nodes.append(replace[n.id])
        elif n.id not in removed:
            new_nodes.append(n)
    for n in new_nodes:
        n.inputs = [replace.get(i.id, i) for i in n.inputs]
    g.nodes = new_nodes
    g.outputs = [replace.get(o.id, o) for o in g.outputs]
    return True
