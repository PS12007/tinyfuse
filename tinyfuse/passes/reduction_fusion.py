"""Reduction prologue fusion.

An elementwise chain whose only consumer is a last-axis reduction is computed
inside the reduction loop instead of being written to memory first:

    %0 = mul(%x, %x)                 ->   %0 = fused_reduce(%x) {kind=mean,
    %1 = reduce_mean(%0)                         program=[t0=mul(in0,in0)]}

This is what makes a hand-written RMSNorm or layernorm cheap: ``mean(x*x)``
streams ``x`` once and never materialises ``x*x``.
"""

from __future__ import annotations

from ..ir import Graph, Node
from ..ops import ShapeError, broadcast_shapes
from .fusion import build_program, is_elementwise


def reduction_fusion(g: Graph) -> bool:
    users = g.users()
    outs = g.output_ids()
    changed = False
    for i, r in enumerate(g.nodes):
        if not r.op.startswith("reduce_"):
            continue
        p = r.inputs[0]
        if not is_elementwise(p) or p.id in outs:
            continue
        if {u.id for u in users.get(p.id, [])} != {r.id}:
            continue
        program, ext = build_program([p])
        try:
            if not ext or broadcast_shapes(*[e.shape for e in ext]) != p.shape:
                continue
        except ShapeError:
            continue
        m = Node("fused_reduce", ext, {"kind": r.op[len("reduce_"):], "program": program,
                                       "in_shape": tuple(p.shape),
                                       "axis": r.attrs.get("axis", -1),
                                       "keepdims": r.attrs.get("keepdims", True)})
        assert m.shape == r.shape
        g.nodes[i] = m
        g.replace_all_uses(r, m)
        changed = True
    return changed
