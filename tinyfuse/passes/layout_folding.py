"""Layout folding: let matmuls read transposed / reshaped operands in place.

A ``transpose`` is just a strided view of its input, and so is a ``reshape``
of a contiguous view.  Instead of materialising them, this pass rewrites a
matmul operand that is a chain of such ops into its *base* tensor plus a
view ``(logical shape, element strides)`` stored on the node as ``a_view`` /
``b_view``.  The matmul's packing step already copies every operand once, so
reading through strides costs nothing extra; the transpose kernel, its
memory traffic and its buffer all disappear (DCE removes the dead layout ops).

Example (attention scores)::

    %k  = fused_matmul(%h, %wk, %bk)        [64,256]
    %r  = reshape(%k)                        [64,4,64]
    %kt = transpose(%r) {perm=[1,2,0]}       [4,64,64]
    %s  = fused_matmul(%q, %kt)
 ->
    %s  = fused_matmul(%q, %k) {b_view=f32[4,64,64] strides [64,1,256]}
"""

from __future__ import annotations

from typing import Tuple

from ..ir import Graph, Node
from ..ops import contiguous_strides

View = Tuple[Tuple[int, ...], Tuple[int, ...]]


def resolve_view(x: Node) -> Tuple[Node, View]:
    """Follow transpose/reshape producers of ``x`` back to a base tensor.
    Returns (base, (shape, strides)) such that x == view of base."""
    if x.op == "transpose":
        base, (shape, strides) = resolve_view(x.inputs[0])
        perm = x.attrs["perm"]
        return base, (tuple(shape[p] for p in perm), tuple(strides[p] for p in perm))
    if x.op == "reshape":
        base, (shape, strides) = resolve_view(x.inputs[0])
        if tuple(strides) == contiguous_strides(shape) or len(shape) == 0:
            return base, (tuple(x.shape), contiguous_strides(x.shape))
        return x, (tuple(x.shape), contiguous_strides(x.shape))
    return x, (tuple(x.shape), contiguous_strides(x.shape))


def layout_folding(g: Graph) -> bool:
    changed = False
    for i, n in enumerate(g.nodes):
        if n.op not in ("matmul", "fused_matmul"):
            continue
        new_inputs = list(n.inputs)
        attrs = dict(n.attrs)
        touched = False
        for idx, key in ((0, "a_view"), (1, "b_view")):
            x = n.inputs[idx]
            if x.op not in ("transpose", "reshape") or attrs.get(key) is not None:
                continue
            base, view = resolve_view(x)
            if base is x:
                continue
            if view == (tuple(base.shape), contiguous_strides(base.shape)):
                view = None   # plain reshape of a contiguous tensor
            new_inputs[idx] = base
            attrs[key] = view
            touched = True
        if not touched:
            continue
        if n.op == "matmul":
            attrs.setdefault("epilogue", None)
        m = Node("fused_matmul", new_inputs, attrs)
        assert m.shape == n.shape, (m.shape, n.shape)
        g.nodes[i] = m
        g.replace_all_uses(n, m)
        changed = True
    return changed
