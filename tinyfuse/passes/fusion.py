"""Elementwise fusion.

Groups of elementwise ops (primitive ones such as ``add``/``relu`` or
already-fused ``fused_elementwise`` nodes) are merged into a single
``fused_elementwise`` node whose ``program`` attribute is a small
straight-line expression evaluated once per output element.  The C backend
emits one loop nest per fused node, so intermediates never touch memory.

Grouping rule (applied in reverse topological order): a node ``p`` joins a
group ``G`` iff

* ``p`` is not a graph output,
* *every* user of ``p`` is already in ``G``, and
* ``p`` has exactly the shape of ``G``'s root (so fusion never causes a
  broadcast producer to be recomputed per element of a bigger output).

Because every non-root member only feeds members of its own group, all
edges leaving a group start at its root, so fusing can never create a cycle.
Scalar constants are inlined into the program as literals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..ir import Graph, Node
from ..ops import ELEMENTWISE_ARITY

Operand = tuple


def is_elementwise(n: Node) -> bool:
    return n.op in ELEMENTWISE_ARITY or n.op == "fused_elementwise"


def _scalar_literal(n: Node) -> Optional[float]:
    if n.op == "const" and n.attrs["value"].size == 1:
        return float(n.attrs["value"].reshape(()))
    return None


def build_program(members: Sequence[Node], anchor: Optional[Node] = None,
                  inline_scalars: bool = True) -> Tuple[tuple, List[Node]]:
    """Turn ``members`` (topologically ordered elementwise nodes) into a
    program.  Values produced by members become temporaries; everything
    else becomes an external input (deduplicated) or, for scalar constants,
    a literal.  If ``anchor`` is given (a matmul / fused_matmul), references
    to it become the ``("acc",)`` operand (preceded by its own epilogue, if
    any).  Returns ``(program, external_inputs)``."""
    program: List[tuple] = []
    ext: List[Node] = []
    ext_idx: Dict[int, int] = {}
    value: Dict[int, Operand] = {}

    def external(x: Node) -> Operand:
        lit = _scalar_literal(x) if inline_scalars else None
        if lit is not None:
            return ("c", lit)
        if x.id not in ext_idx:
            ext_idx[x.id] = len(ext)
            ext.append(x)
        return ("in", ext_idx[x.id])

    def operand(x: Node) -> Operand:
        return value[x.id] if x.id in value else external(x)

    def inline(prog, input_operands, acc_operand=None):
        base = len(program)
        for op, opnds in prog:
            new = []
            for o in opnds:
                if o[0] == "in":
                    new.append(input_operands[o[1]])
                elif o[0] == "t":
                    new.append(("t", base + o[1]))
                elif o[0] == "acc":
                    new.append(acc_operand)
                else:
                    new.append(o)
            program.append((op, tuple(new)))
        return ("t", len(program) - 1)

    if anchor is not None:
        ep = anchor.attrs.get("epilogue") if anchor.op == "fused_matmul" else None
        if ep:
            extra = [external(x) for x in anchor.inputs[2:]]
            value[anchor.id] = inline(ep, extra, ("acc",))
        else:
            value[anchor.id] = ("acc",)

    for m in members:
        if m.op == "fused_elementwise":
            value[m.id] = inline(m.attrs["program"], [operand(x) for x in m.inputs])
        else:
            program.append((m.op, tuple(operand(x) for x in m.inputs)))
            value[m.id] = ("t", len(program) - 1)
    return tuple(program), ext


def elementwise_fusion(g: Graph) -> bool:
    users = g.users()
    outs = g.output_ids()
    group_of: Dict[int, int] = {}
    groups: Dict[int, List[Node]] = {}   # gid -> members (reverse topo order)
    root_of: Dict[int, Node] = {}

    for n in reversed(g.nodes):
        if not is_elementwise(n):
            continue
        us = users.get(n.id, [])
        gids = {group_of.get(u.id) for u in us}
        if n.id not in outs and us and len(gids) == 1 and None not in gids:
            gid = gids.pop()
            if root_of[gid].shape == n.shape:
                group_of[n.id] = gid
                groups[gid].append(n)
                continue
        gid = n.id
        group_of[n.id] = gid
        groups[gid] = [n]
        root_of[gid] = n

    changed = False
    replace: Dict[int, Node] = {}
    removed = set()
    for gid, members in groups.items():
        if len(members) < 2:
            continue
        members = list(reversed(members))  # topological order
        root = root_of[gid]
        program, ext = build_program(members)
        if not ext:  # keep shape inference well-defined
            program, ext = build_program(members, inline_scalars=False)
        fused = Node("fused_elementwise", ext, {"program": program})
        assert fused.shape == root.shape, (fused.shape, root.shape)
        replace[root.id] = fused
        removed.update(m.id for m in members)
        changed = True

    if not changed:
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
