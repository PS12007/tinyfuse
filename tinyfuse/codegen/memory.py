"""Buffer assignment and workspace planning.

Every node value lives in a *storage*:

* ``input``  -- a caller-supplied input array
* ``const``  -- a constant array owned by the compiled module
* ``output`` -- a caller-visible output array (allocated per call)
* ``ws``     -- a slice of the module's workspace

``reshape`` is free: it shares the storage of its operand.  Workspace slices
are assigned with a greedy best-fit allocator over value lifetimes, so
intermediates whose lifetimes do not overlap reuse the same memory.  A
slice is only released *after* the node that last reads it has had its own
output allocated, so a kernel never writes into memory it is reading
(which makes ``restrict`` on kernel arguments valid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..ir import Graph, Node
from ..ops import get_op, numel

ALIGN = 16  # floats (64 bytes)


def _round(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


@dataclass
class Storage:
    sid: int
    kind: str          # input | const | output | ws
    size: int          # in floats
    index: int = -1    # input/const/output index
    offset: int = -1   # workspace offset (floats) for kind == ws
    first: int = -1    # node index that defines it
    last: int = -1     # last node index that reads it


@dataclass
class MemoryPlan:
    node_storage: Dict[int, Storage]
    storages: List[Storage]
    consts: List[Node]
    output_copies: List[Tuple[int, Storage]]   # (output index, source storage)
    ws_size: int                               # floats used by tensors
    naive_ws_size: int = 0                     # what it would be without reuse

    def describe(self) -> str:
        return (f"workspace: {self.ws_size * 4 / 1024:.1f} KiB "
                f"(no reuse: {self.naive_ws_size * 4 / 1024:.1f} KiB)")


def plan_memory(g: Graph, reuse: bool = True) -> MemoryPlan:
    node_storage: Dict[int, Storage] = {}
    storages: List[Storage] = []
    consts: List[Node] = []

    def new(kind, size, **kw):
        s = Storage(len(storages), kind, size, **kw)
        storages.append(s)
        return s

    for i, n in enumerate(g.inputs):
        node_storage[n.id] = new("input", numel(n.shape), index=i)
    for idx, n in enumerate(g.nodes):
        if n.op == "input":
            continue
        if n.op == "const":
            node_storage[n.id] = new("const", numel(n.shape), index=len(consts))
            consts.append(n)
        elif get_op(n.op).alias:
            node_storage[n.id] = node_storage[n.inputs[0].id]
        else:
            node_storage[n.id] = new("ws", numel(n.shape), first=idx)

    # outputs: write straight into the caller's array when we can
    copies: List[Tuple[int, Storage]] = []
    claimed = set()
    for oi, o in enumerate(g.outputs):
        st = node_storage[o.id]
        if st.kind == "ws" and st.sid not in claimed:
            st.kind = "output"
            st.index = oi
            claimed.add(st.sid)
        else:
            copies.append((oi, st))

    # lifetimes
    end = len(g.nodes)
    for idx, n in enumerate(g.nodes):
        for x in n.inputs:
            st = node_storage[x.id]
            st.last = max(st.last, idx)
    for _, st in copies:
        st.last = end
    for st in storages:
        if st.kind == "ws" and st.last < st.first:
            st.last = st.first  # defined but never read (shouldn't happen after DCE)

    # greedy best-fit allocation of ws storages
    free: List[Tuple[int, int]] = []   # (offset, size)
    top = 0
    by_first: Dict[int, List[Storage]] = {}
    by_last: Dict[int, List[Storage]] = {}
    for st in storages:
        if st.kind == "ws":
            by_first.setdefault(st.first, []).append(st)
            by_last.setdefault(st.last, []).append(st)

    def alloc(size):
        nonlocal top
        size = _round(size)
        best = None
        if reuse:
            for k, (off, sz) in enumerate(free):
                if sz >= size and (best is None or sz < free[best][1]):
                    best = k
        if best is None:
            off = top
            top += size
            return off
        off, sz = free.pop(best)
        if sz > size:
            free.append((off + size, sz - size))
        return off

    def release(off, size):
        size = _round(size)
        free.append((off, size))
        # coalesce neighbours
        free.sort()
        merged: List[Tuple[int, int]] = []
        for o, s in free:
            if merged and merged[-1][0] + merged[-1][1] == o:
                merged[-1] = (merged[-1][0], merged[-1][1] + s)
            else:
                merged.append((o, s))
        free[:] = merged

    for idx in range(end + 1):
        for st in by_first.get(idx, []):
            st.offset = alloc(st.size)
        for st in by_last.get(idx, []):
            release(st.offset, st.size)

    naive = sum(_round(st.size) for st in storages if st.kind == "ws")
    return MemoryPlan(node_storage, storages, consts, copies, top, naive)
