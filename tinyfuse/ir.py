"""Graph IR.

The IR is a single static-single-assignment dataflow graph:

* every :class:`Node` produces exactly one float32 tensor with a static shape;
* ``Graph.nodes`` holds every node (inputs, constants, ops) in a valid
  topological order;
* ``Graph.inputs`` lists the ``input`` nodes in call-signature order and
  ``Graph.outputs`` lists the nodes whose values are returned.

Shapes are inferred eagerly when a node is created (``Graph.add``) and can be
re-checked at any time with :func:`verify`.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .ops import OPS, ShapeError, get_op

_ids = itertools.count()


class Node:
    __slots__ = ("id", "op", "inputs", "attrs", "shape", "dtype", "name")

    def __init__(self, op: str, inputs: Sequence["Node"], attrs: Optional[dict] = None,
                 shape: Optional[Tuple[int, ...]] = None, name: Optional[str] = None):
        self.id = next(_ids)
        self.op = op
        self.inputs: List[Node] = list(inputs)
        self.attrs: dict = dict(attrs or {})
        self.dtype = "f32"
        self.name = name
        if shape is None:
            shape = infer_shape(op, [i.shape for i in self.inputs], self.attrs)
        self.shape: Tuple[int, ...] = tuple(int(d) for d in shape)

    def __repr__(self):
        return f"Node({self.op}#{self.id}, shape={list(self.shape)})"


def infer_shape(op: str, in_shapes, attrs) -> Tuple[int, ...]:
    opdef = get_op(op)
    if opdef.arity is not None and len(in_shapes) != opdef.arity:
        raise ShapeError(f"{op} expects {opdef.arity} inputs, got {len(in_shapes)}")
    return tuple(opdef.infer([tuple(s) for s in in_shapes], attrs))


class Graph:
    def __init__(self, name: str = "graph"):
        self.name = name
        self.inputs: List[Node] = []
        self.nodes: List[Node] = []
        self.outputs: List[Node] = []

    # -- construction ------------------------------------------------------
    def add_input(self, shape, name: str) -> Node:
        n = Node("input", [], {"shape": tuple(shape)}, name=name)
        self.inputs.append(n)
        self.nodes.append(n)
        return n

    def add_const(self, value) -> Node:
        arr = np.array(value, dtype=np.float32, order="C")  # copy; keeps 0-d shape
        arr.setflags(write=False)
        n = Node("const", [], {"value": arr})
        self.nodes.append(n)
        return n

    def add(self, op: str, inputs: Sequence[Node], attrs: Optional[dict] = None) -> Node:
        n = Node(op, inputs, attrs)
        self.nodes.append(n)
        return n

    # -- queries -------------------------------------------------------------
    def users(self) -> Dict[int, List[Node]]:
        """Map node id -> list of user nodes (one entry per use)."""
        u: Dict[int, List[Node]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for i in n.inputs:
                u.setdefault(i.id, []).append(n)
        return u

    def output_ids(self):
        return {o.id for o in self.outputs}

    # -- mutation helpers ------------------------------------------------------
    def replace_all_uses(self, old: Node, new: Node):
        for n in self.nodes:
            n.inputs = [new if i is old else i for i in n.inputs]
        self.outputs = [new if o is old else o for o in self.outputs]

    def insert_before(self, anchor: Node, node: Node):
        self.nodes.insert(self.nodes.index(anchor), node)

    def toposort(self):
        """Re-order ``nodes`` topologically (stable w.r.t. the current order)."""
        order: List[Node] = []
        seen = set()
        on_stack = set()
        for root in list(self.nodes):
            if root.id in seen:
                continue
            stack = [(root, 0)]
            while stack:
                n, i = stack.pop()
                if i == 0:
                    if n.id in seen:
                        continue
                    on_stack.add(n.id)
                if i < len(n.inputs):
                    stack.append((n, i + 1))
                    c = n.inputs[i]
                    if c.id in on_stack:
                        raise IRError("cycle detected in graph")
                    if c.id not in seen:
                        stack.append((c, 0))
                else:
                    on_stack.discard(n.id)
                    seen.add(n.id)
                    order.append(n)
        self.nodes = order

    def copy(self) -> "Graph":
        """Deep-copy the graph structure (constant arrays are shared; they are
        read-only)."""
        g = Graph(self.name)
        m: Dict[int, Node] = {}
        for n in self.nodes:
            c = Node(n.op, [m[i.id] for i in n.inputs], dict(n.attrs), shape=n.shape, name=n.name)
            m[n.id] = c
            g.nodes.append(c)
        g.inputs = [m[i.id] for i in self.inputs]
        g.outputs = [m[o.id] for o in self.outputs]
        return g

    def op_counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for n in self.nodes:
            c[n.op] = c.get(n.op, 0) + 1
        return c

    def __str__(self):
        return format_graph(self)


class IRError(Exception):
    pass


# --------------------------------------------------------------------------
# Pretty printer
# --------------------------------------------------------------------------

def _fmt_shape(s):
    return "f32[" + ",".join(str(d) for d in s) + "]"


def _fmt_float(v: float) -> str:
    r = repr(float(v))
    return r[:-2] if r.endswith(".0") else r


def format_program(program) -> str:
    def opnd(o):
        if o[0] == "in":
            return f"in{o[1]}"
        if o[0] == "t":
            return f"t{o[1]}"
        if o[0] == "c":
            return _fmt_float(o[1])
        return "acc"
    return "; ".join(f"t{j}={op}({','.join(opnd(o) for o in ops)})"
                     for j, (op, ops) in enumerate(program))


def _fmt_attrs(n: Node) -> str:
    parts = []
    for k in sorted(n.attrs):
        v = n.attrs[k]
        if n.op == "input" and k == "shape":
            continue
        if k == "value":
            arr = v
            if arr.size <= 4:
                parts.append("value=" + np.array2string(arr, separator=",").replace("\n", ""))
            else:
                parts.append(f"value=<{arr.size} elems>")
        elif k in ("program", "epilogue"):
            if v is not None:
                parts.append(f"{k}=[{format_program(v)}]")
        elif k == "tile":
            parts.append(f"tile={v}")
        else:
            if isinstance(v, (list, tuple)):
                v = "[" + ",".join(str(x) for x in v) + "]"
            parts.append(f"{k}={v}")
    return (" {" + ", ".join(parts) + "}") if parts else ""


def format_graph(g: Graph) -> str:
    """Render the graph as text. Value names are renumbered (%0, %1, ...)
    in node order so output is deterministic and diff-friendly."""
    names: Dict[int, str] = {}
    counter = 0
    for n in g.nodes:
        if n.op == "input":
            names[n.id] = "%" + (n.name or f"arg{len(names)}")
    for n in g.nodes:
        if n.op != "input":
            names[n.id] = f"%{counter}"
            counter += 1

    def nm(n):
        return names.get(n.id, f"%?{n.id}")

    sig = ", ".join(f"{nm(i)}: {_fmt_shape(i.shape)}" for i in g.inputs)
    lines = [f"graph {g.name}({sig}) {{"]
    for n in g.nodes:
        if n.op == "input":
            continue
        args = ", ".join(nm(i) for i in n.inputs)
        lines.append(f"  {nm(n)}: {_fmt_shape(n.shape)} = {n.op}({args}){_fmt_attrs(n)}")
    lines.append("  return " + ", ".join(nm(o) for o in g.outputs))
    lines.append("}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------

def verify(g: Graph) -> None:
    """Check structural and type invariants; raise :class:`IRError` on failure.

    * node ids are unique and every operand is defined earlier in ``nodes``;
    * every op is registered and has the right arity;
    * stored shapes equal re-inferred shapes;
    * inputs are ``input`` nodes, appear in ``nodes``, and have unique names;
    * outputs are non-empty and defined in the graph;
    * constants are float32 arrays whose shape matches.
    """
    seen = set()
    for n in g.nodes:
        if n.id in seen:
            raise IRError(f"node {n} appears twice")
        if n.op not in OPS:
            raise IRError(f"unknown op '{n.op}' in {n}")
        for i in n.inputs:
            if i.id not in seen:
                raise IRError(f"{n} uses {i} before it is defined (or it is not in the graph)")
        try:
            s = infer_shape(n.op, [i.shape for i in n.inputs], n.attrs)
        except (ShapeError, KeyError) as e:
            raise IRError(f"shape inference failed for {n}: {e}") from e
        if tuple(s) != tuple(n.shape):
            raise IRError(f"{n}: stored shape {n.shape} != inferred {s}")
        if n.op == "const":
            v = n.attrs["value"]
            if not isinstance(v, np.ndarray) or v.dtype != np.float32:
                raise IRError(f"{n}: constant must be a float32 ndarray")
        if n.op == "input" and n not in g.inputs:
            raise IRError(f"{n}: input node not listed in graph.inputs")
        seen.add(n.id)
    names = set()
    for i in g.inputs:
        if i.op != "input":
            raise IRError(f"graph input {i} is not an input node")
        if i.id not in seen:
            raise IRError(f"graph input {i} missing from nodes")
        if i.name in names:
            raise IRError(f"duplicate input name {i.name}")
        names.add(i.name)
    if not g.outputs:
        raise IRError("graph has no outputs")
    for o in g.outputs:
        if o.id not in seen:
            raise IRError(f"output {o} is not defined in the graph")


def live_nodes(g: Graph) -> Iterable[Node]:
    """Nodes reachable from the outputs (plus all graph inputs)."""
    live = {i.id for i in g.inputs}
    stack = list(g.outputs)
    while stack:
        n = stack.pop()
        if n.id in live and n.op != "input":
            continue
        live.add(n.id)
        stack.extend(n.inputs)
    return [n for n in g.nodes if n.id in live]
