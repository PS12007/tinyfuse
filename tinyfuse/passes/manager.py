from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

from ..ir import Graph, format_graph, verify
from .constant_folding import constant_folding
from .cse import common_subexpression_elimination
from .dce import dead_code_elimination
from .fusion import elementwise_fusion
from .layout_folding import layout_folding
from .matmul_fusion import matmul_epilogue_fusion

PassFn = Callable[[Graph], bool]

_CLEANUP: List[PassFn] = [constant_folding, common_subexpression_elimination,
                          dead_code_elimination]

PIPELINES: Dict[str, List[PassFn]] = {
    # no optimisation at all: one kernel per traced op
    "none": [],
    # graph cleanups, but every op still gets its own kernel
    "unfused": _CLEANUP,
    # cleanups + fusion
    "fused": _CLEANUP + [matmul_epilogue_fusion, elementwise_fusion, layout_folding,
                         dead_code_elimination],
}


class PassManager:
    def __init__(self, passes: Sequence[PassFn], verify_each: bool = True,
                 max_iterations: int = 1):
        self.passes = list(passes)
        self.verify_each = verify_each
        self.max_iterations = max_iterations
        self.log: List[Tuple[str, bool]] = []
        self.snapshots: List[Tuple[str, str]] = []

    def run(self, g: Graph, keep_snapshots: bool = False) -> Graph:
        if self.verify_each:
            verify(g)
        if keep_snapshots:
            self.snapshots.append(("input", format_graph(g)))
        for _ in range(self.max_iterations):
            any_change = False
            for p in self.passes:
                changed = p(g)
                self.log.append((p.__name__, changed))
                any_change |= changed
                if self.verify_each:
                    verify(g)
                if keep_snapshots:
                    self.snapshots.append((p.__name__, format_graph(g)))
            if not any_change:
                break
        return g


def optimize(g: Graph, pipeline: str = "fused", verify_each: bool = True) -> Graph:
    """Run a named pipeline on a copy of ``g`` and return the copy."""
    g = g.copy()
    PassManager(PIPELINES[pipeline], verify_each=verify_each, max_iterations=2).run(g)
    return g
