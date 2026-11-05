"""Optimisation passes.

Each pass is a function ``pass_fn(graph) -> bool`` that rewrites the graph in
place and returns whether anything changed.  :class:`PassManager` runs a
pipeline, optionally verifying the IR after every pass.
"""

from .constant_folding import constant_folding
from .cse import common_subexpression_elimination
from .dce import dead_code_elimination
from .fusion import elementwise_fusion
from .matmul_fusion import matmul_epilogue_fusion
from .manager import PIPELINES, PassManager, optimize

__all__ = [
    "constant_folding", "common_subexpression_elimination", "dead_code_elimination",
    "elementwise_fusion", "matmul_epilogue_fusion", "PassManager", "PIPELINES", "optimize",
]
