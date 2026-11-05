"""tinyfuse: a small tracing ML compiler that emits fused, tiled C."""

from .frontend import (Tensor, add, constant, div, exp, gelu, layernorm, matmul, max, maximum,
                       mean, mul, neg, relu, reshape, rsqrt, sigmoid, softmax, sqrt, sub, sum,
                       tanh, trace, transpose)
from .interpreter import run_graph
from .ir import Graph, IRError, Node, format_graph, verify

__version__ = "0.1.0"
from .compiler import CompiledFunction, compile, compile_graph
from .passes import optimize
