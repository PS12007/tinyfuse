"""tinyfuse: a small tracing ML compiler that emits fused, tiled C."""

from .frontend import (Tensor, add, constant, div, exp, gelu, layernorm, matmul, maximum,
                       mul, neg, relu, reshape, sigmoid, softmax, sub, tanh, trace, transpose)
from .interpreter import run_graph
from .ir import Graph, IRError, Node, format_graph, verify

__version__ = "0.1.0"
from .compiler import CompiledFunction, compile, compile_graph
from .passes import optimize
