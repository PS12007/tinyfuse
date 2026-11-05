"""High-level driver: trace -> optimise -> generate C -> compile -> load."""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Union

from .codegen.c_backend import CodegenOptions, generate_c
from .frontend import trace
from .ir import Graph, verify
from .passes import optimize
from .runtime import CompiledModule, native_vector_width


class CompiledFunction:
    def __init__(self, graph: Graph, optimized: Graph, module: CompiledModule):
        self.graph = graph            # as traced
        self.optimized = optimized    # after passes
        self.module = module

    @property
    def source(self) -> str:
        return self.module.source

    def __call__(self, *args, **kw):
        return self.module(*args, **kw)


def compile_graph(g: Graph, pipeline: str = "fused", options: Optional[CodegenOptions] = None,
                  tuning=None) -> CompiledFunction:
    """Optimise and compile an already-traced graph.

    ``pipeline`` is one of ``"none"``, ``"unfused"``, ``"fused"`` (see
    :data:`tinyfuse.passes.PIPELINES`).  ``tuning`` may be an
    :class:`tinyfuse.autotune.TuningDB` whose best tiles are used."""
    verify(g)
    opt = optimize(g, pipeline)
    options = options or CodegenOptions(vector_width=native_vector_width())
    if tuning is not None:
        tiles = dict(options.tiles)
        for shape, tile in tuning.tiles_for(opt, options).items():
            if shape not in tiles and shape[:4] not in tiles:   # explicit tiles win
                tiles[shape] = tile
        options = dataclasses.replace(options, tiles=tiles)
    gen = generate_c(opt, options)
    return CompiledFunction(g, opt, CompiledModule(gen, parallel=options.parallel))


def compile(fn: Union[Callable, Graph], *shapes, pipeline: str = "fused",
            options: Optional[CodegenOptions] = None, tuning=None, **named_shapes):
    """Trace ``fn`` with the given input shapes and compile it to native code."""
    g = fn if isinstance(fn, Graph) else trace(fn, *shapes, **named_shapes)
    return compile_graph(g, pipeline=pipeline, options=options, tuning=tuning)
