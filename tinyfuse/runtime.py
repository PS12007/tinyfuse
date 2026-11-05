"""Compile generated C with the system compiler and load it with ctypes."""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .codegen.c_backend import GeneratedModule

CC = os.environ.get("TINYFUSE_CC", "gcc")
BASE_FLAGS = ["-O3", "-march=native", "-ffast-math", "-fopenmp-simd", "-std=gnu11",
              "-fPIC", "-shared"]


class CompileError(RuntimeError):
    pass


def cache_dir() -> Path:
    d = os.environ.get("TINYFUSE_CACHE")
    p = Path(d) if d else Path(tempfile.gettempdir()) / "tinyfuse_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


@functools.lru_cache(maxsize=None)
def native_vector_width() -> int:
    """SIMD width in floats that ``-march=native`` gives us."""
    try:
        out = subprocess.run([CC, "-march=native", "-dM", "-E", "-x", "c", "/dev/null"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 4
    if "__AVX512F__" in out:
        return 16
    if "__AVX__" in out:
        return 8
    return 4


def compile_shared(source: str, parallel: bool = False, extra_flags: Sequence[str] = ()) -> Path:
    flags = BASE_FLAGS + (["-fopenmp"] if parallel else []) + list(extra_flags)
    key = hashlib.sha256((CC + " ".join(flags) + "\n" + source).encode()).hexdigest()[:24]
    d = cache_dir()
    so = d / f"tf_{key}.so"
    if so.exists():
        return so
    c = d / f"tf_{key}.c"
    c.write_text(source)
    tmp = d / f"tf_{key}.{os.getpid()}.so.tmp"
    r = subprocess.run([CC, *flags, str(c), "-o", str(tmp), "-lm"], capture_output=True, text=True)
    if r.returncode != 0:
        raise CompileError(f"C compilation failed:\n{r.stderr}\n(source in {c})")
    os.replace(tmp, so)  # atomic: safe with concurrent compiles
    return so


class CompiledModule:
    """A callable wrapping the native ``tf_run`` entry point.

    Not safe for concurrent calls from several threads: the workspace is
    shared between calls."""

    def __init__(self, gen: GeneratedModule, parallel: bool = False):
        self.gen = gen
        self.source = gen.source
        self.so_path = compile_shared(gen.source, parallel=parallel)
        self._lib = ctypes.CDLL(str(self.so_path))
        self._fn = self._lib.tf_run
        self._fn.argtypes = [ctypes.c_void_p]
        self._fn.restype = None
        g = gen.graph
        self.input_shapes = [tuple(i.shape) for i in g.inputs]
        self.input_names = [i.name for i in g.inputs]
        self.output_shapes = [tuple(o.shape) for o in g.outputs]
        self._consts = [np.asarray(c, dtype=np.float32, order="C") for c in gen.consts]
        self._ws = np.zeros(max(gen.workspace_floats, 16), dtype=np.float32)
        n = gen.n_inputs + gen.n_outputs + len(self._consts) + 1
        self._ptrs = (ctypes.c_void_p * n)()
        base = gen.n_inputs + gen.n_outputs
        for i, c in enumerate(self._consts):
            self._ptrs[base + i] = c.ctypes.data
        self._ptrs[n - 1] = self._ws.ctypes.data

    def profile(self, reset: bool = False):
        """Per-kernel cumulative seconds (only for modules built with
        ``CodegenOptions(profile=True)``): list of (kernel name, comment, s)."""
        if not hasattr(self._lib, "tf_profile_data"):
            raise RuntimeError("module was not built with profile=True")
        fn = self._lib.tf_profile_data
        fn.restype = ctypes.POINTER(ctypes.c_double)
        data = fn()
        ks = self.gen.kernels
        res = [(k.name, k.comment, data[i]) for i, k in enumerate(ks)]
        if reset:
            for i in range(len(ks)):
                data[i] = 0.0
        return res

    def __call__(self, *args, out: Optional[List[np.ndarray]] = None):
        if len(args) != len(self.input_shapes):
            raise TypeError(f"expected {len(self.input_shapes)} inputs, got {len(args)}")
        keep = []
        for i, (a, shape) in enumerate(zip(args, self.input_shapes)):
            a = np.asarray(a, dtype=np.float32, order="C")
            if a.shape != shape:
                raise ValueError(f"input '{self.input_names[i]}' expects shape {shape}, "
                                 f"got {a.shape}")
            keep.append(a)
            self._ptrs[i] = a.ctypes.data
        if out is None:
            out = [np.empty(s, dtype=np.float32) for s in self.output_shapes]
        for i, (o, shape) in enumerate(zip(out, self.output_shapes)):
            if o.shape != shape or o.dtype != np.float32 or not o.flags.c_contiguous:
                raise ValueError(f"output {i} must be a C-contiguous float32 array of {shape}")
            self._ptrs[self.gen.n_inputs + i] = o.ctypes.data
        self._fn(self._ptrs)
        return out[0] if len(out) == 1 else tuple(out)
