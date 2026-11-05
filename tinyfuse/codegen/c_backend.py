"""C code generation.

The whole graph becomes one C translation unit:

* one ``static`` kernel function per (non-alias) node, specialised to the
  node's static shapes (all sizes are compile-time constants, so GCC can
  fully unroll and vectorise);
* an entry point ``void tf_run(float* const* bufs)`` that calls the kernels
  in order on pointers into the inputs / outputs / constants / workspace.

``bufs`` layout: ``[inputs..., outputs..., constants..., workspace]``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ir import Graph, Node
from ..ops import (ELEMENTWISE_ARITY, broadcast_strides, contiguous_strides,
                   matmul_operand_shapes, matmul_shape, numel)
from .memory import MemoryPlan, plan_memory


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MatmulTile:
    """Blocking parameters for the matmul kernel.

    ``mr`` x ``nv*VW`` is the register-blocked micro-tile (VW = SIMD width in
    floats); ``mb``/``nb``/``kb`` are the cache-blocking tile sizes."""
    mb: int = 96
    nb: int = 256
    kb: int = 256
    mr: int = 6
    nv: int = 2

    def key(self):
        return (self.mb, self.nb, self.kb, self.mr, self.nv)


@dataclass
class CodegenOptions:
    vector_width: int = 16          # floats per SIMD register (16 = AVX-512)
    default_tile: Optional[MatmulTile] = None
    # (batch, M, N, K[, epilogue signature]) -> MatmulTile; consulted before default_tile
    tiles: Dict[Tuple[int, int, int, int], MatmulTile] = field(default_factory=dict)
    parallel: bool = False          # OpenMP parallel loops
    profile: bool = False           # time every kernel call (see CompiledModule.profile)
    # upper bound on OpenMP threads per matmul (sizes per-thread packing scratch)
    max_threads: int = field(default_factory=lambda: os.cpu_count() or 1)

    def tile_for(self, batch, m, n, k, epilogue: str = "") -> MatmulTile:
        t = self.tiles.get((batch, m, n, k, epilogue)) or self.tiles.get((batch, m, n, k))
        if t is None:
            t = self.default_tile or default_tile_for(self.vector_width)
        return t


def default_tile_for(vw: int) -> MatmulTile:
    if vw >= 16:  # AVX-512: 6x4 zmm accumulators = 24 of 32 registers
        return MatmulTile(mb=48, nb=512, kb=256, mr=6, nv=4)
    if vw == 8:   # AVX2: 6x2 ymm accumulators = 12 of 16 registers
        return MatmulTile(mb=96, nb=256, kb=256, mr=6, nv=2)
    return MatmulTile(mb=64, nb=128, kb=256, mr=4, nv=2)


def epilogue_signature(n: Node) -> str:
    """Text form of a matmul node's epilogue ("" if none); part of tuning keys."""
    from ..ir import format_program
    ep = n.attrs.get("epilogue") if n.op == "fused_matmul" else None
    return format_program(ep) if ep else ""


def clamp_tile(t: MatmulTile, m: int, n: int, k: int, vw: int) -> MatmulTile:
    """Shrink cache tiles to the problem and round them to micro-tile multiples."""
    nr = t.nv * vw
    kb = max(1, min(t.kb, k))
    nb = max(nr, min(t.nb, -(-n // nr) * nr)) // nr * nr
    mb = max(t.mr, min(t.mb, -(-m // t.mr) * t.mr)) // t.mr * t.mr
    return MatmulTile(mb, nb, kb, t.mr, t.nv)


# --------------------------------------------------------------------------
# Scalar expression helpers (shared by elementwise kernels and epilogues)
# --------------------------------------------------------------------------

def c_float(v: float) -> str:
    v = float(np.float32(v))
    if math.isnan(v):
        return "NAN"
    if math.isinf(v):
        return "INFINITY" if v > 0 else "(-INFINITY)"
    r = repr(v)
    if "e" not in r and "." not in r:
        r += ".0"
    return r + "f"


def c_scalar_op(op: str, a: List[str]) -> str:
    if op == "add":
        return f"({a[0]} + {a[1]})"
    if op == "sub":
        return f"({a[0]} - {a[1]})"
    if op == "mul":
        return f"({a[0]} * {a[1]})"
    if op == "div":
        return f"({a[0]} / {a[1]})"
    if op == "maximum":
        return f"fmaxf({a[0]}, {a[1]})"
    if op == "relu":
        return f"fmaxf({a[0]}, 0.0f)"
    if op == "neg":
        return f"(-{a[0]})"
    if op == "exp":
        return f"expf({a[0]})"
    if op == "tanh":
        return f"tanhf({a[0]})"
    if op == "sigmoid":
        # exponent clamped: libmvec's expf takes a slow scalar path on underflow
        return f"(1.0f / (1.0f + expf(fmaxf(fminf(-{a[0]}, 80.0f), -80.0f))))"
    if op == "sqrt":
        return f"sqrtf({a[0]})"
    if op == "rsqrt":
        return f"(1.0f / sqrtf({a[0]}))"
    if op == "gelu":
        # 0.5*x*(1+tanh(y)) == x * sigmoid(2y): one expf instead of tanhf (faster
        # with libmvec).  The exponent is clamped to [-80, 80]: that never
        # overflows, and avoids libmvec's very slow scalar path for underflow.
        x = a[0]
        return (f"({x} / (1.0f + expf(fmaxf(fminf(-1.5957691216f * "
                f"({x} + 0.044715f * {x} * {x} * {x}), 80.0f), -80.0f))))")
    raise NotImplementedError(op)


def emit_program(program, in_exprs: Sequence[str], acc_expr: Optional[str] = None,
                 prefix: str = "t") -> Tuple[List[str], str]:
    """Return C statements evaluating ``program`` and the result expression."""
    stmts = []

    def opnd(o):
        if o[0] == "in":
            return in_exprs[o[1]]
        if o[0] == "t":
            return f"{prefix}{o[1]}"
        if o[0] == "c":
            return c_float(o[1])
        if o[0] == "acc":
            assert acc_expr is not None
            return acc_expr
        raise ValueError(o)

    for j, (op, operands) in enumerate(program):
        stmts.append(f"const float {prefix}{j} = {c_scalar_op(op, [opnd(o) for o in operands])};")
    return stmts, f"{prefix}{len(program) - 1}"


def node_program(n: Node):
    """Elementwise program for a primitive elementwise node or fused node."""
    if n.op == "fused_elementwise":
        return n.attrs["program"]
    assert n.op in ELEMENTWISE_ARITY
    return ((n.op, tuple(("in", i) for i in range(len(n.inputs)))),)


# --------------------------------------------------------------------------
# Loop-nest helpers
# --------------------------------------------------------------------------

def merge_dims(shape: Sequence[int], strides_list: Sequence[Sequence[int]]):
    """Collapse adjacent dims that are contiguous for every operand and drop
    size-1 dims.  Returns (shape, strides_list) with fewer dims."""
    dims = [(d, [s[i] for s in strides_list]) for i, d in enumerate(shape) if d != 1]
    if not dims:
        return [1], [[0] for _ in strides_list]
    merged = [dims[-1]]
    for d, st in reversed(dims[:-1]):
        id_, ist = merged[0]
        if all(st[k] == ist[k] * id_ for k in range(len(st))):
            merged[0] = (d * id_, ist)
        else:
            merged.insert(0, (d, st))
    shape = [d for d, _ in merged]
    strides = [[st[k] for _, st in merged] for k in range(len(strides_list))]
    return shape, strides


def _batch_offset(strides, batch_dims) -> str:
    """C expression for the element offset of flat batch index ``bt``."""
    terms = []
    rem = 1
    for d in range(len(batch_dims) - 1, -1, -1):
        if strides[d]:
            terms.append(f"((bt / {rem}L) % {batch_dims[d]}L) * {strides[d]}L")
        rem *= batch_dims[d]
    return " + ".join(terms) if terms else "0"


def _index(vars_, strides) -> str:
    terms = []
    for v, s in zip(vars_, strides):
        if s == 0:
            continue
        terms.append(v if s == 1 else f"{v}*{s}L")
    return " + ".join(terms) if terms else "0"


class Writer:
    def __init__(self):
        self.lines: List[str] = []
        self.ind = 0

    def __call__(self, line: str = ""):
        self.lines.append("    " * self.ind + line if line else "")

    def open(self, line: str):
        self(line + " {")
        self.ind += 1

    def close(self, suffix: str = ""):
        self.ind -= 1
        self("}" + suffix)

    def text(self):
        return "\n".join(self.lines)


PRELUDE = r"""/* generated by tinyfuse -- do not edit */
#include <math.h>
#include <float.h>
#include <string.h>
#define TF_MIN(a, b) ((a) < (b) ? (a) : (b))
typedef float tf_vf __attribute__((vector_size(TF_VW * 4), aligned(4)));
"""


# --------------------------------------------------------------------------
# Kernel generators
# --------------------------------------------------------------------------

@dataclass
class Kernel:
    name: str
    source: str
    args: List[str]          # C expressions passed at the call site
    scratch: int = 0         # floats of scratch memory required
    comment: str = ""


class CodeGen:
    def __init__(self, g: Graph, opts: CodegenOptions, plan: MemoryPlan):
        self.g = g
        self.opts = opts
        self.plan = plan
        self.vw = opts.vector_width

    # buffer name for a node's value
    def buf(self, n: Node) -> str:
        return f"s{self.plan.node_storage[n.id].sid}"

    def omp_for(self, w: Writer, work: int):
        if self.opts.parallel and work >= (1 << 15):
            w("#pragma omp parallel for schedule(static)")

    # ---- elementwise ------------------------------------------------------
    def gen_elementwise(self, n: Node, name: str) -> Kernel:
        out = n.shape
        program = node_program(n)
        strides = [list(broadcast_strides(i.shape, out)) for i in n.inputs]
        ostr = list(broadcast_strides(out, out))
        shape, all_strides = merge_dims(out, strides + [ostr])
        in_strides, o_strides = all_strides[:-1], all_strides[-1]
        nd = len(shape)
        vars_ = [f"i{d}" for d in range(nd)]
        w = Writer()
        params = [f"const float* restrict in{k}" for k in range(len(n.inputs))] + \
                 ["float* restrict out"]
        w.open(f"static void {name}({', '.join(params)})")
        for d in range(nd):
            if d == 0 and nd > 1:
                self.omp_for(w, numel(out))
            if d == nd - 1:
                if nd == 1 and self.opts.parallel and numel(out) >= (1 << 15):
                    # a fully merged 1-D loop: split it across threads *and* SIMD
                    w("#pragma omp parallel for simd schedule(static)")
                else:
                    w("#pragma omp simd")
            w.open(f"for (long {vars_[d]} = 0; {vars_[d]} < {shape[d]}L; ++{vars_[d]})")
        in_exprs = [f"in{k}[{_index(vars_, in_strides[k])}]" for k in range(len(n.inputs))]
        stmts, res = emit_program(program, in_exprs)
        for s in stmts:
            w(s)
        w(f"out[{_index(vars_, o_strides)}] = {res};")
        for _ in range(nd):
            w.close()
        w.close()
        return Kernel(name, w.text(), [self.buf(i) for i in n.inputs] + [self.buf(n)])

    # ---- matmul (+ fused epilogue) -----------------------------------------
    @staticmethod
    def _mm_pack_b(w: Writer, p_expr: str):
        """Pack B[k0:k0+kb, panel p] into a zero-padded [kb][NR] panel."""
        w.open("")
        w(f"const int p = {p_expr};")
        w("float* restrict dst = Bp + (long)p * kb * NR;")
        w("const int jj = j0 + p * NR;")
        w("const int wdt = TF_MIN(NR, N - jj);")
        w.open("for (int k = 0; k < kb; ++k)")
        w("const float* restrict src = Bb + (long)(k0 + k) * SBK + (long)jj * SBN;")
        w.open("if (wdt == NR)")
        w("for (int c = 0; c < NR; ++c) dst[k * NR + c] = src[c * SBN];")
        w.close()
        w.open("else")
        w("for (int c = 0; c < wdt; ++c) dst[k * NR + c] = src[c * SBN];")
        w("for (int c = wdt; c < NR; ++c) dst[k * NR + c] = 0.0f;")
        w.close()
        w.close()
        w.close()

    @staticmethod
    def _mm_pack_a(w: Writer):
        """Pack A[i0:i0+mb, k0:k0+kb] into mp zero-padded [kb][MR] panels at Ap."""
        w.open("for (int q = 0; q < mp; ++q)")
        w("float* restrict dst = Ap + (long)q * kb * MR;")
        w("const int ii = i0 + q * MR;")
        w("const int hgt = TF_MIN(MR, M - ii);")
        w.open("for (int r = 0; r < MR; ++r)")
        w.open("if (r < hgt)")
        w("const float* restrict src = Ab + (long)(ii + r) * SAM + (long)k0 * SAK;")
        w("for (int k = 0; k < kb; ++k) dst[k * MR + r] = src[k * SAK];")
        w.close()
        w.open("else")
        w("for (int k = 0; k < kb; ++k) dst[k * MR + r] = 0.0f;")
        w.close()
        w.close()
        w.close()

    @staticmethod
    def _mm_microtile(w: Writer):
        """C[ii:ii+MR, jj:jj+NR] (+)= Ap panel q x Bp panel p, accumulators in
        registers; partial tiles go through a small temporary."""
        w("const int ii = i0 + q * MR;")
        w("const int hgt = TF_MIN(MR, M - ii);")
        w("const float* restrict ap = Ap + (long)q * kb * MR;")
        w("const int jj = j0 + p * NR;")
        w("const int wdt = TF_MIN(NR, N - jj);")
        w("const float* restrict bp = Bp + (long)p * kb * NR;")
        w("const int full = (hgt == MR) && (wdt == NR);")
        w("tf_vf acc[MR][NV];")
        w("float tmp[MR * NR] __attribute__((aligned(64)));")
        w.open("if (k0 == 0)")
        w("#pragma GCC unroll 16")
        w("for (int r = 0; r < MR; ++r)")
        w("    for (int v = 0; v < NV; ++v) acc[r][v] = (tf_vf){0};")
        w.close()
        w.open("else if (full)")
        w("#pragma GCC unroll 16")
        w("for (int r = 0; r < MR; ++r)")
        w("    for (int v = 0; v < NV; ++v) acc[r][v] = *(const tf_vf*)(Cb + (long)(ii + r) * SCM + jj + v * VW);")
        w.close()
        w.open("else")
        w("memset(tmp, 0, sizeof(tmp));")
        w("for (int r = 0; r < hgt; ++r)")
        w("    for (int c = 0; c < wdt; ++c) tmp[r * NR + c] = Cb[(long)(ii + r) * SCM + jj + c];")
        w("for (int r = 0; r < MR; ++r)")
        w("    for (int v = 0; v < NV; ++v) acc[r][v] = *(const tf_vf*)(tmp + r * NR + v * VW);")
        w.close()
        w.open("for (int k = 0; k < kb; ++k)")
        w("tf_vf bv[NV];")
        w("#pragma GCC unroll 8")
        w("for (int v = 0; v < NV; ++v) bv[v] = *(const tf_vf*)(bp + k * NR + v * VW);")
        w("#pragma GCC unroll 16")
        w.open("for (int r = 0; r < MR; ++r)")
        w("const float av = ap[k * MR + r];")
        w("#pragma GCC unroll 8")
        w("for (int v = 0; v < NV; ++v) acc[r][v] += av * bv[v];")
        w.close()
        w.close()
        w.open("if (full)")
        w("#pragma GCC unroll 16")
        w("for (int r = 0; r < MR; ++r)")
        w("    for (int v = 0; v < NV; ++v) *(tf_vf*)(Cb + (long)(ii + r) * SCM + jj + v * VW) = acc[r][v];")
        w.close()
        w.open("else")
        w("for (int r = 0; r < MR; ++r)")
        w("    for (int v = 0; v < NV; ++v) *(tf_vf*)(tmp + r * NR + v * VW) = acc[r][v];")
        w("for (int r = 0; r < hgt; ++r)")
        w("    for (int c = 0; c < wdt; ++c) Cb[(long)(ii + r) * SCM + jj + c] = tmp[r * NR + c];")
        w.close()

    @staticmethod
    def _mm_epilogue(w: Writer, epilogue, ex_strides, col_lo: str, col_hi: str):
        """Apply the epilogue to rows i0..i0+mb, columns [col_lo, col_hi)."""
        w.open("for (int r = 0; r < mb; ++r)")
        w("const long i = i0 + r;")
        w("float* restrict crow = Cb + i * SCM;")
        w("#pragma omp simd")
        w.open(f"for (long j = {col_lo}; j < {col_hi}; ++j)")
        in_exprs = [f"eb{k}[{_index(['i', 'j'], st[-2:])}]" for k, st in enumerate(ex_strides)]
        stmts, res = emit_program(epilogue, in_exprs, acc_expr="crow[j]")
        for s_ in stmts:
            w(s_)
        w(f"crow[j] = {res};")
        w.close()
        w.close()

    def _mm_serial_body(self, w, batch_dims, a_batch, b_batch, c_batch, ex_strides, epilogue):
        BATCH = numel(batch_dims)
        w("float* restrict Bp = scratch;")
        w("float* restrict Ap = scratch + KB * NB;")
        w.open(f"for (long bt = 0; bt < {BATCH}L; ++bt)")
        w(f"const float* restrict Ab = A + {_batch_offset(a_batch, batch_dims)};")
        w(f"const float* restrict Bb = B + {_batch_offset(b_batch, batch_dims)};")
        w(f"float* restrict Cb = C + {_batch_offset(c_batch, batch_dims)};")
        for k, st in enumerate(ex_strides):
            w(f"const float* restrict eb{k} = e{k} + {_batch_offset(st[:-2], batch_dims)};")
        w("(void)Ab; (void)Bb;")
        w.open("for (int j0 = 0; j0 < N; j0 += NB)")
        w("const int nb = TF_MIN(NB, N - j0);")
        w("const int np = (nb + NR - 1) / NR;")
        w.open("for (int k0 = 0; k0 < K; k0 += KB)")
        w("const int kb = TF_MIN(KB, K - k0);")
        w("const int last = (k0 + KB >= K); (void)last;")
        w.open("for (int pp = 0; pp < np; ++pp)")
        self._mm_pack_b(w, "pp")
        w.close()
        w.open("for (int i0 = 0; i0 < M; i0 += MB)")
        w("const int mb = TF_MIN(MB, M - i0);")
        w("const int mp = (mb + MR - 1) / MR;")
        self._mm_pack_a(w)
        w.open("for (int q = 0; q < mp; ++q)")
        w.open("for (int p = 0; p < np; ++p)")
        self._mm_microtile(w)
        w.close()
        w.close()
        if epilogue:
            # Applied once per finished MB x NB block, while it is still in L2.
            # (Measured equal to applying it per micro-tile, and it keeps the
            # micro-kernel loop body minimal.)
            w.open("if (last)")
            self._mm_epilogue(w, epilogue, ex_strides, "j0", "j0 + nb")
            w.close()
        w.close()  # i0
        w.close()  # k0
        w.close()  # j0
        w.close()  # bt
        w.close()  # fn

    def _mm_parallel_body(self, w, BATCH, batch_dims, a_batch, b_batch, c_batch, ex_strides,
                          epilogue, nthreads, grp):
        """Barrier-free OpenMP matmul.  A unit = (batch, column block, group of
        ``grp`` row blocks) is computed start to finish by one thread, which
        packs B once per K block for the whole group and A per row block, in
        a private scratch slice.  No synchronisation until the kernel ends."""
        w(f"const int nt = TF_MIN(omp_get_max_threads(), {nthreads});")
        w(f"const int NIB = (M + MB - 1) / MB, NJB = (N + NB - 1) / NB, GRP = {grp};")
        w("const int NG = (NIB + GRP - 1) / GRP;")
        w("#pragma omp parallel num_threads(nt)")
        w.open("")
        w("float* restrict Bp = scratch + (long)omp_get_thread_num() * (KB * NB + MB * KB);")
        w("float* restrict Ap = Bp + KB * NB;")
        w("#pragma omp for collapse(3) schedule(static)")
        w.open(f"for (long bt = 0; bt < {BATCH}L; ++bt)")
        w.open("for (int jb = 0; jb < NJB; ++jb)")
        w.open("for (int g = 0; g < NG; ++g)")
        w(f"const float* restrict Ab = A + {_batch_offset(a_batch, batch_dims)};")
        w(f"const float* restrict Bb = B + {_batch_offset(b_batch, batch_dims)};")
        w(f"float* restrict Cb = C + {_batch_offset(c_batch, batch_dims)};")
        for k, st in enumerate(ex_strides):
            w(f"const float* restrict eb{k} = e{k} + {_batch_offset(st[:-2], batch_dims)};")
        w("(void)Ab; (void)Bb;")
        w("const int j0 = jb * NB;")
        w("const int nb = TF_MIN(NB, N - j0), np = (nb + NR - 1) / NR;")
        w("const int ib_end = TF_MIN((g + 1) * GRP, NIB);")
        w.open("for (int k0 = 0; k0 < K; k0 += KB)")
        w("const int kb = TF_MIN(KB, K - k0);")
        w("const int last = (k0 + KB >= K); (void)last;")
        w.open("for (int pp = 0; pp < np; ++pp)")
        self._mm_pack_b(w, "pp")
        w.close()
        w.open("for (int ib = g * GRP; ib < ib_end; ++ib)")
        w("const int i0 = ib * MB;")
        w("const int mb = TF_MIN(MB, M - i0), mp = (mb + MR - 1) / MR;")
        self._mm_pack_a(w)
        w.open("for (int q = 0; q < mp; ++q)")
        w.open("for (int p = 0; p < np; ++p)")
        self._mm_microtile(w)
        w.close()
        w.close()
        if epilogue:
            w.open("if (last)")
            self._mm_epilogue(w, epilogue, ex_strides, "j0", "j0 + nb")
            w.close()
        w.close()  # ib
        w.close()  # k0
        w.close()  # g
        w.close()  # jb
        w.close()  # bt
        w.close()  # parallel region

    def gen_matmul(self, n: Node, name: str) -> Kernel:
        a, b = n.inputs[0], n.inputs[1]
        extra = n.inputs[2:]
        epilogue = n.attrs.get("epilogue") if n.op == "fused_matmul" else None
        sa, sb = matmul_operand_shapes(n)
        out = matmul_shape(sa, sb)                    # logical result shape
        c_perm = n.attrs.get("c_perm") if n.op == "fused_matmul" else None
        if c_perm:   # result stored permuted: stride of logical dim k in memory
            stored = list(contiguous_strides(n.shape))
            c_str = [stored[list(c_perm).index(k)] for k in range(len(out))]
        else:
            c_str = list(contiguous_strides(out))
        assert c_str[-1] == 1
        SCM = c_str[-2]
        c_batch = c_str[:-2]
        a_view, b_view = n.attrs.get("a_view"), n.attrs.get("b_view")
        a_str = a_view[1] if a_view else contiguous_strides(sa)
        b_str = b_view[1] if b_view else contiguous_strides(sb)
        M, K = sa[-2], sa[-1]
        N = sb[-1]
        batch_dims = out[:-2]
        BATCH = numel(batch_dims)
        SAM, SAK = a_str[-2], a_str[-1]
        SBK, SBN = b_str[-2], b_str[-1]
        a_batch = list(a_str[:-2])
        b_batch = list(b_str[:-2]) if len(sb) > 2 else [0] * len(batch_dims)
        vw = self.vw
        t = clamp_tile(self.opts.tile_for(BATCH, M, N, K, epilogue_signature(n)), M, N, K, vw)
        MB, NB, KB, MR, NV = t.key()
        NR = NV * vw
        par = self.opts.parallel
        nthreads = max(1, self.opts.max_threads) if par else 1
        if par:
            # barrier-free decomposition: units = (batch, column block, row block);
            # shrink NB until there are >= 2 units per thread (when N allows it)
            nib = -(-M // MB)
            while BATCH * nib * -(-N // NB) < 2 * nthreads and NB > NR:
                NB = max(NR, (NB // 2) // NR * NR)
            scratch = nthreads * (KB * NB + MB * KB)
        else:
            scratch = KB * NB + MB * KB

        ex_strides = [list(broadcast_strides(e.shape, out)) for e in extra]

        w = Writer()
        params = ["const float* restrict A", "const float* restrict B", "float* restrict C"]
        params += [f"const float* restrict e{k}" for k in range(len(extra))]
        params += ["float* restrict scratch"]
        w(f"/* {n.op} [{BATCH}x] {M}x{K} @ {K}x{N}  tile mb={MB} nb={NB} kb={KB} "
          f"mr={MR} nr={NR}{' parallel' if par else ''} */")
        w.open(f"static void {name}({', '.join(params)})")
        w(f"enum {{ M = {M}, N = {N}, K = {K}, MB = {MB}, NB = {NB}, KB = {KB}, "
          f"MR = {MR}, NV = {NV}, NR = {NR}, VW = {vw} }};")
        w(f"const long SAM = {SAM}L, SAK = {SAK}L, SBK = {SBK}L, SBN = {SBN}L, SCM = {SCM}L;  "
          f"/* operand / result strides (views) */")
        if par:
            # group row blocks per unit so B is packed once per group, keeping
            # >= 2 units per thread where possible
            nib, njb = -(-M // MB), -(-N // NB)
            grp = max(1, (BATCH * nib * njb) // (2 * nthreads))
            grp = min(grp, nib)
            self._mm_parallel_body(w, BATCH, batch_dims, a_batch, b_batch, c_batch, ex_strides,
                                   epilogue, nthreads, grp)
            w.close()  # fn
        else:
            self._mm_serial_body(w, batch_dims, a_batch, b_batch, c_batch, ex_strides, epilogue)
        args = [self.buf(a), self.buf(b), self.buf(n)] + [self.buf(e) for e in extra] + ["scratch"]
        return Kernel(name, w.text(), args, scratch=scratch)

    # ---- reductions (plain and with a fused elementwise prologue) -----------
    def gen_reduce(self, n: Node, name: str) -> Kernel:
        if n.op == "fused_reduce":
            kind, program, in_shape = n.attrs["kind"], n.attrs["program"], tuple(n.attrs["in_shape"])
        else:
            kind, in_shape = n.op[len("reduce_"):], tuple(n.inputs[0].shape)
            program = (("add", (("in", 0), ("c", 0.0))),)   # identity prologue
        L = in_shape[-1]
        outer = in_shape[:-1] or (1,)
        strides = [list(broadcast_strides(x.shape, in_shape)) for x in n.inputs]
        if len(in_shape) == 1:
            strides = [[0] + st for st in strides]
        inner = [st[-1] for st in strides]
        o_str = list(broadcast_strides(outer, outer))
        shape, all_st = merge_dims(outer, [st[:-1] for st in strides] + [o_str])
        in_outer, out_outer = all_st[:-1], all_st[-1]
        nd = len(shape)
        vars_ = [f"i{d}" for d in range(nd)]
        w = Writer()
        params = [f"const float* restrict in{k}" for k in range(len(n.inputs))] + \
                 ["float* restrict out"]
        w.open(f"static void {name}({', '.join(params)})")
        for d in range(nd):
            if d == 0:
                self.omp_for(w, numel(in_shape))
            w.open(f"for (long {vars_[d]} = 0; {vars_[d]} < {shape[d]}L; ++{vars_[d]})")
        bases = [f"in{k} + {_index(vars_, in_outer[k])}" for k in range(len(n.inputs))]
        for k, b in enumerate(bases):
            w(f"const float* restrict r{k} = {b};")
        init = "-FLT_MAX" if kind == "max" else "0.0f"
        red = "max" if kind == "max" else "+"
        w(f"float acc = {init};")
        w(f"#pragma omp simd reduction({red}:acc)")
        w.open(f"for (long j = 0; j < {L}L; ++j)")
        in_exprs = [f"r{k}[{_index(['j'], [inner[k]])}]" for k in range(len(n.inputs))]
        stmts, res = emit_program(program, in_exprs)
        for s_ in stmts:
            w(s_)
        w(f"acc = fmaxf(acc, {res});" if kind == "max" else f"acc += {res};")
        w.close()
        scale = f" * (1.0f / {L}.0f)" if kind == "mean" else ""
        w(f"out[{_index(vars_, out_outer)}] = acc{scale};")
        for _ in range(nd):
            w.close()
        w.close()
        return Kernel(name, w.text(), [self.buf(x) for x in n.inputs] + [self.buf(n)])

    # ---- softmax ------------------------------------------------------------
    def gen_softmax(self, n: Node, name: str) -> Kernel:
        L = n.shape[-1]
        R = numel(n.shape) // L
        w = Writer()
        w.open(f"static void {name}(const float* restrict x, float* restrict y)")
        self.omp_for(w, numel(n.shape))
        w.open(f"for (long r = 0; r < {R}L; ++r)")
        w(f"const float* restrict xr = x + r * {L}L;")
        w(f"float* restrict yr = y + r * {L}L;")
        w("float m = -FLT_MAX;")
        w("#pragma omp simd reduction(max:m)")
        w(f"for (long j = 0; j < {L}L; ++j) m = fmaxf(m, xr[j]);")
        w("float s = 0.0f;")
        w("#pragma omp simd reduction(+:s)")
        w(f"for (long j = 0; j < {L}L; ++j) {{ const float e = expf(xr[j] - m); yr[j] = e; s += e; }}")
        w("const float inv = 1.0f / s;")
        w("#pragma omp simd")
        w(f"for (long j = 0; j < {L}L; ++j) yr[j] *= inv;")
        w.close()
        w.close()
        return Kernel(name, w.text(), [self.buf(n.inputs[0]), self.buf(n)])

    # ---- layernorm ------------------------------------------------------------
    def gen_layernorm(self, n: Node, name: str) -> Kernel:
        L = n.shape[-1]
        R = numel(n.shape) // L
        eps = c_float(n.attrs["eps"])
        w = Writer()
        w.open(f"static void {name}(const float* restrict x, const float* restrict g, "
               f"const float* restrict b, float* restrict y)")
        self.omp_for(w, numel(n.shape))
        w.open(f"for (long r = 0; r < {R}L; ++r)")
        w(f"const float* restrict xr = x + r * {L}L;")
        w(f"float* restrict yr = y + r * {L}L;")
        w("float s = 0.0f;")
        w("#pragma omp simd reduction(+:s)")
        w(f"for (long j = 0; j < {L}L; ++j) s += xr[j];")
        w(f"const float mean = s * (1.0f / {L}.0f);")
        w("float v = 0.0f;")
        w("#pragma omp simd reduction(+:v)")
        w(f"for (long j = 0; j < {L}L; ++j) {{ const float d = xr[j] - mean; v += d * d; }}")
        w(f"const float rstd = 1.0f / sqrtf(v * (1.0f / {L}.0f) + {eps});")
        w("#pragma omp simd")
        w(f"for (long j = 0; j < {L}L; ++j) yr[j] = (xr[j] - mean) * rstd * g[j] + b[j];")
        w.close()
        w.close()
        args = [self.buf(i) for i in n.inputs] + [self.buf(n)]
        return Kernel(name, w.text(), args)

    # ---- transpose -------------------------------------------------------------
    def gen_transpose(self, n: Node, name: str) -> Kernel:
        x = n.inputs[0]
        perm = n.attrs["perm"]
        in_c = list(broadcast_strides(x.shape, x.shape))
        istr = [in_c[p] for p in perm]                 # input stride per output dim
        ostr = list(broadcast_strides(n.shape, n.shape))
        shape, (istr, ostr) = merge_dims(n.shape, [istr, ostr])
        nd = len(shape)
        w = Writer()
        w.open(f"static void {name}(const float* restrict x, float* restrict y)")
        vars_ = [f"i{d}" for d in range(nd)]
        # the output dim that is contiguous in the input (if not the last one)
        tdim = next((d for d in range(nd) if istr[d] == 1), None)
        if nd >= 2 and tdim is not None and tdim != nd - 1:
            T = 32
            outer = [d for d in range(nd) if d not in (tdim, nd - 1)]
            for d in outer:
                w.open(f"for (long {vars_[d]} = 0; {vars_[d]} < {shape[d]}L; ++{vars_[d]})")
            a, bb = vars_[tdim], vars_[nd - 1]
            w.open(f"for (long {a}0 = 0; {a}0 < {shape[tdim]}L; {a}0 += {T})")
            w.open(f"for (long {bb}0 = 0; {bb}0 < {shape[nd - 1]}L; {bb}0 += {T})")
            w.open(f"for (long {a} = {a}0; {a} < TF_MIN({a}0 + {T}, {shape[tdim]}L); ++{a})")
            w("#pragma omp simd")
            w.open(f"for (long {bb} = {bb}0; {bb} < TF_MIN({bb}0 + {T}, {shape[nd - 1]}L); ++{bb})")
            w(f"y[{_index(vars_, ostr)}] = x[{_index(vars_, istr)}];")
            for _ in range(4 + len(outer)):
                w.close()
        else:
            for d in range(nd):
                if d == nd - 1:
                    w("#pragma omp simd")
                w.open(f"for (long {vars_[d]} = 0; {vars_[d]} < {shape[d]}L; ++{vars_[d]})")
            w(f"y[{_index(vars_, ostr)}] = x[{_index(vars_, istr)}];")
            for _ in range(nd):
                w.close()
        w.close()
        return Kernel(name, w.text(), [self.buf(x), self.buf(n)])

    def gen(self, n: Node, name: str) -> Optional[Kernel]:
        if n.op in ("input", "const", "reshape"):
            return None
        if n.op in ELEMENTWISE_ARITY or n.op == "fused_elementwise":
            return self.gen_elementwise(n, name)
        if n.op in ("matmul", "fused_matmul"):
            return self.gen_matmul(n, name)
        if n.op == "softmax":
            return self.gen_softmax(n, name)
        if n.op.startswith("reduce_") or n.op == "fused_reduce":
            return self.gen_reduce(n, name)
        if n.op == "layernorm":
            return self.gen_layernorm(n, name)
        if n.op == "transpose":
            return self.gen_transpose(n, name)
        raise NotImplementedError(f"no C lowering for op '{n.op}'")


# --------------------------------------------------------------------------
# Module generation
# --------------------------------------------------------------------------

@dataclass
class GeneratedModule:
    source: str
    graph: Graph
    plan: MemoryPlan
    n_inputs: int
    n_outputs: int
    consts: List[np.ndarray]
    workspace_floats: int
    kernels: List[Kernel]


def generate_c(g: Graph, opts: Optional[CodegenOptions] = None,
               reuse_memory: bool = True) -> GeneratedModule:
    opts = opts or CodegenOptions()
    plan = plan_memory(g, reuse=reuse_memory)
    cg = CodeGen(g, opts, plan)
    kernels: List[Kernel] = []
    for idx, n in enumerate(g.nodes):
        k = cg.gen(n, f"k{len(kernels)}_{n.op}")
        if k is not None:
            k.comment = f"node {idx}: {n.op} -> {list(n.shape)}"
            kernels.append(k)

    n_in, n_out, n_const = len(g.inputs), len(g.outputs), len(plan.consts)
    scratch = max((k.scratch for k in kernels), default=0)
    scratch = (scratch + 15) // 16 * 16
    ws_total = plan.ws_size + scratch

    w = Writer()
    w("void tf_run(float* const* bufs)")
    w("{")
    w.ind += 1
    ws_index = n_in + n_out + n_const
    w(f"float* const ws = bufs[{ws_index}];")
    w(f"float* const scratch = ws + {plan.ws_size}L; (void)scratch;")
    for st in plan.storages:
        if st.kind == "input":
            w(f"float* const s{st.sid} = bufs[{st.index}];")
        elif st.kind == "output":
            w(f"float* const s{st.sid} = bufs[{n_in + st.index}];")
        elif st.kind == "const":
            w(f"float* const s{st.sid} = bufs[{n_in + n_out + st.index}];")
        else:
            w(f"float* const s{st.sid} = ws + {st.offset}L;")
    for i, k in enumerate(kernels):
        if opts.profile:
            w("{ struct timespec t0_, t1_; clock_gettime(CLOCK_MONOTONIC, &t0_);")
        w(f"{k.name}({', '.join(k.args)});  /* {k.comment} */")
        if opts.profile:
            w("clock_gettime(CLOCK_MONOTONIC, &t1_);")
            w(f"tf_prof[{i}] += (t1_.tv_sec - t0_.tv_sec) + 1e-9 * (t1_.tv_nsec - t0_.tv_nsec); }}")
    for oi, st in plan.output_copies:
        w(f"memcpy(bufs[{n_in + oi}], s{st.sid}, sizeof(float) * {st.size}L);")
    w.ind -= 1
    w("}")

    prof = ""
    if opts.profile:
        prof = (f"#include <time.h>\nstatic double tf_prof[{max(len(kernels), 1)}];\n"
                "double* tf_profile_data(void) { return tf_prof; }\n")
    omp = "#include <omp.h>\n" if opts.parallel else ""
    src = (f"#define TF_VW {opts.vector_width}\n" + PRELUDE + omp + prof + "\n" +
           "\n\n".join(k.source for k in kernels) + "\n\n" + w.text() + "\n")
    consts = [c.attrs["value"] for c in plan.consts]
    return GeneratedModule(src, g, plan, n_in, n_out, consts, ws_total, kernels)
