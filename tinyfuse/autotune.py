"""Matmul tile-size autotuner.

For each matmul shape ``(batch, M, N, K)`` the tuner compiles a standalone
matmul kernel for a set of candidate :class:`MatmulTile` configurations,
times each one, and records the fastest in a JSON :class:`TuningDB`.
``tinyfuse.compile(..., tuning=db)`` then uses the recorded tile for every
matmul / fused_matmul with that shape.

Search space (pruned):

* micro-tile ``mr x nv`` vectors, kept within the register file:
  ``mr*nv`` accumulators + ``nv`` B vectors + 1 broadcast <= #SIMD registers;
* cache tiles ``mb`` (multiple of ``mr``), ``nb`` (multiple of ``nv*VW``), ``kb``,
  clamped to the problem size and de-duplicated after clamping.

Timing uses the minimum over several repetitions (robust to noisy
neighbours), and the default tile is always a candidate, so tuning can
never pick something slower than the default *as measured*.

Usage::

    python -m tinyfuse.autotune --shape 256x512x2048 --shape 64x256x1024
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .codegen.c_backend import CodegenOptions, MatmulTile, clamp_tile, default_tile_for
from .ir import Graph
from .ops import numel

Key = Tuple  # (batch, M, N, K) or (batch, M, N, K, epilogue-signature)


def cpu_fingerprint() -> str:
    model = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return hashlib.sha1(model.encode()).hexdigest()[:10]


def node_key(n) -> Key:
    """Tuning key of a matmul / fused_matmul node.  The epilogue is part of
    the key because its cost depends on the micro-tile shape."""
    from .codegen.c_backend import epilogue_signature
    from .ops import matmul_operand_shapes
    sa, sb = matmul_operand_shapes(n)
    return (numel(n.shape[:-2]), sa[-2], sb[-1], sa[-1], epilogue_signature(n))


def matmul_nodes(g: Graph) -> List[Tuple[Key, object]]:
    """One representative node per distinct tuning key, in graph order."""
    out, seen = [], set()
    for n in g.nodes:
        if n.op in ("matmul", "fused_matmul"):
            k = node_key(n)
            if k not in seen:
                seen.add(k)
                out.append((k, n))
    return out


def matmul_shapes(g: Graph) -> List[Key]:
    return [k for k, _ in matmul_nodes(g)]


def candidate_tiles(m: int, n: int, k: int, vw: int, max_candidates: int = 32,
                    seed: int = 0) -> List[MatmulTile]:
    regs = 32 if vw >= 16 else 16
    micro = [(mr, nv) for mr in (4, 6, 8, 12) for nv in (1, 2, 3, 4)
             if mr * nv + nv + 1 <= regs]
    cands = set()
    for (mr, nv), mbm, nb, kb in itertools.product(micro, (4, 8, 16, 32), (128, 256, 512, 1024),
                                                   (64, 128, 256, 512)):
        cands.add(clamp_tile(MatmulTile(mr * mbm, nb, kb, mr, nv), m, n, k, vw))
    cands = sorted(cands, key=lambda t: t.key())
    default = clamp_tile(default_tile_for(vw), m, n, k, vw)
    rnd = random.Random(seed)
    rnd.shuffle(cands)
    picked = [default] + [c for c in cands if c != default][:max_candidates - 1]
    return picked


def _time_min(fn, args, reps: int, min_time: float) -> float:
    fn(*args)
    best = float("inf")
    t_end = time.perf_counter() + min_time
    r = 0
    while r < reps or time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
        r += 1
        if r > 50 * reps:
            break
    return best


def _subgraph(node) -> Graph:
    """A graph containing just ``node`` with fresh inputs for its operands."""
    from .ir import Node
    g = Graph("tune")
    ins = [g.add_input(x.shape, f"in{i}") for i, x in enumerate(node.inputs)]
    n = Node(node.op, ins, dict(node.attrs))
    g.nodes.append(n)
    g.outputs = [n]
    return g


def tune_node(node, vw: Optional[int] = None, max_candidates: int = 32, reps: int = 5,
              min_time: float = 0.05, confirm_top: int = 4, confirm_time: float = 0.5,
              min_gain: float = 0.03, verbose: bool = False,
              seed: int = 0) -> Tuple[MatmulTile, float, List[dict]]:
    """Search tiles for one matmul / fused_matmul node (with its epilogue).

    Phase 1 screens every candidate (min of a short burst).  Phase 2 re-times
    the ``confirm_top`` fastest plus the default *interleaved* (one call each
    per round, median over rounds), so slow drifts in machine load cannot
    decide the winner, and keeps the default unless a challenger beats it by
    more than ``min_gain``.  Returns (best tile, GFLOP/s, phase-1 results);
    ``results[0]`` is always the default tile."""
    from .compiler import compile_graph
    from .interpreter import run_graph
    from .runtime import native_vector_width
    vw = vw or native_vector_width()
    batch, m, n, k = node_key(node)[:4]
    g = _subgraph(node)
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal(i.shape).astype(np.float32) for i in g.inputs]
    ref = run_graph(g, xs)[0]
    flops = 2.0 * batch * m * n * k
    results, fns = [], {}
    for t in candidate_tiles(m, n, k, vw, max_candidates, seed):
        f = compile_graph(g, "none", CodegenOptions(vector_width=vw, default_tile=t))
        if not np.allclose(f(*xs), ref, rtol=1e-3, atol=1e-3):  # never pick a broken kernel
            raise AssertionError(f"tile {t} gives wrong results")
        s = _time_min(f, xs, reps, min_time)
        results.append({"tile": list(t.key()), "seconds": s, "gflops": flops / s / 1e9})
        fns[t] = f
        if verbose:
            print(f"    {t.key()}: {flops / s / 1e9:7.1f} GFLOP/s")
    default = MatmulTile(*results[0]["tile"])
    ranked = sorted(results[1:], key=lambda r: r["seconds"])[:confirm_top]
    finalists = [default] + [MatmulTile(*r["tile"]) for r in ranked]
    times = {t: [] for t in finalists}
    t_end = time.perf_counter() + confirm_time
    rounds = 0
    while rounds < 5 or (time.perf_counter() < t_end and rounds < 2000):
        for t in finalists:
            t0 = time.perf_counter()
            fns[t](*xs)
            times[t].append(time.perf_counter() - t0)
        rounds += 1
    med = {t: float(np.median(v)) for t, v in times.items()}
    best = min(med, key=med.get)
    if med[best] > med[default] * (1 - min_gain):
        best = default
    if verbose:
        print(f"    confirm ({rounds} rounds): " + ", ".join(
            f"{t.key()}={flops / med[t] / 1e9:.1f}" for t in finalists))
    for r in results:
        t = MatmulTile(*r["tile"])
        if t in med:
            r["confirm_gflops"] = flops / med[t] / 1e9
    return best, flops / med[best] / 1e9, results


def tune_matmul(batch: int, m: int, n: int, k: int, **kw):
    """Tune a plain (epilogue-free) matmul of the given shape."""
    from . import frontend as F
    sa = (batch, m, k) if batch > 1 else (m, k)
    sb = (batch, k, n) if batch > 1 else (k, n)
    g = F.trace(lambda x, y: x @ y, sa, sb)
    return tune_node(g.outputs[0], **kw)


class TuningDB:
    """Persistent map from matmul shape (+ CPU, SIMD width) to best tile."""

    def __init__(self, path: Optional[os.PathLike] = None):
        self.path = Path(path) if path else None
        self.entries: Dict[str, dict] = {}
        if self.path and self.path.exists():
            self.entries = json.loads(self.path.read_text())

    @staticmethod
    def key(shape: Key, vw: int) -> str:
        b, m, n, k = shape[:4]
        epi = shape[4] if len(shape) > 4 else ""
        return f"cpu={cpu_fingerprint()},vw={vw},b={b},m={m},n={n},k={k},epilogue={epi}"

    def get(self, shape: Key, vw: int) -> Optional[MatmulTile]:
        e = self.entries.get(self.key(shape, vw))
        return MatmulTile(*e["tile"]) if e else None

    def record(self, shape: Key, vw: int, tile: MatmulTile, gflops: float,
               default_gflops: float, n_candidates: int):
        self.entries[self.key(shape, vw)] = {
            "shape": {"batch": shape[0], "m": shape[1], "n": shape[2], "k": shape[3],
                      "epilogue": shape[4] if len(shape) > 4 else ""},
            "tile": list(tile.key()), "tile_names": ["mb", "nb", "kb", "mr", "nv"],
            "gflops": round(gflops, 2), "default_gflops": round(default_gflops, 2),
            "candidates": n_candidates,
        }

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True) + "\n")

    def tiles_for(self, g: Graph, options: CodegenOptions) -> Dict[Key, MatmulTile]:
        out = {}
        for s in matmul_shapes(g):
            t = self.get(s, options.vector_width)
            if t is not None:
                out[s] = t
        return out

    def tune(self, shapes: Iterable[Key], vw: Optional[int] = None, retune: bool = False,
             verbose: bool = False, **kw) -> "TuningDB":
        """Tune plain matmuls of the given (batch, M, N, K) shapes."""
        from . import frontend as F
        items = []
        for s in shapes:
            b, m, n, k = s[:4]
            sa = (b, m, k) if b > 1 else (m, k)
            sb = (b, k, n) if b > 1 else (k, n)
            node = F.trace(lambda x, y: x @ y, sa, sb).outputs[0]
            items.append((node_key(node), node))
        return self._tune_nodes(items, vw, retune, verbose, **kw)

    def _tune_nodes(self, items, vw=None, retune=False, verbose=False, **kw) -> "TuningDB":
        from .runtime import native_vector_width
        vw = vw or native_vector_width()
        for s, node in items:
            if not retune and self.get(s, vw) is not None:
                continue
            if verbose:
                print(f"tuning {node.op} batch={s[0]} M={s[1]} N={s[2]} K={s[3]} "
                      f"epilogue=[{s[4]}]")
            tile, gf, res = tune_node(node, vw=vw, verbose=verbose, **kw)
            default_gf = res[0].get("confirm_gflops", res[0]["gflops"])
            self.record(s, vw, tile, gf, default_gf, len(res))
            if verbose:
                print(f"  best {tile.key()} {gf:.1f} GFLOP/s (default {default_gf:.1f})")
            self.save()
        return self

    def tune_graph(self, g: Graph, **kw) -> "TuningDB":
        from .passes import optimize
        return self._tune_nodes(matmul_nodes(optimize(g, "fused")), **kw)


def _parse_shape(s: str) -> Key:
    parts = [int(x) for x in s.lower().split("x")]
    if len(parts) == 3:
        return (1, parts[0], parts[2], parts[1])   # MxKxN
    if len(parts) == 4:
        return (parts[0], parts[1], parts[3], parts[2])  # BxMxKxN
    raise argparse.ArgumentTypeError("shape must be MxKxN or BxMxKxN")


def main(argv=None):
    ap = argparse.ArgumentParser(description="tinyfuse matmul autotuner")
    ap.add_argument("--shape", action="append", type=_parse_shape, required=True,
                    help="MxKxN or BxMxKxN (repeatable)")
    ap.add_argument("--db", default="tuning.json")
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--retune", action="store_true")
    a = ap.parse_args(argv)
    TuningDB(a.db).tune(a.shape, max_candidates=a.candidates, retune=a.retune, verbose=True)


if __name__ == "__main__":
    main()
