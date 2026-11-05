"""Benchmark tinyfuse (unfused / fused / fused+autotuned) against NumPy and
PyTorch eager on CPU, for a 2-layer MLP and a transformer block.

Methodology
-----------
* Thread counts are pinned *before* NumPy/PyTorch are imported
  (OPENBLAS/OMP/MKL_NUM_THREADS, torch.set_num_threads); tinyfuse is built
  with OpenMP only when --threads > 1.
* Every variant's output is checked against NumPy (float64) at 1e-4 before
  any timing.
* Single-threaded, variants are timed *interleaved*: each round calls every
  variant once, for many rounds, so slow drifts in machine load hit all
  variants equally.  With several threads, interleaving makes the thread pools
  of different runtimes fight for cores, so variants are timed in separate
  warmed-up blocks instead (3 passes).  We report the median and the minimum
  per-call time; charts use the median.

Usage::

    python benchmarks/run_benchmarks.py --threads 1 --out docs/benchmarks_1t.csv
"""

from __future__ import annotations

import argparse
import os
import sys


def _pin_threads(n: int):
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tuning-db", default=os.path.join(os.path.dirname(__file__), "tuning.json"))
    ap.add_argument("--rounds", type=int, default=0, help="0 = auto (time budget)")
    ap.add_argument("--budget", type=float, default=6.0, help="seconds per config")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smallest sizes only")
    ap.add_argument("--models", default="mlp,transformer",
                    help="comma-separated subset of mlp,transformer,elementwise")
    ap.add_argument("--blocked", action="store_true", default=None,
                    help="time each variant in its own block (default when --threads > 1)")
    a = ap.parse_args(argv)
    if a.blocked is None:
        a.blocked = a.threads > 1
    _pin_threads(a.threads)

    import csv
    import platform
    import time

    import numpy as np

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import tinyfuse as tf
    from tinyfuse.autotune import TuningDB
    from tinyfuse.codegen import CodegenOptions
    from tinyfuse.models import (ew_chain, ew_chain_shapes, make_transformer_block, mlp2,
                                 mlp2_shapes, np_ew_chain, np_mlp2, random_inputs)
    from tinyfuse.runtime import native_vector_width

    try:
        import torch
        torch.set_num_threads(a.threads)
        F = torch.nn.functional
    except ImportError:  # pragma: no cover
        torch = None

    vw = native_vector_width()
    opts = lambda: CodegenOptions(vector_width=vw, parallel=a.threads > 1)  # noqa: E731
    db = TuningDB(a.tuning_db)

    # ---- workloads ----------------------------------------------------------
    MLP = [(16, 256, 1024, 256), (64, 512, 2048, 512), (256, 1024, 4096, 1024)]
    TB = [(64, 256, 4, 1024), (128, 512, 8, 2048), (256, 768, 12, 3072)]
    EW = [(256, 1024), (1024, 1024), (4096, 1024)]
    if a.quick:
        MLP, TB, EW = MLP[:1], TB[:1], EW[:1]

    def torch_ew(x, a_, b_):
        h = x * a_ + b_
        return F.gelu(h, approximate="tanh") * torch.sigmoid(x) + x * 0.5 - torch.tanh(h) * 0.1

    def torch_mlp(x, w1, b1, w2, b2):
        return torch.relu(x @ w1 + b1) @ w2 + b2

    def make_torch_block(seq, d, h, ff):
        dh = d // h

        def block(x, g1, be1, wq, bq, wk, bk, wv, bv, wo, bo, g2, be2, w1, b1, w2, b2):
            hh = F.layer_norm(x, (d,), g1, be1, 1e-5)
            q = (hh @ wq + bq).view(seq, h, dh).transpose(0, 1)
            k = (hh @ wk + bk).view(seq, h, dh).permute(1, 2, 0)
            v = (hh @ wv + bv).view(seq, h, dh).transpose(0, 1)
            att = torch.softmax((q @ k) * (1.0 / dh ** 0.5), -1)
            o = (att @ v).transpose(0, 1).reshape(seq, d)
            x = x + (o @ wo + bo)
            hh = F.layer_norm(x, (d,), g2, be2, 1e-5)
            return x + (F.gelu(hh @ w1 + b1, approximate="tanh") @ w2 + b2)
        return block

    def mlp_flops(b, i, hd, o):
        return 2 * b * (i * hd + hd * o)

    def tb_flops(s, d, h, ff):
        return 2 * s * (4 * d * d + 2 * d * ff) + 2 * 2 * s * s * d

    rows = []
    wanted = a.models.split(",")
    workloads = []
    if "mlp" in wanted:
        workloads += [("mlp", c, mlp_flops(*c), "GFLOP/s") for c in MLP]
    if "transformer" in wanted:
        workloads += [("transformer", c, tb_flops(*c), "GFLOP/s") for c in TB]
    if "elementwise" in wanted:
        # throughput in elements: this chain is limited by its 3 transcendentals
        # (~3.5 ns/element single-threaded), not by memory bandwidth
        workloads += [("elementwise", c, c[0] * c[1], "Melem/s") for c in EW]
    for model, cfg, flops, unit in workloads:
        if model == "mlp":
            fn, np_fn, shapes = mlp2, np_mlp2, mlp2_shapes(*cfg)
            th_fn = torch_mlp
            label = "batch={} {}→{}→{}".format(*cfg)
        elif model == "elementwise":
            fn, np_fn, shapes = ew_chain, np_ew_chain, ew_chain_shapes(*cfg)
            th_fn = torch_ew
            label = "{} x {}".format(*cfg)
        else:
            fn, np_fn, shapes = make_transformer_block(*cfg)
            th_fn = make_torch_block(*cfg) if torch else None
            label = "seq={} d={} h={} ff={}".format(*cfg)
        xs = random_inputs(shapes, seed=1)
        g = tf.trace(fn, **shapes)

        t0 = time.perf_counter()
        variants = {
            "tinyfuse-unfused": tf.compile_graph(g, "unfused", opts()),
            "tinyfuse-fused": tf.compile_graph(g, "fused", opts()),
        }
        has_matmul = any(n.op == "matmul" for n in g.nodes)
        if not a.no_tune and has_matmul:   # the tuner only tunes matmul tiles
            db.tune_graph(g, verbose=True)
            variants["tinyfuse-fused+tuned"] = tf.compile_graph(g, "fused", opts(), tuning=db)
        compile_s = time.perf_counter() - t0
        variants["numpy"] = np_fn
        if torch is not None:
            txs = [torch.from_numpy(x) for x in xs]

            def run_torch(*_, _f=th_fn, _x=txs):
                with torch.no_grad():
                    return _f(*_x)
            variants["pytorch"] = run_torch

        ref = np_fn(*[x.astype(np.float64) for x in xs])
        for name, f in variants.items():
            out = f(*xs)
            out = out.numpy() if hasattr(out, "numpy") else out
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4, err_msg=name)

        times = {k: [] for k in variants}
        if not a.blocked:
            # interleaved: one call per variant per round
            for f in variants.values():
                for _ in range(3):
                    f(*xs)
            t_start = time.perf_counter()
            r = 0
            while True:
                for name, f in variants.items():
                    t = time.perf_counter()
                    f(*xs)
                    times[name].append(time.perf_counter() - t)
                r += 1
                if a.rounds and r >= a.rounds:
                    break
                if not a.rounds and (time.perf_counter() - t_start > a.budget and r >= 10):
                    break
        else:
            # blocked: with several threads, interleaving makes the thread pools
            # of different runtimes (OpenMP, OpenBLAS, torch) fight each other;
            # instead each variant runs alone in a warmed-up block, and the
            # blocks are repeated in 3 passes to spread machine drift.
            per = a.budget / (3 * len(variants))
            for _ in range(3):
                for name, f in variants.items():
                    for _ in range(3):
                        f(*xs)
                    t_end = time.perf_counter() + per
                    n = 0
                    while n < 5 or time.perf_counter() < t_end:
                        t = time.perf_counter()
                        f(*xs)
                        times[name].append(time.perf_counter() - t)
                        n += 1
                    time.sleep(0.05)
            r = min(len(v) for v in times.values())
        np_med = float(np.median(times["numpy"]))
        scale = 1e6 if unit == "Melem/s" else 1e9
        print(f"\n{model} {label}  ({'blocked' if a.blocked else 'interleaved'}, {r} rounds, compile {compile_s:.1f}s, {a.threads} thread(s))")
        for name, ts in times.items():
            med, mn = float(np.median(ts)), float(np.min(ts))
            rows.append(dict(model=model, config=label, cfg="x".join(map(str, cfg)),
                             variant=name, threads=a.threads, rounds=r,
                             median_ms=round(med * 1e3, 4), min_ms=round(mn * 1e3, 4),
                             gflops_median=round(flops / med / scale, 2), unit=unit,
                             speedup_vs_numpy=round(np_med / med, 3)))
            print(f"  {name:22s} median {med * 1e3:9.3f} ms   min {mn * 1e3:9.3f} ms   "
                  f"{flops / med / scale:7.1f} {unit}   x{np_med / med:5.2f} vs numpy")

    out = a.out or os.path.join(os.path.dirname(__file__), "..", "docs",
                                f"benchmarks_{a.threads}t.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = os.path.splitext(out)[0] + "_env.txt"
    with open(meta, "w") as fh:
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                    if l.startswith("model name")), platform.processor())
        fh.write(f"cpu: {cpu}\nlogical cpus: {os.cpu_count()}\nthreads: {a.threads}\n"
                 f"simd width (floats): {vw}\npython: {platform.python_version()}\n"
                 f"numpy: {np.__version__}\ntorch: {torch.__version__ if torch else 'n/a'}\n")
        try:
            cfg = np.show_config(mode="dicts")
            fh.write(f"numpy blas: {cfg['Build Dependencies']['blas']['name']}\n")
        except Exception:
            pass
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
