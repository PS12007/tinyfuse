"""Short end-to-end demo: trace, fuse, compile, check, time.
python examples/demo.py"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tinyfuse as tf  # noqa: E402


def mlp(x, w1, b1, w2, b2):
    return tf.gelu(x @ w1 + b1) @ w2 + b2


shapes = dict(x=(64, 512), w1=(512, 2048), b1=(2048,), w2=(2048, 512), b2=(512,))
t0 = time.perf_counter()
f = tf.compile(mlp, **shapes)
print(f"compiled in {time.perf_counter() - t0:.2f}s\n")
print(f.optimized, "\n")

rng = np.random.default_rng(0)
x, w1, b1, w2, b2 = [rng.standard_normal(s).astype(np.float32) / 8 for s in shapes.values()]


def ref(x, w1, b1, w2, b2):
    h = x @ w1 + b1
    return (0.5 * h * (1 + np.tanh(0.7978845608 * (h + 0.044715 * h ** 3)))) @ w2 + b2


print("max abs error vs numpy:", float(np.abs(f(x, w1, b1, w2, b2) - ref(x, w1, b1, w2, b2)).max()))


def best(fn, n=30):
    fn(x, w1, b1, w2, b2)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn(x, w1, b1, w2, b2)
        ts.append(time.perf_counter() - t)
    return np.median(ts) * 1e3


print(f"tinyfuse: {best(f):.2f} ms   numpy: {best(ref):.2f} ms")
