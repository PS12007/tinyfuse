"""Trace a small model, show the IR before/after optimisation, compile it to C,
and check it against NumPy.   python examples/quickstart.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tinyfuse as tf  # noqa: E402


def mlp(x, w1, b1, w2, b2):
    return tf.relu(x @ w1 + b1) @ w2 + b2


shapes = dict(x=(32, 64), w1=(64, 128), b1=(128,), w2=(128, 10), b2=(10,))
f = tf.compile(mlp, **shapes)
print("traced IR:\n", f.graph, "\n", sep="")
print("optimised IR:\n", f.optimized, "\n", sep="")

rng = np.random.default_rng(0)
xs = [rng.standard_normal(s).astype(np.float32) for s in shapes.values()]
x, w1, b1, w2, b2 = xs
ref = np.maximum(x @ w1 + b1, 0) @ w2 + b2
print("max |tinyfuse - numpy| =", float(np.abs(f(*xs) - ref).max()))
print(f"generated C: {len(f.source.splitlines())} lines, cached at {f.module.so_path}")
