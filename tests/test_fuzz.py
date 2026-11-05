"""Randomised differential testing: random graphs of elementwise, layout,
reduction and matmul ops, compiled with every pipeline, must agree with the
reference interpreter."""
import numpy as np
import pytest

import tinyfuse as tf
from tinyfuse.codegen import CodegenOptions, MatmulTile
from tinyfuse.runtime import native_vector_width
from conftest import assert_close

VW = native_vector_width()


def random_program(seed):
    rng = np.random.default_rng(seed)
    R, C = int(rng.integers(1, 9)), int(rng.integers(1, 20))

    def fn(x, y, b):
        vals = [x, y, x * 0.5]
        for _ in range(int(rng.integers(3, 12))):
            a = vals[int(rng.integers(len(vals)))]
            c = vals[int(rng.integers(len(vals)))]
            k = int(rng.integers(12))
            if k == 0:
                v = a + c
            elif k == 1:
                v = a * c
            elif k == 2:
                v = a - b
            elif k == 3:
                v = tf.relu(a)
            elif k == 4:
                v = tf.tanh(a) * 2.0
            elif k == 5:
                v = tf.gelu(a)
            elif k == 6:
                v = tf.softmax(a)
            elif k == 7:
                v = (a @ c.T) @ c * 0.1
            elif k == 8:
                v = tf.layernorm(a, b, b)
            elif k == 9:
                v = a.T.T + 0.0
            elif k == 10:
                v = tf.sigmoid(a.reshape(C, R).T)
            else:
                v = tf.maximum(a, c) / 3.0
            vals.append(v)
        n_out = int(rng.integers(1, 3))
        return vals[-n_out:]

    return fn, [(R, C), (R, C), (C,)]


@pytest.mark.parametrize("seed", range(40))
def test_random_graphs(seed):
    fn, shapes = random_program(seed)
    g = tf.trace(fn, *shapes)
    rng = np.random.default_rng(seed + 1000)
    xs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    ref = tf.run_graph(g, xs)
    tiny = CodegenOptions(vector_width=VW, default_tile=MatmulTile(mb=3, nb=VW, kb=2, mr=3, nv=1))
    tiny_par = CodegenOptions(vector_width=VW, parallel=True, max_threads=3,
                              default_tile=MatmulTile(mb=3, nb=VW, kb=2, mr=3, nv=1))
    for pipeline, opts in [("none", None), ("fused", None), ("fused", tiny), ("fused", tiny_par)]:
        # re-trace: the random program consumes its rng while tracing
        fn2, _ = random_program(seed)
        f = tf.compile(fn2, *shapes, pipeline=pipeline, options=opts)
        outs = f(*xs)
        outs = list(outs) if isinstance(outs, tuple) else [outs]
        for o, r in zip(outs, ref):
            assert_close(o, r)
