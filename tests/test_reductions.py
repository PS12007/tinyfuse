import numpy as np
import pytest

import tinyfuse as tf
from tinyfuse.codegen import CodegenOptions
from tinyfuse.ops import np_layernorm
from tinyfuse.passes import elementwise_fusion, reduction_fusion
from tinyfuse.runtime import native_vector_width
from conftest import assert_close, rand
from passes_util import check_pass

VW = native_vector_width()
REF = {"sum": np.sum, "mean": np.mean, "max": np.max}


@pytest.mark.parametrize("kind", sorted(REF))
@pytest.mark.parametrize("shape", [(1,), (7,), (3, 17), (2, 3, 33), (4, 1000)])
@pytest.mark.parametrize("keepdims", [True, False])
def test_reduce_ops(kind, shape, keepdims, rng):
    x = rand(rng, *shape)
    fn = lambda x: getattr(tf, kind)(x, keepdims=keepdims)   # noqa: E731
    ref = REF[kind](x.astype(np.float64), axis=-1, keepdims=keepdims)
    for p in ("none", "fused"):
        f = tf.compile(fn, shape, pipeline=p)
        assert f(x).shape == ref.shape
        assert_close(f(x), ref)


def test_reduce_shape_rules():
    g = tf.trace(lambda x: (tf.sum(x), tf.mean(x, keepdims=False)), (2, 3, 4))
    assert [o.shape for o in g.outputs] == [(2, 3, 1), (2, 3)]
    with pytest.raises(ValueError):
        tf.trace(lambda x: tf.sum(x, axis=0), (2, 3))


def test_prologue_fusion_ir():
    changed, _ = check_pass(lambda x: tf.mean(x * x), [(4, 8)], [reduction_fusion], before="""
graph f(%x: f32[4,8]) {
  %0: f32[4,8] = mul(%x, %x)
  %1: f32[4,1] = reduce_mean(%0) {axis=-1, keepdims=True}
  return %1
}""", after="""
graph f(%x: f32[4,8]) {
  %0: f32[4,1] = fused_reduce(%x) {axis=-1, in_shape=[4,8], keepdims=True, kind=mean, program=[t0=mul(in0,in0)]}
  return %0
}""")
    assert changed


def test_prologue_fusion_absorbs_fused_elementwise_with_broadcast():
    g = tf.trace(lambda x, b: tf.max(tf.exp(x + b) * 2.0), (5, 6), (6,))
    elementwise_fusion(g)
    assert reduction_fusion(g)
    from tinyfuse.passes import dead_code_elimination
    dead_code_elimination(g)
    assert g.op_counts() == {"input": 2, "fused_reduce": 1}


def test_no_prologue_fusion_when_value_is_reused():
    # x*x is also an output -> must be materialised, reduction stays plain
    g = tf.trace(lambda x: (x * x, tf.sum(x * x)), (3, 4))
    from tinyfuse.passes import common_subexpression_elimination
    common_subexpression_elimination(g)
    assert reduction_fusion(g) is False


def rmsnorm(x, g):
    return x * tf.rsqrt(tf.mean(x * x) + 1e-6) * g


def my_layernorm(x, g, b):
    mu = tf.mean(x)
    d = x - mu
    return d * tf.rsqrt(tf.mean(d * d) + 1e-5) * g + b


@pytest.mark.parametrize("parallel", [False, True])
def test_handwritten_norms(parallel, rng):
    opts = CodegenOptions(vector_width=VW, parallel=parallel)
    x, gg, bb = rand(rng, 64, 300) * 3 + 1, rand(rng, 300), rand(rng, 300)
    f = tf.compile(rmsnorm, (64, 300), (300,), options=opts)
    x64 = x.astype(np.float64)
    assert_close(f(x, gg), x64 / np.sqrt((x64 ** 2).mean(-1, keepdims=True) + 1e-6) * gg)
    assert f.optimized.op_counts().get("fused_reduce") == 1
    h = tf.compile(my_layernorm, (64, 300), (300,), (300,), options=opts)
    assert_close(h(x, gg, bb), np_layernorm(x64, gg, bb, 1e-5))
    # and it agrees with the built-in layernorm kernel
    builtin = tf.compile(lambda x, g, b: tf.layernorm(x, g, b), (64, 300), (300,), (300,))
    assert_close(h(x, gg, bb), builtin(x, gg, bb))


def test_sqrt_rsqrt(rng):
    x = np.abs(rand(rng, 4, 9)) + 0.1
    f = tf.compile(lambda x: (tf.sqrt(x), tf.rsqrt(x)), (4, 9))
    a, b = f(x)
    assert_close(a, np.sqrt(x.astype(np.float64)))
    assert_close(b, 1 / np.sqrt(x.astype(np.float64)))
