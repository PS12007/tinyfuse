"""Every op, compiled to C, against NumPy (tolerance 1e-4)."""
import numpy as np
import pytest

import tinyfuse as tf
from tinyfuse.codegen import CodegenOptions, MatmulTile
from tinyfuse.ops import np_gelu, np_layernorm, np_sigmoid, np_softmax
from tinyfuse.runtime import native_vector_width
from conftest import assert_close, rand

PIPELINES = ["none", "unfused", "fused"]
VW = native_vector_width()
TINY = CodegenOptions(vector_width=VW, default_tile=MatmulTile(mb=6, nb=VW, kb=5, mr=3, nv=1))


def run(fn, shapes, xs, pipeline="fused", options=None):
    f = tf.compile(fn, *shapes, pipeline=pipeline, options=options)
    return f(*xs)


UNARY = {
    "relu": (tf.relu, lambda x: np.maximum(x, 0)),
    "gelu": (tf.gelu, np_gelu),
    "neg": (tf.neg, np.negative),
    "exp": (tf.exp, np.exp),
    "tanh": (tf.tanh, np.tanh),
    "sigmoid": (tf.sigmoid, np_sigmoid),
}


@pytest.mark.parametrize("name", sorted(UNARY))
@pytest.mark.parametrize("shape", [(1,), (7,), (3, 17), (2, 3, 33)])
def test_unary(name, shape, rng):
    f, ref = UNARY[name]
    x = rand(rng, *shape) * 3
    assert_close(run(f, [shape], [x]), ref(x.astype(np.float64)))


BINARY = {
    "add": (lambda a, b: a + b, np.add),
    "sub": (lambda a, b: a - b, np.subtract),
    "mul": (lambda a, b: a * b, np.multiply),
    "div": (lambda a, b: a / b, np.divide),
    "maximum": (tf.maximum, np.maximum),
}


@pytest.mark.parametrize("name", sorted(BINARY))
@pytest.mark.parametrize("sa,sb", [((5,), (5,)), ((4, 9), (9,)), ((4, 1), (1, 9)),
                                   ((2, 3, 4), (3, 1)), ((), (3, 2)), ((6, 1, 5), (1, 7, 1))])
def test_binary_broadcasting(name, sa, sb, rng):
    f, ref = BINARY[name]
    a = rand(rng, *sa)
    b = rand(rng, *sb)
    if name == "div":
        b = np.sign(b) * (np.abs(b) + 0.5)
    for p in ("none", "fused"):
        assert_close(run(f, [sa, sb], [a, b], p), ref(a.astype(np.float64), b))


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (2, 3, 4), (7, 13, 5), (16, 16, 16),
                                   (33, 65, 47), (64, 128, 96), (100, 300, 70), (5, 600, 3)])
@pytest.mark.parametrize("tiny", [False, True])
def test_matmul_2d(m, k, n, tiny, rng):
    a, b = rand(rng, m, k), rand(rng, k, n)
    out = run(lambda a, b: a @ b, [(m, k), (k, n)], [a, b], options=TINY if tiny else None)
    assert_close(out, a.astype(np.float64) @ b)


@pytest.mark.parametrize("sa,sb", [((3, 5, 7), (7, 4)), ((3, 5, 7), (3, 7, 4)),
                                   ((2, 3, 9, 6), (2, 3, 6, 11)), ((4, 1, 8), (4, 8, 1))])
@pytest.mark.parametrize("tiny", [False, True])
def test_matmul_batched(sa, sb, tiny, rng):
    a, b = rand(rng, *sa), rand(rng, *sb)
    out = run(lambda a, b: a @ b, [sa, sb], [a, b], options=TINY if tiny else None)
    assert_close(out, np.matmul(a.astype(np.float64), b))


@pytest.mark.parametrize("act", ["relu", "gelu", None])
@pytest.mark.parametrize("m,k,n", [(1, 4, 3), (9, 17, 31), (50, 70, 130)])
@pytest.mark.parametrize("tiny", [False, True])
def test_fused_matmul_bias_act(act, m, k, n, tiny, rng):
    def f(x, w, b):
        y = x @ w + b
        return getattr(tf, act)(y) if act else y
    x, w, b = rand(rng, m, k), rand(rng, k, n), rand(rng, n)
    ref = x.astype(np.float64) @ w + b
    ref = {"relu": lambda v: np.maximum(v, 0), "gelu": np_gelu, None: lambda v: v}[act](ref)
    for p in PIPELINES:
        assert_close(run(f, [(m, k), (k, n), (n,)], [x, w, b], p, TINY if tiny else None), ref)


@pytest.mark.parametrize("tiny", [False, True])
def test_fused_matmul_epilogue_broadcasts_in_batch(tiny, rng):
    # epilogue inputs with every broadcast pattern over [2,3,M,N]
    sa, sb = (2, 3, 7, 5), (2, 3, 5, 9)
    shapes = [sa, sb, (9,), (7, 1), (3, 1, 1), (2, 1, 7, 9), ()]
    xs = [rand(rng, *s) for s in shapes]

    def f(a, b, c0, c1, c2, c3, c4):
        return tf.sigmoid((a @ b) * c4 + c0) - c1 * c2 + c3

    a, b, c0, c1, c2, c3, c4 = [x.astype(np.float64) for x in xs]
    ref = np_sigmoid(np.matmul(a, b) * c4 + c0) - c1 * c2 + c3
    f_ = tf.compile(f, *shapes, options=TINY if tiny else None)
    assert f_.optimized.op_counts().get("fused_matmul") == 1
    assert_close(f_(*xs), ref)


@pytest.mark.parametrize("shape", [(1,), (5,), (3, 17), (2, 4, 64), (6, 1000)])
def test_softmax(shape, rng):
    x = rand(rng, *shape) * 4
    for p in ("none", "fused"):
        assert_close(run(tf.softmax, [shape], [x], p), np_softmax(x.astype(np.float64)))


def test_softmax_is_stable_for_large_inputs(rng):
    x = rand(rng, 4, 33) * 1000
    assert_close(run(tf.softmax, [(4, 33)], [x]), np_softmax(x.astype(np.float64)))


@pytest.mark.parametrize("shape", [(1,), (8,), (3, 17), (2, 5, 64), (4, 1024)])
def test_layernorm(shape, rng):
    d = shape[-1]
    x, g, b = rand(rng, *shape) * 2 + 1, rand(rng, d), rand(rng, d)
    out = run(lambda x, g, b: tf.layernorm(x, g, b, eps=1e-5), [shape, (d,), (d,)], [x, g, b])
    assert_close(out, np_layernorm(x.astype(np.float64), g, b, 1e-5))


@pytest.mark.parametrize("shape,perm", [
    ((3, 4), (1, 0)), ((37, 70), (1, 0)), ((2, 3, 4), (0, 2, 1)), ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4), (1, 0, 2)), ((5, 6, 7, 8), (3, 1, 0, 2)), ((64, 4, 32), (1, 2, 0)),
    ((1, 5, 1, 3), (3, 2, 1, 0)), ((2, 3), (0, 1)),
])
def test_transpose(shape, perm, rng):
    x = rand(rng, *shape)
    for p in ("none", "fused"):
        assert_close(run(lambda x: x.transpose(perm), [shape], [x], p), np.transpose(x, perm))


@pytest.mark.parametrize("shape,new", [((2, 6), (3, 4)), ((24,), (2, 3, 4)), ((2, 3, 4), (-1,)),
                                       ((4, 6), (2, -1, 3))])
def test_reshape(shape, new, rng):
    x = rand(rng, *shape)
    # reshape is a view: check it on its own (output aliasing) and inside a chain
    assert_close(run(lambda x: x.reshape(new), [shape], [x]), x.reshape(new))
    out = run(lambda x: tf.relu(tf.exp(x).reshape(new)) * 2.0, [shape], [x])
    assert_close(out, np.maximum(np.exp(x.astype(np.float64)), 0).reshape(new) * 2)


def test_constants_are_embedded(rng):
    w = rand(rng, 6, 5)
    b = rand(rng, 5)
    f = tf.compile(lambda x: tf.relu(x @ tf.constant(w) + b), (3, 6))
    x = rand(rng, 3, 6)
    assert_close(f(x), np.maximum(x.astype(np.float64) @ w + b, 0))


def test_special_outputs(rng):
    """Outputs that are inputs, constants, aliases, or duplicated."""
    def f(x, y):
        r = tf.relu(y)
        return x, tf.constant(np.full((2,), 3.0)), r, r, r.reshape(6), x.reshape(3, 2)
    x, y = rand(rng, 2, 3), rand(rng, 2, 3)
    for p in PIPELINES:
        outs = run(f, [(2, 3), (2, 3)], [x, y], p)
        assert len(outs) == 6
        assert_close(outs[0], x)
        assert_close(outs[1], [3, 3])
        assert_close(outs[2], np.maximum(y, 0))
        assert_close(outs[3], np.maximum(y, 0))
        assert_close(outs[4], np.maximum(y, 0).reshape(6))
        assert_close(outs[5], x.reshape(3, 2))
        assert not np.shares_memory(outs[0], x)


def test_out_parameter_and_repeated_calls(rng):
    f = tf.compile(lambda a, b: tf.relu(a @ b + 1.0), (4, 5), (5, 6))
    out = np.empty((4, 6), np.float32)
    for _ in range(3):
        a, b = rand(rng, 4, 5), rand(rng, 5, 6)
        r = f(a, b, out=[out])
        assert r is out
        assert_close(out, np.maximum(a.astype(np.float64) @ b + 1, 0))


def test_input_validation_and_dtype_conversion(rng):
    f = tf.compile(lambda a: a * 2.0, (2, 3))
    with pytest.raises(ValueError):
        f(np.zeros((3, 2)))
    with pytest.raises(TypeError):
        f()
    with pytest.raises(ValueError):
        f(np.zeros((2, 3)), out=[np.zeros((2, 3))])  # float64 out
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert_close(f(x), x * 2)
    assert_close(f(np.asfortranarray(x)), x * 2)          # non-contiguous input
    assert_close(f(x.T.copy().T), x * 2)


def test_parallel_build_matches(rng):
    from tinyfuse.models import make_transformer_block, random_inputs
    blk, npb, sh = make_transformer_block(16, 32, 2, 64)
    xs = random_inputs(sh)
    opts = CodegenOptions(vector_width=VW, parallel=True)
    f = tf.compile(blk, options=opts, **sh)
    assert "-fopenmp" not in f.source  # flag lives in the build, not the source
    assert_close(f(*xs), npb(*[x.astype(np.float64) for x in xs]))


def test_profiler_reports_every_kernel(rng):
    opts = CodegenOptions(vector_width=VW, profile=True)
    f = tf.compile(lambda x, w: tf.softmax(tf.relu(x @ w)).T, (8, 8), (8, 8), options=opts)
    x = rand(rng, 8, 8)
    f(x, x)
    prof = f.module.profile(reset=True)
    assert [p[0].split("_", 1)[1] for p in prof] == ["fused_matmul", "softmax", "transpose"]
    assert all(p[2] > 0 for p in prof)
    assert all(p[2] == 0 for p in f.module.profile())
    g = tf.compile(lambda x: x * 2.0, (2,))
    with pytest.raises(RuntimeError):
        g.module.profile()
