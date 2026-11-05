"""End-to-end: 2-layer MLP and a transformer block vs NumPy (tol 1e-4)."""
import numpy as np
import pytest

import tinyfuse as tf
from tinyfuse.codegen import CodegenOptions, MatmulTile
from tinyfuse.models import make_transformer_block, mlp2, mlp2_shapes, np_mlp2, random_inputs
from tinyfuse.runtime import native_vector_width
from conftest import assert_close

VW = native_vector_width()
TINY = CodegenOptions(vector_width=VW, default_tile=MatmulTile(mb=6, nb=VW, kb=7, mr=3, nv=1))


@pytest.mark.parametrize("pipeline", ["none", "unfused", "fused"])
@pytest.mark.parametrize("dims", [(1, 4, 8, 2), (32, 64, 128, 10), (65, 129, 257, 31),
                                  (256, 512, 1024, 256)])
def test_mlp2(pipeline, dims):
    sh = mlp2_shapes(*dims)
    xs = random_inputs(sh, seed=sum(dims))
    f = tf.compile(mlp2, pipeline=pipeline, **sh)
    assert_close(f(*xs), np_mlp2(*[x.astype(np.float64) for x in xs]))


def test_mlp2_fused_structure():
    f = tf.compile(mlp2, **mlp2_shapes(8, 16, 32, 4))
    assert f.optimized.op_counts() == {"input": 5, "fused_matmul": 2}
    assert f.source.count("static void") == 2


@pytest.mark.parametrize("pipeline", ["none", "unfused", "fused"])
@pytest.mark.parametrize("cfg", [(4, 8, 2, 16), (16, 64, 4, 256), (33, 96, 3, 200),
                                 (128, 256, 8, 1024)])
def test_transformer_block(pipeline, cfg):
    blk, npb, sh = make_transformer_block(*cfg)
    xs = random_inputs(sh, seed=cfg[0])
    f = tf.compile(blk, pipeline=pipeline, **sh)
    assert_close(f(*xs), npb(*[x.astype(np.float64) for x in xs]))


def test_transformer_block_tiny_tiles():
    blk, npb, sh = make_transformer_block(19, 24, 3, 40)
    xs = random_inputs(sh, seed=3)
    f = tf.compile(blk, options=TINY, **sh)
    assert_close(f(*xs), npb(*[x.astype(np.float64) for x in xs]))


def test_transformer_fused_structure():
    blk, _, sh = make_transformer_block(16, 32, 2, 64)
    f = tf.compile(blk, **sh)
    c = f.optimized.op_counts()
    # q,k,v,o,ff1,ff2 + 2 attention matmuls; the scores matmul fuses the scale,
    # att@v has no consumer to fuse
    assert c["fused_matmul"] == 7 and c["matmul"] == 1
    assert c.get("fused_elementwise", 0) == 0
    assert not any(op in c for op in ("add", "mul", "gelu"))
    unf = tf.compile(blk, pipeline="unfused", **sh)
    assert unf.module.gen.plan.ws_size > f.module.gen.plan.ws_size


def test_against_pytorch_if_available():
    torch = pytest.importorskip("torch")
    blk, _, sh = make_transformer_block(16, 32, 4, 64)
    xs = random_inputs(sh, seed=7)
    f = tf.compile(blk, **sh)
    T = [torch.from_numpy(x.astype(np.float64)) for x in xs]
    x, g1, be1, wq, bq, wk, bk, wv, bv, wo, bo, g2, be2, w1, b1, w2, b2 = T
    F = torch.nn.functional
    h = F.layer_norm(x, (32,), g1, be1, 1e-5)
    q = (h @ wq + bq).view(16, 4, 8).transpose(0, 1)
    k = (h @ wk + bk).view(16, 4, 8).transpose(0, 1)
    v = (h @ wv + bv).view(16, 4, 8).transpose(0, 1)
    att = torch.softmax(q @ k.transpose(1, 2) / np.sqrt(8), -1)
    o = (att @ v).transpose(0, 1).reshape(16, 32)
    x = x + o @ wo + bo
    y = x + F.gelu(F.layer_norm(x, (32,), g2, be2, 1e-5) @ w1 + b1, approximate="tanh") @ w2 + b2
    assert_close(f(*xs), y.numpy())


@pytest.mark.parametrize("threads", [2, 4])
def test_parallel_builds_match(threads):
    blk, npb, sh = make_transformer_block(33, 48, 3, 100)
    xs = random_inputs(sh, seed=5)
    ref = npb(*[x.astype(np.float64) for x in xs])
    for tile in (None, MatmulTile(mb=6, nb=VW, kb=7, mr=3, nv=1)):
        opts = CodegenOptions(vector_width=VW, parallel=True, max_threads=threads,
                              default_tile=tile)
        f = tf.compile(blk, options=opts, **sh)
        for _ in range(3):   # repeated calls: no stale state between runs
            assert_close(f(*xs), ref)
    sh2 = mlp2_shapes(37, 70, 130, 9)
    xs2 = random_inputs(sh2, seed=6)
    f = tf.compile(mlp2, options=CodegenOptions(vector_width=VW, parallel=True,
                                                max_threads=threads), **sh2)
    assert_close(f(*xs2), np_mlp2(*[x.astype(np.float64) for x in xs2]))
