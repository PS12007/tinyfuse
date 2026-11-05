"""Reference models written against the tinyfuse frontend (used by tests,
benchmarks and the README).  Each has a matching NumPy implementation."""

from __future__ import annotations

import math

import numpy as np

from . import frontend as tf
from .ops import np_gelu, np_layernorm, np_softmax


def mlp2(x, w1, b1, w2, b2):
    """2-layer MLP: relu(x @ w1 + b1) @ w2 + b2."""
    return tf.relu(x @ w1 + b1) @ w2 + b2


def np_mlp2(x, w1, b1, w2, b2):
    return np.maximum(x @ w1 + b1, 0) @ w2 + b2


def mlp2_shapes(batch, d_in, d_hidden, d_out):
    return dict(x=(batch, d_in), w1=(d_in, d_hidden), b1=(d_hidden,),
                w2=(d_hidden, d_out), b2=(d_out,))


def make_transformer_block(seq: int, d_model: int, n_heads: int, d_ff: int):
    """Pre-LayerNorm transformer block on a single sequence ``x: [seq, d_model]``:

        h = x + Wo · MHA(LN1(x))
        y = h + W2 · gelu(W1 · LN2(h))
    """
    assert d_model % n_heads == 0
    dh = d_model // n_heads
    scale = 1.0 / math.sqrt(dh)

    def block(x, ln1_g, ln1_b, wq, bq, wk, bk, wv, bv, wo, bo, ln2_g, ln2_b, w1, b1, w2, b2):
        h = tf.layernorm(x, ln1_g, ln1_b)
        q = (h @ wq + bq).reshape(seq, n_heads, dh).transpose(1, 0, 2)   # [H,S,dh]
        k = (h @ wk + bk).reshape(seq, n_heads, dh).transpose(1, 2, 0)   # [H,dh,S]
        v = (h @ wv + bv).reshape(seq, n_heads, dh).transpose(1, 0, 2)   # [H,S,dh]
        att = tf.softmax((q @ k) * scale)                                # [H,S,S]
        o = (att @ v).transpose(1, 0, 2).reshape(seq, d_model)           # [S,D]
        x = x + (o @ wo + bo)
        h = tf.layernorm(x, ln2_g, ln2_b)
        return x + (tf.gelu(h @ w1 + b1) @ w2 + b2)

    D, F = d_model, d_ff
    shapes = dict(x=(seq, D), ln1_g=(D,), ln1_b=(D,), wq=(D, D), bq=(D,), wk=(D, D), bk=(D,),
                  wv=(D, D), bv=(D,), wo=(D, D), bo=(D,), ln2_g=(D,), ln2_b=(D,),
                  w1=(D, F), b1=(F,), w2=(F, D), b2=(D,))

    def np_block(x, ln1_g, ln1_b, wq, bq, wk, bk, wv, bv, wo, bo, ln2_g, ln2_b, w1, b1, w2, b2):
        h = np_layernorm(x, ln1_g, ln1_b, 1e-5)
        q = (h @ wq + bq).reshape(seq, n_heads, dh).transpose(1, 0, 2)
        k = (h @ wk + bk).reshape(seq, n_heads, dh).transpose(1, 2, 0)
        v = (h @ wv + bv).reshape(seq, n_heads, dh).transpose(1, 0, 2)
        att = np_softmax((q @ k) * scale)
        o = (att @ v).transpose(1, 0, 2).reshape(seq, D)
        x = x + (o @ wo + bo)
        h = np_layernorm(x, ln2_g, ln2_b, 1e-5)
        return x + (np_gelu(h @ w1 + b1) @ w2 + b2)

    return block, np_block, shapes


def random_inputs(shapes: dict, seed: int = 0, scale: float = 1.0):
    """Random float32 inputs; weight matrices are scaled by 1/sqrt(fan_in)
    and LayerNorm gains are centred on 1 to keep activations O(1)."""
    rng = np.random.default_rng(seed)
    out = []
    for name, s in shapes.items():
        a = rng.standard_normal(s).astype(np.float32)
        if len(s) == 2 and name != "x":
            a /= np.sqrt(s[0])
        if name.endswith("_g"):
            a = 1.0 + 0.1 * a
        out.append(a * scale if name == "x" else a)
    return out
