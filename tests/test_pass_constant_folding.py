import numpy as np

import tinyfuse as tf
from tinyfuse.passes import constant_folding
from passes_util import check_pass


def test_folds_constant_subexpressions():
    changed, _ = check_pass(
        lambda x: x * (tf.constant(np.full((3,), 2.0)) + 1.0) + 0.0, [(3,)], [constant_folding],
        before="""
graph f(%x: f32[3]) {
  %0: f32[3] = const() {value=[2.,2.,2.]}
  %1: f32[] = const() {value=1.}
  %2: f32[3] = add(%0, %1)
  %3: f32[3] = mul(%x, %2)
  %4: f32[] = const() {value=0.}
  %5: f32[3] = add(%3, %4)
  return %5
}""",
        after="""
graph f(%x: f32[3]) {
  %0: f32[3] = const() {value=[3.,3.,3.]}
  %1: f32[3] = mul(%x, %0)
  return %1
}""")
    assert changed


def test_folds_through_matmul_softmax_layernorm_transpose():
    w = np.arange(6, dtype=np.float32).reshape(2, 3)

    def f(x):
        c = tf.softmax(tf.constant(w).T @ tf.constant(w))            # all-constant
        g = tf.layernorm(tf.constant(w), tf.constant(np.ones(3)), tf.constant(np.zeros(3)))
        return x @ c + g.reshape(2, 3)

    g = tf.trace(f, (2, 3))
    constant_folding(g)
    from tinyfuse.passes import dead_code_elimination
    dead_code_elimination(g)
    assert g.op_counts() == {"input": 1, "const": 2, "matmul": 1, "add": 1}
    from tinyfuse.ops import np_layernorm, np_softmax
    x = np.ones((2, 3), np.float32)
    ref = x @ np_softmax(w.T @ w) + np_layernorm(w, 1, 0, 1e-5)
    np.testing.assert_allclose(tf.run_graph(g, [x])[0], ref, rtol=1e-5)


def test_algebraic_identities_and_layout_chains():
    check_pass(
        lambda x: tf.relu(x).reshape(6).reshape(2, 3).transpose(1, 0).transpose(1, 0) * 1.0,
        [(2, 3)], [constant_folding],
        before="""
graph f(%x: f32[2,3]) {
  %0: f32[2,3] = relu(%x)
  %1: f32[6] = reshape(%0) {shape=[6]}
  %2: f32[2,3] = reshape(%1) {shape=[2,3]}
  %3: f32[3,2] = transpose(%2) {perm=[1,0]}
  %4: f32[2,3] = transpose(%3) {perm=[1,0]}
  %5: f32[] = const() {value=1.}
  %6: f32[2,3] = mul(%4, %5)
  return %6
}""",
        after="""
graph f(%x: f32[2,3]) {
  %0: f32[2,3] = relu(%x)
  return %0
}""")


def test_transpose_composition_keeps_nontrivial_perm():
    check_pass(lambda x: x.transpose(1, 2, 0).transpose(1, 0, 2), [(2, 3, 4)], [constant_folding],
               before="""
graph f(%x: f32[2,3,4]) {
  %0: f32[3,4,2] = transpose(%x) {perm=[1,2,0]}
  %1: f32[4,3,2] = transpose(%0) {perm=[1,0,2]}
  return %1
}""",
               after="""
graph f(%x: f32[2,3,4]) {
  %0: f32[4,3,2] = transpose(%x) {perm=[2,1,0]}
  return %0
}""")


def test_identity_that_would_change_shape_is_not_applied():
    # x[3] + zeros-scalar is fine, but x[3] * ones[2,3] broadcasts: must be kept
    g = tf.trace(lambda x: x * tf.constant(np.ones((2, 3))), (3,))
    constant_folding(g)
    assert g.outputs[0].shape == (2, 3)
    assert g.outputs[0].op == "mul"


def test_noop_returns_false_and_double_negation():
    g = tf.trace(lambda x: tf.relu(x), (3,))
    assert constant_folding(g) is False
    g = tf.trace(lambda x: -(-x), (3,))
    assert constant_folding(g) is True
    assert g.outputs[0] is g.inputs[0]


def test_output_that_is_constant():
    g = tf.trace(lambda x: (x + 1.0, tf.constant(2.0) * 3.0), (2,))
    constant_folding(g)
    assert g.outputs[1].op == "const"
    assert float(g.outputs[1].attrs["value"]) == 6.0
