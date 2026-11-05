import numpy as np

import tinyfuse as tf
from tinyfuse.passes import (constant_folding, dead_code_elimination, elementwise_fusion,
                             matmul_epilogue_fusion, optimize)
from tinyfuse.interpreter import run_graph
from passes_util import check_pass


def test_elementwise_chain_with_broadcast_and_scalar_literal():
    changed, _ = check_pass(
        lambda x, y, z: tf.gelu(x * y + z) - 1.0, [(4, 8), (4, 8), (8,)], [elementwise_fusion],
        before="""
graph f(%x: f32[4,8], %y: f32[4,8], %z: f32[8]) {
  %0: f32[4,8] = mul(%x, %y)
  %1: f32[4,8] = add(%0, %z)
  %2: f32[4,8] = gelu(%1)
  %3: f32[] = const() {value=1.}
  %4: f32[4,8] = sub(%2, %3)
  return %4
}""",
        after="""
graph f(%x: f32[4,8], %y: f32[4,8], %z: f32[8]) {
  %0: f32[4,8] = fused_elementwise(%x, %y, %z) {program=[t0=mul(in0,in1); t1=add(t0,in2); t2=gelu(t1); t3=sub(t2,1)]}
  return %0
}""")
    assert changed


def test_multi_use_value_fused_when_all_users_in_group():
    check_pass(lambda x: tf.relu(x) * tf.exp(x) + tf.relu(x) * 2.0, [(5,)],
               [elementwise_fusion],
               before="""
graph f(%x: f32[5]) {
  %0: f32[5] = relu(%x)
  %1: f32[5] = exp(%x)
  %2: f32[5] = mul(%0, %1)
  %3: f32[5] = relu(%x)
  %4: f32[] = const() {value=2.}
  %5: f32[5] = mul(%3, %4)
  %6: f32[5] = add(%2, %5)
  return %6
}""",
               after="""
graph f(%x: f32[5]) {
  %0: f32[5] = fused_elementwise(%x) {program=[t0=relu(in0); t1=exp(in0); t2=mul(t0,t1); t3=relu(in0); t4=mul(t3,2); t5=add(t2,t4)]}
  return %0
}""")


def test_fusion_stops_at_non_elementwise_ops_and_outputs():
    # relu(x) is an output *and* feeds the add; softmax blocks fusion
    check_pass(lambda x: (tf.relu(x), tf.softmax(tf.relu(x) + 1.0) * 3.0 + 1.0), [(2, 4)],
               [elementwise_fusion],
               before="""
graph f(%x: f32[2,4]) {
  %0: f32[2,4] = relu(%x)
  %1: f32[2,4] = relu(%x)
  %2: f32[] = const() {value=1.}
  %3: f32[2,4] = add(%1, %2)
  %4: f32[2,4] = softmax(%3) {axis=-1}
  %5: f32[] = const() {value=3.}
  %6: f32[2,4] = mul(%4, %5)
  %7: f32[] = const() {value=1.}
  %8: f32[2,4] = add(%6, %7)
  return %0, %8
}""",
               after="""
graph f(%x: f32[2,4]) {
  %0: f32[2,4] = relu(%x)
  %1: f32[2,4] = fused_elementwise(%x) {program=[t0=relu(in0); t1=add(t0,1)]}
  %2: f32[2,4] = softmax(%1) {axis=-1}
  %3: f32[2,4] = fused_elementwise(%2) {program=[t0=mul(in0,3); t1=add(t0,1)]}
  return %0, %3
}""")


def test_value_used_by_two_groups_is_not_duplicated():
    def f(x, w):
        a = tf.exp(x)
        return (a + 1.0) @ w, tf.relu(a)     # a feeds two different groups
    g = tf.trace(f, (3, 3), (3, 3))
    elementwise_fusion(g)
    assert g.op_counts().get("exp", 0) == 1


def test_broadcast_producer_is_not_fused_into_bigger_consumer():
    g = tf.trace(lambda b, x: tf.exp(b) + x, (8,), (4, 8))
    assert elementwise_fusion(g) is False


def test_refusion_merges_fused_nodes():
    g = tf.trace(lambda x: tf.relu(tf.exp(x) + 1.0) * 2.0, (4,))
    elementwise_fusion(g)
    dead_code_elimination(g)
    # build a second level of fusion by hand: fused -> neg
    n = g.add("neg", [g.outputs[0]])
    g.outputs = [n]
    assert elementwise_fusion(g)
    dead_code_elimination(g)
    assert g.op_counts() == {"input": 1, "fused_elementwise": 1}
    x = np.linspace(-2, 2, 4).astype(np.float32)
    np.testing.assert_allclose(run_graph(g, [x])[0], -(np.maximum(np.exp(x) + 1, 0) * 2),
                               rtol=1e-6)


def test_matmul_bias_relu_fusion():
    changed, _ = check_pass(
        lambda x, w, b: tf.relu(x @ w + b), [(4, 8), (8, 16), (16,)], [matmul_epilogue_fusion],
        before="""
graph f(%x: f32[4,8], %w: f32[8,16], %b: f32[16]) {
  %0: f32[4,16] = matmul(%x, %w)
  %1: f32[4,16] = add(%0, %b)
  %2: f32[4,16] = relu(%1)
  return %2
}""",
        after="""
graph f(%x: f32[4,8], %w: f32[8,16], %b: f32[16]) {
  %0: f32[4,16] = fused_matmul(%x, %w, %b) {epilogue=[t0=add(acc,in0); t1=relu(t0)]}
  return %0
}""")
    assert changed


def test_matmul_bias_gelu_residual_fusion():
    check_pass(lambda x, w, b, r: r + tf.gelu(x @ w + b), [(4, 8), (8, 8), (8,), (4, 8)],
               [matmul_epilogue_fusion],
               before="""
graph f(%x: f32[4,8], %w: f32[8,8], %b: f32[8], %r: f32[4,8]) {
  %0: f32[4,8] = matmul(%x, %w)
  %1: f32[4,8] = add(%0, %b)
  %2: f32[4,8] = gelu(%1)
  %3: f32[4,8] = add(%r, %2)
  return %3
}""",
               after="""
graph f(%x: f32[4,8], %w: f32[8,8], %b: f32[8], %r: f32[4,8]) {
  %0: f32[4,8] = fused_matmul(%x, %w, %b, %r) {epilogue=[t0=add(acc,in0); t1=gelu(t0); t2=add(in1,t1)]}
  return %0
}""")


def test_matmul_with_multiple_users_is_not_fused():
    g = tf.trace(lambda x, w: (tf.relu(x @ w), x @ w), (2, 2), (2, 2))
    from tinyfuse.passes import common_subexpression_elimination
    common_subexpression_elimination(g)
    assert matmul_epilogue_fusion(g) is False


def test_matmul_output_stops_chain_and_shape_change_stops_chain():
    g = tf.trace(lambda x, w: x @ w, (2, 2), (2, 2))
    assert matmul_epilogue_fusion(g) is False
    # mm[2,3] + y[4,2,3] broadcasts to a bigger shape -> not absorbed
    g = tf.trace(lambda x, w, y: (x @ w) + y, (2, 2), (2, 3), (4, 2, 3))
    assert matmul_epilogue_fusion(g) is False


def test_matmul_absorbs_already_fused_elementwise():
    g = tf.trace(lambda x, w, b: tf.relu(x @ w + b) * 0.5, (3, 4), (4, 5), (5,))
    elementwise_fusion(g)
    assert g.op_counts()["fused_elementwise"] == 1
    assert matmul_epilogue_fusion(g)
    dead_code_elimination(g)
    assert g.op_counts() == {"input": 3, "fused_matmul": 1}
    assert "epilogue=[t0=add(acc,in0); t1=relu(t0); t2=mul(t1,0.5)]" in str(g)


def test_fused_matmul_absorbs_more_consumers_on_second_run():
    g = tf.trace(lambda x, w, b: tf.relu(x @ w + b), (3, 4), (4, 5), (5,))
    matmul_epilogue_fusion(g)
    n = g.add("neg", [g.outputs[0]])
    g.outputs = [n]
    assert matmul_epilogue_fusion(g)
    dead_code_elimination(g)
    assert "epilogue=[t0=add(acc,in0); t1=relu(t0); t2=neg(t1)]" in str(g)


def test_full_pipeline_on_mlp():
    from tinyfuse.models import mlp2, mlp2_shapes
    g = tf.trace(mlp2, **mlp2_shapes(8, 16, 32, 4))
    o = optimize(g, "fused")
    assert o.op_counts() == {"input": 5, "fused_matmul": 2}
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal(i.shape).astype(np.float32) for i in g.inputs]
    np.testing.assert_allclose(run_graph(g, xs)[0], run_graph(o, xs)[0], rtol=1e-5, atol=1e-6)
    assert optimize(g, "none").op_counts() == g.op_counts()
