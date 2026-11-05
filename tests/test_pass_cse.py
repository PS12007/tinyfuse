import numpy as np

import tinyfuse as tf
from tinyfuse.passes import common_subexpression_elimination as cse
from passes_util import check_pass


def test_merges_duplicates_including_commuted_operands():
    changed, _ = check_pass(
        lambda x, y: tf.relu(x + y) * tf.relu(y + x) + tf.relu(x + y), [(3,), (3,)], [cse],
        before="""
graph f(%x: f32[3], %y: f32[3]) {
  %0: f32[3] = add(%x, %y)
  %1: f32[3] = relu(%0)
  %2: f32[3] = add(%y, %x)
  %3: f32[3] = relu(%2)
  %4: f32[3] = mul(%1, %3)
  %5: f32[3] = add(%x, %y)
  %6: f32[3] = relu(%5)
  %7: f32[3] = add(%4, %6)
  return %7
}""",
        after="""
graph f(%x: f32[3], %y: f32[3]) {
  %0: f32[3] = add(%x, %y)
  %1: f32[3] = relu(%0)
  %2: f32[3] = mul(%1, %1)
  %3: f32[3] = add(%2, %1)
  return %3
}""")
    assert changed


def test_does_not_merge_noncommutative_swaps_or_different_attrs():
    def f(x, y):
        return [x - y, y - x, x.transpose(1, 0, 2), x.transpose(2, 1, 0), x @ y, y @ x]
    g = tf.trace(f, (2, 2, 2), (2, 2, 2))
    assert cse(g) is False


def test_merges_equal_constants_and_layout_ops():
    def f(x):
        a = x * tf.constant(np.ones(3) * 2)
        b = x * tf.constant(np.ones(3) * 2)
        c = x * tf.constant(np.ones(3) * 3)
        return a.reshape(3, 1) + b.reshape(3, 1) + c.reshape(3, 1)
    g = tf.trace(f, (3,))
    assert cse(g)
    counts = g.op_counts()
    assert counts["const"] == 2 and counts["mul"] == 2 and counts["reshape"] == 2


def test_never_merges_inputs_and_dedups_outputs():
    g = tf.trace(lambda x, y: (tf.relu(x), tf.relu(x), tf.relu(y)), (3,), (3,))
    assert cse(g)
    assert g.outputs[0] is g.outputs[1]
    assert g.outputs[2] is not g.outputs[0]
    assert len(g.inputs) == 2
