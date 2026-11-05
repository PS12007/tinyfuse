import tinyfuse as tf
from tinyfuse.passes import dead_code_elimination
from passes_util import check_pass


def test_removes_unused_values():
    def f(x, unused):
        a = tf.exp(x)            # dead
        b = tf.relu(x)
        _ = a @ a.T              # dead chain
        return b
    changed, g = check_pass(f, [(3, 3), (2,)], [dead_code_elimination], cleanup=False,
                            before="""
graph f(%x: f32[3,3], %unused: f32[2]) {
  %0: f32[3,3] = exp(%x)
  %1: f32[3,3] = relu(%x)
  %2: f32[3,3] = transpose(%0) {perm=[1,0]}
  %3: f32[3,3] = matmul(%0, %2)
  return %1
}""",
                            after="""
graph f(%x: f32[3,3], %unused: f32[2]) {
  %0: f32[3,3] = relu(%x)
  return %0
}""")
    assert changed
    assert len(g.inputs) == 2   # signature is preserved


def test_noop():
    g = tf.trace(lambda x: tf.relu(x), (3,))
    assert dead_code_elimination(g) is False


def test_output_is_input():
    g = tf.trace(lambda x, y: (x, tf.relu(y)), (3,), (3,))
    assert dead_code_elimination(g) is False
    assert g.outputs[0] is g.inputs[0]
