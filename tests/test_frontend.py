import numpy as np
import pytest

import tinyfuse as tf
from tinyfuse.ir import verify


def test_trace_records_ops_and_shapes():
    def f(x, w):
        return tf.relu(x @ w)
    g = tf.trace(f, (4, 8), (8, 16))
    verify(g)
    assert [n.op for n in g.nodes] == ["input", "input", "matmul", "relu"]
    assert g.outputs[0].shape == (4, 16)
    assert [i.name for i in g.inputs] == ["x", "w"]


def test_named_shapes_and_mixed():
    g = tf.trace(lambda a, b: a + b, (2, 3), b=(3,))
    assert g.inputs[1].shape == (3,)
    with pytest.raises(TypeError):
        tf.trace(lambda a, b: a + b, (2, 3))
    with pytest.raises(TypeError):
        tf.trace(lambda a: a, (2,), (3,))
    with pytest.raises(TypeError):
        tf.trace(lambda a: a, a=(2,), z=(3,))


def test_operator_overloads():
    def f(x, y):
        return [x + y, x - y, x * y, x / y, -x, 2.0 * x, 1.0 - x, 3.0 / x, x + 1]
    g = tf.trace(f, (3,), (3,))
    ops = [o.op for o in g.outputs]
    assert ops == ["add", "sub", "mul", "div", "neg", "mul", "sub", "div", "add"]
    # reflected ops keep operand order
    assert g.outputs[6].inputs[0].op == "const"


def test_numpy_array_on_left_defers_to_tensor():
    w = np.ones((3, 3), dtype=np.float32)
    g = tf.trace(lambda x: [w @ x, w + x, w * x], (3, 3))
    assert [o.op for o in g.outputs] == ["matmul", "add", "mul"]
    assert g.outputs[0].inputs[0].op == "const"


def test_layout_methods():
    def f(x):
        return [x.reshape(6, 4), x.reshape((-1,)), x.T, x.transpose(1, 0, 2), x.transpose()]
    g = tf.trace(f, (2, 3, 4))
    assert [o.shape for o in g.outputs] == [(6, 4), (24,), (4, 3, 2), (3, 2, 4), (4, 3, 2)]


def test_python_control_flow_is_unrolled():
    def f(x):
        for _ in range(3):
            x = tf.relu(x) + 1.0
        return x
    g = tf.trace(f, (5,))
    assert g.op_counts()["relu"] == 3


def test_multiple_outputs_and_constant():
    def f(x):
        c = tf.constant(np.arange(4, dtype=np.float32))
        return x + c, x * c
    g = tf.trace(f, (4,))
    assert len(g.outputs) == 2


def test_ops_outside_trace_raise():
    with pytest.raises(RuntimeError):
        tf.relu(np.zeros(3))


def test_shape_errors_at_trace_time():
    with pytest.raises(ValueError):
        tf.trace(lambda a, b: a @ b, (2, 3), (4, 5))
    with pytest.raises(ValueError):
        tf.trace(lambda a, b: a + b, (2, 3), (4,))
    with pytest.raises(ValueError):
        tf.trace(lambda a: a.reshape(7), (2, 3))
    with pytest.raises(ValueError):
        tf.trace(lambda a: tf.softmax(a, axis=0), (2, 3))
    with pytest.raises(ValueError):
        tf.trace(lambda a, g, b: tf.layernorm(a, g, b), (2, 3), (2,), (3,))
    with pytest.raises(TypeError):
        tf.trace(lambda a: a + "x", (2,))
