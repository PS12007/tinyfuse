import numpy as np
import pytest

import tinyfuse as tf
from tinyfuse.interpreter import run_graph
from tinyfuse.ir import Graph, IRError, Node, format_graph, verify
from tinyfuse.ops import ShapeError, broadcast_shapes, broadcast_strides, matmul_shape


@pytest.mark.parametrize("a,b,out", [
    ((3, 1), (1, 4), (3, 4)), ((5,), (2, 5), (2, 5)), ((), (2, 3), (2, 3)),
    ((2, 1, 3), (4, 1), (2, 4, 3)),
])
def test_broadcast_shapes(a, b, out):
    assert broadcast_shapes(a, b) == out == np.broadcast_shapes(a, b)


def test_broadcast_errors_and_strides():
    with pytest.raises(ShapeError):
        broadcast_shapes((3,), (4,))
    assert broadcast_strides((4,), (3, 4)) == (0, 1)
    assert broadcast_strides((3, 1), (3, 4)) == (1, 0)
    assert broadcast_strides((2, 3, 4), (2, 3, 4)) == (12, 4, 1)


@pytest.mark.parametrize("a,b,out", [
    ((2, 3), (3, 4), (2, 4)), ((5, 2, 3), (3, 4), (5, 2, 4)), ((5, 2, 3), (5, 3, 4), (5, 2, 4)),
    ((2, 7, 2, 3), (2, 7, 3, 4), (2, 7, 2, 4)),
])
def test_matmul_shapes(a, b, out):
    assert matmul_shape(a, b) == out


@pytest.mark.parametrize("a,b", [((2, 3), (4, 5)), ((3,), (3, 4)), ((5, 2, 3), (6, 3, 4))])
def test_matmul_shape_errors(a, b):
    with pytest.raises(ShapeError):
        matmul_shape(a, b)


def test_printer_is_deterministic_and_readable():
    g = tf.trace(lambda x, w, b: tf.gelu(x @ w + b) * 0.5, (2, 3), (3, 4), (4,), name="f")
    assert format_graph(g) == """\
graph f(%x: f32[2,3], %w: f32[3,4], %b: f32[4]) {
  %0: f32[2,4] = matmul(%x, %w)
  %1: f32[2,4] = add(%0, %b)
  %2: f32[2,4] = gelu(%1)
  %3: f32[] = const() {value=0.5}
  %4: f32[2,4] = mul(%2, %3)
  return %4
}"""
    assert str(g) == format_graph(g)


def test_printer_shows_attrs_and_big_consts():
    g = tf.trace(lambda x: tf.softmax(x.transpose(1, 0)) + tf.constant(np.zeros((3, 2))),
                 (2, 3), name="g")
    s = format_graph(g)
    assert "transpose(%x) {perm=[1,0]}" in s
    assert "softmax(%0) {axis=-1}" in s
    assert "value=<6 elems>" in s


def _small_graph():
    return tf.trace(lambda x, y: tf.relu(x + y), (2, 2), (2, 2))


def test_verifier_accepts_valid_graph():
    verify(_small_graph())


def test_verifier_rejects_use_before_def():
    g = _small_graph()
    g.nodes.reverse()
    with pytest.raises(IRError, match="before it is defined"):
        verify(g)


def test_verifier_rejects_wrong_stored_shape():
    g = _small_graph()
    g.nodes[-1].shape = (4,)
    with pytest.raises(IRError, match="stored shape"):
        verify(g)


def test_verifier_rejects_bad_shapes_after_mutation():
    g = _small_graph()
    add = g.nodes[-2]
    add.inputs[1] = g.add_const(np.zeros((3,)))
    g.toposort()
    with pytest.raises(IRError, match="shape inference failed"):
        verify(g)


def test_verifier_rejects_unknown_op_dangling_outputs_and_dups():
    g = _small_graph()
    g.nodes[-1].op = "frobnicate"
    with pytest.raises(IRError, match="unknown op"):
        verify(g)
    g = _small_graph()
    g.outputs = []
    with pytest.raises(IRError, match="no outputs"):
        verify(g)
    g = _small_graph()
    g.outputs = [Node("relu", [g.inputs[0]])]
    with pytest.raises(IRError, match="not defined"):
        verify(g)
    g = _small_graph()
    g.inputs[1].name = "x"
    with pytest.raises(IRError, match="duplicate input"):
        verify(g)
    g = _small_graph()
    g.nodes.append(g.nodes[-1])
    with pytest.raises(IRError, match="twice"):
        verify(g)


def test_verifier_rejects_bad_const_and_program():
    g = _small_graph()
    c = g.add_const(np.ones(2))
    c.attrs["value"] = np.ones(2, dtype=np.float64)
    with pytest.raises(IRError, match="float32"):
        verify(g)
    g = _small_graph()
    n = g.add("fused_elementwise", g.inputs, {"program": (("relu", (("in", 0),)),)})
    n.attrs["program"] = (("relu", (("in", 5),)),)
    with pytest.raises(IRError):
        verify(g)


def test_toposort_and_cycle_detection():
    g = _small_graph()
    order = [n.id for n in g.nodes]
    g.nodes = list(reversed(g.nodes))
    g.toposort()
    verify(g)
    assert sorted(order) == sorted(n.id for n in g.nodes)
    relu = g.nodes[-1]
    add = relu.inputs[0]
    add.inputs[0] = relu
    with pytest.raises(IRError, match="cycle"):
        g.toposort()


def test_copy_is_independent():
    g = _small_graph()
    c = g.copy()
    c.nodes[-1].op = "neg"
    assert g.nodes[-1].op == "relu"
    verify(c)
    x = np.array([[1, -2], [3, -4]], dtype=np.float32)
    np.testing.assert_array_equal(run_graph(g, [x, x])[0], np.maximum(2 * x, 0))
    np.testing.assert_array_equal(run_graph(c, [x, x])[0], -2 * x)


def test_interpreter_checks_inputs():
    g = _small_graph()
    with pytest.raises(TypeError):
        run_graph(g, [np.zeros((2, 2))])
    with pytest.raises(ValueError):
        run_graph(g, [np.zeros((2, 2)), np.zeros((3, 2))])
