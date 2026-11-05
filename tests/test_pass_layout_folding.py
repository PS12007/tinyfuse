import numpy as np
import pytest

import tinyfuse as tf
from tinyfuse.codegen import CodegenOptions, MatmulTile
from tinyfuse.passes import layout_folding
from tinyfuse.runtime import native_vector_width
from conftest import assert_close, rand
from passes_util import check_pass

VW = native_vector_width()


def test_transpose_operand_becomes_view():
    changed, _ = check_pass(lambda a, b: a @ b.T, [(3, 4), (5, 4)], [layout_folding],
                            before="""
graph f(%a: f32[3,4], %b: f32[5,4]) {
  %0: f32[4,5] = transpose(%b) {perm=[1,0]}
  %1: f32[3,5] = matmul(%a, %0)
  return %1
}""",
                            after="""
graph f(%a: f32[3,4], %b: f32[5,4]) {
  %0: f32[3,5] = fused_matmul(%a, %b) {b_view=f32[4,5] strides [1,4]}
  return %0
}""")
    assert changed


def test_attention_head_split_folds_through_reshape():
    def f(q, k):
        qh = q.reshape(8, 2, 4).transpose(1, 0, 2)    # [2,8,4]
        kh = k.reshape(8, 2, 4).transpose(1, 2, 0)    # [2,4,8]
        return qh @ kh
    check_pass(f, [(8, 8), (8, 8)], [layout_folding], before="""
graph f(%q: f32[8,8], %k: f32[8,8]) {
  %0: f32[8,2,4] = reshape(%q) {shape=[8,2,4]}
  %1: f32[2,8,4] = transpose(%0) {perm=[1,0,2]}
  %2: f32[8,2,4] = reshape(%k) {shape=[8,2,4]}
  %3: f32[2,4,8] = transpose(%2) {perm=[1,2,0]}
  %4: f32[2,8,8] = matmul(%1, %3)
  return %4
}""", after="""
graph f(%q: f32[8,8], %k: f32[8,8]) {
  %0: f32[2,8,8] = fused_matmul(%q, %k) {a_view=f32[2,8,4] strides [4,8,1], b_view=f32[2,4,8] strides [4,1,8]}
  return %0
}""")


def test_reshape_of_transposed_data_is_not_folded():
    # transpose then reshape mixes dims -> not a strided view; must stay
    g = tf.trace(lambda a, b: a.transpose(1, 0, 2).reshape(6, 4) @ b, (2, 3, 4), (4, 5))
    assert layout_folding(g) is False


def test_views_survive_epilogue_fusion_and_reruns():
    g = tf.trace(lambda a, b, c: tf.relu(a @ b.T + c), (3, 4), (5, 4), (5,))
    o = tf.optimize(g)
    assert o.op_counts() == {"input": 3, "fused_matmul": 1}
    s = str(o)
    assert "b_view=f32[4,5] strides [1,4]" in s and "relu" in s


SHAPE_CASES = [
    (lambda a, b: a @ b.T, [(7, 13), (11, 13)]),
    (lambda a, b: a.T @ b, [(13, 7), (13, 11)]),
    (lambda a, b: a.T @ b.T, [(13, 7), (11, 13)]),
    (lambda a, b: a.transpose(1, 0, 2) @ b, [(9, 3, 5), (3, 5, 6)]),
    (lambda a, b: a.transpose(1, 0, 2) @ b.transpose(1, 2, 0), [(9, 3, 5), (9, 3, 5)]),
    (lambda a, b: a.reshape(9, 3, 5).transpose(1, 0, 2) @ b, [(9, 15), (5, 6)]),
    (lambda a, b: tf.gelu(a.T @ b + 1.0) * 2.0, [(17, 33), (17, 40)]),
]


@pytest.mark.parametrize("case", range(len(SHAPE_CASES)))
@pytest.mark.parametrize("opts", ["default", "tiny", "tiny-parallel"])
def test_folded_matmuls_match_numpy(case, opts, rng):
    fn, shapes = SHAPE_CASES[case]
    tile = MatmulTile(mb=6, nb=VW, kb=4, mr=3, nv=1)
    o = {"default": None,
         "tiny": CodegenOptions(vector_width=VW, default_tile=tile),
         "tiny-parallel": CodegenOptions(vector_width=VW, default_tile=tile, parallel=True,
                                         max_threads=3)}[opts]
    f = tf.compile(fn, *shapes, options=o)
    assert "transpose" not in f.optimized.op_counts()
    xs = [rand(rng, *s) for s in shapes]
    ref = tf.run_graph(tf.trace(fn, *shapes), xs)[0]
    assert_close(f(*xs), ref)


def test_output_transpose_folds_into_store():
    check_pass(lambda a, b: (a @ b).transpose(1, 0, 2), [(2, 3, 4), (2, 4, 5)], [layout_folding],
               before="""
graph f(%a: f32[2,3,4], %b: f32[2,4,5]) {
  %0: f32[2,3,5] = matmul(%a, %b)
  %1: f32[3,2,5] = transpose(%0) {perm=[1,0,2]}
  return %1
}""", after="""
graph f(%a: f32[2,3,4], %b: f32[2,4,5]) {
  %0: f32[3,2,5] = fused_matmul(%a, %b) {c_perm=[1,0,2]}
  return %0
}""")


def test_output_transpose_not_folded_when_last_axis_moves_or_value_shared():
    g = tf.trace(lambda a, b: (a @ b).T, (3, 4), (4, 5))          # moves the last axis
    assert layout_folding(g) is False
    g = tf.trace(lambda a, b: ((a @ b).transpose(1, 0, 2), a @ b), (2, 3, 4), (2, 4, 5))
    from tinyfuse.passes import common_subexpression_elimination
    common_subexpression_elimination(g)
    assert layout_folding(g) is False                             # matmul has 2 users


OUT_CASES = [
    (lambda a, b: (a @ b).transpose(1, 0, 2), [(4, 7, 9), (4, 9, 11)]),
    (lambda a, b: ((a @ b) * 2.0 + 1.0).transpose(1, 0, 2), [(3, 13, 5), (3, 5, 17)]),
    (lambda a, b: (a @ b).transpose(1, 0, 2).reshape(13, 3 * 17) @ tf.constant(np.ones((51, 4))),
     [(3, 13, 5), (3, 5, 17)]),
    (lambda a, b: (a @ b).transpose(2, 1, 0, 3), [(2, 3, 5, 6), (2, 3, 6, 7)]),
]


@pytest.mark.parametrize("case", range(len(OUT_CASES)))
@pytest.mark.parametrize("opts", ["default", "tiny", "tiny-parallel"])
def test_output_folded_matmuls_match_numpy(case, opts, rng):
    fn, shapes = OUT_CASES[case]
    tile = MatmulTile(mb=6, nb=VW, kb=4, mr=3, nv=1)
    o = {"default": None,
         "tiny": CodegenOptions(vector_width=VW, default_tile=tile),
         "tiny-parallel": CodegenOptions(vector_width=VW, default_tile=tile, parallel=True,
                                         max_threads=3)}[opts]
    f = tf.compile(fn, *shapes, options=o)
    assert "transpose" not in f.optimized.op_counts()
    xs = [rand(rng, *s) for s in shapes]
    assert_close(f(*xs), tf.run_graph(tf.trace(fn, *shapes), xs)[0])
