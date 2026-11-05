import numpy as np

import tinyfuse as tf
from tinyfuse.codegen import plan_memory
from tinyfuse.models import make_transformer_block
from tinyfuse.passes import optimize


def _check_no_overlap(g, plan):
    ws = [s for s in plan.storages if s.kind == "ws"]
    for i, a in enumerate(ws):
        assert a.offset >= 0 and a.offset + a.size <= plan.ws_size
        for b in ws[i + 1:]:
            live_overlap = not (a.last < b.first or b.last < a.first)
            mem_overlap = not (a.offset + a.size <= b.offset or b.offset + b.size <= a.offset)
            assert not (live_overlap and mem_overlap), (a, b)


def test_chain_reuses_memory():
    def f(x):
        for _ in range(10):
            x = tf.exp(x) * 0.5
        return x
    g = optimize(tf.trace(f, (1000,)), "none")
    plan = plan_memory(g)
    _check_no_overlap(g, plan)
    assert plan.ws_size < plan.naive_ws_size / 4
    assert plan_memory(g, reuse=False).ws_size == plan.naive_ws_size


def test_transformer_plan_is_valid():
    blk, _, sh = make_transformer_block(32, 64, 4, 128)
    for p in ("none", "fused"):
        g = optimize(tf.trace(blk, **sh), p)
        plan = plan_memory(g)
        _check_no_overlap(g, plan)
        assert plan.ws_size <= plan.naive_ws_size
        assert "KiB" in plan.describe()


def test_reshape_aliases_and_outputs():
    g = tf.trace(lambda x: tf.relu(x).reshape(6), (2, 3))
    plan = plan_memory(g)
    relu, rs = g.nodes[1], g.nodes[2]
    assert plan.node_storage[relu.id] is plan.node_storage[rs.id]
    assert plan.node_storage[rs.id].kind == "output"   # relu writes straight into output
    assert plan.output_copies == []
    g = tf.trace(lambda x: (x, x.reshape(6)), (2, 3))
    assert len(plan_memory(g).output_copies) == 2
