import numpy as np

import tinyfuse as tf
from tinyfuse.interpreter import run_graph
from tinyfuse.ir import verify


def check_pass(fn, shapes, passes, before, after, seed=0, cleanup=True):
    """Trace ``fn``, check the IR text before, run ``passes``, check the IR
    text after, and check the interpreter gives identical results."""
    from tinyfuse.passes import dead_code_elimination
    g = tf.trace(fn, *shapes, name="f")
    assert str(g) == before.strip("\n"), "\n" + str(g)
    orig = g.copy()
    changed = False
    for p in passes:
        changed |= p(g)
        verify(g)
    if cleanup:
        dead_code_elimination(g)
        verify(g)
    assert str(g) == after.strip("\n"), "\n" + str(g)
    rng = np.random.default_rng(seed)
    xs = [rng.standard_normal(i.shape).astype(np.float32) for i in g.inputs]
    for a, b in zip(run_graph(orig, xs), run_graph(g, xs)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    return changed, g
