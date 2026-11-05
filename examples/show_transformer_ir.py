"""Print the transformer block IR before and after fusion (used for the README)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tinyfuse as tf  # noqa: E402
from tinyfuse.models import make_transformer_block  # noqa: E402

block, _, shapes = make_transformer_block(seq=64, d_model=256, n_heads=4, d_ff=1024)
g = tf.trace(block, **shapes)
print(g)
print()
print(tf.optimize(g, "fused"))
