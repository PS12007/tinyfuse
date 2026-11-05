from __future__ import annotations

from ..ir import Graph, live_nodes


def dead_code_elimination(g: Graph) -> bool:
    """Remove nodes whose values cannot reach a graph output.  Graph inputs
    are part of the signature and are always kept."""
    live = live_nodes(g)
    changed = len(live) != len(g.nodes)
    g.nodes = list(live)
    return changed
