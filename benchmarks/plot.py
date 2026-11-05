"""Render benchmark CSVs into charts under docs/.

    python benchmarks/plot.py docs/benchmarks_1t.csv [docs/benchmarks_4t.csv ...]

For each CSV and model: grouped bars of throughput (GFLOP/s, median time;
higher is better), one group per problem size, one fixed colour per variant.
Also writes a markdown table next to each CSV for the README.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed variant -> colour mapping (validated categorical palette, light mode).
VARIANTS = OrderedDict([
    ("tinyfuse-unfused", "#eb6834"),
    ("tinyfuse-fused", "#2a78d6"),
    ("tinyfuse-fused+tuned", "#1baf7a"),
    ("numpy", "#eda100"),
    ("pytorch", "#e87ba4"),
])
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def plot_model(rows, model, threads, out_png):
    rows = [r for r in rows if r["model"] == model]
    configs = list(OrderedDict.fromkeys(r["config"] for r in rows))
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    val = {(r["config"], r["variant"]): float(r["gflops_median"]) for r in rows}

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(variants)
    group_w = 0.8
    bw = group_w / n
    for i, v in enumerate(variants):
        xs = [c - group_w / 2 + bw * (i + 0.5) for c in range(len(configs))]
        ys = [val.get((cfg, v), 0.0) for cfg in configs]
        bars = ax.bar(xs, ys, width=bw * 0.86, color=VARIANTS[v], label=v, zorder=3,
                      edgecolor=SURFACE, linewidth=1.0)
        if v == "tinyfuse-fused":  # selective direct labels on the headline series
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.0f}", ha="center",
                        va="bottom", fontsize=8, color=TEXT2)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=9, color=TEXT)
    ax.set_ylabel("GFLOP/s (median, higher is better)", fontsize=9, color=TEXT2)
    title = {"mlp": "2-layer MLP", "transformer": "Transformer block"}[model]
    ax.set_title(f"{title} — float32, {threads} thread{'s' if threads != '1' else ''}",
                 fontsize=11, color=TEXT, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT2, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=n, fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.1), labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_png)


def markdown_table(rows):
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    lines = []
    for model in OrderedDict.fromkeys(r["model"] for r in rows):
        lines.append(f"**{model}** — median ms per call (speedup vs NumPy)\n")
        lines.append("| config | " + " | ".join(variants) + " |")
        lines.append("|---|" + "---:|" * len(variants))
        for cfg in OrderedDict.fromkeys(r["config"] for r in rows if r["model"] == model):
            cells = []
            for v in variants:
                r = next((r for r in rows if r["model"] == model and r["config"] == cfg
                          and r["variant"] == v), None)
                cells.append(f"{float(r['median_ms']):.3f} (×{float(r['speedup_vs_numpy']):.2f})"
                             if r else "–")
            lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(paths):
    for p in paths:
        rows = load(p)
        threads = rows[0]["threads"]
        base = os.path.dirname(os.path.abspath(p))
        for model in OrderedDict.fromkeys(r["model"] for r in rows):
            plot_model(rows, model, threads, os.path.join(base, f"bench_{model}_{threads}t.png"))
        md = os.path.splitext(p)[0] + ".md"
        with open(md, "w") as fh:
            fh.write(markdown_table(rows))
        print("wrote", md)


if __name__ == "__main__":
    main(sys.argv[1:] or ["docs/benchmarks_1t.csv"])
