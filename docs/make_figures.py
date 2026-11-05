"""Generate the README figures from tinyfuse's own data.

    python docs/make_figures.py            # all figures
    python docs/make_figures.py graph      # just one (graph|tiling|profile|memory|terminal)

Everything drawn here comes from the real compiler: the traced and fused IR,
the memory plan, the per-kernel profiler, and a captured terminal session.
"""

from __future__ import annotations

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "img")

import tinyfuse as tf  # noqa: E402
from tinyfuse.models import make_transformer_block, random_inputs  # noqa: E402

# palette (validated categorical slots, light surface)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#b9b8b2"

CATEGORY = {
    "matmul": ("matmul", BLUE),
    "fused_matmul": ("matmul + fused epilogue", BLUE),
    "layernorm": ("layernorm / softmax", AQUA),
    "softmax": ("layernorm / softmax", AQUA),
    "transpose": ("transpose / reshape", GRAY),
    "reshape": ("transpose / reshape", GRAY),
    "fused_elementwise": ("elementwise", ORANGE),
}


def category(op):
    return CATEGORY.get(op, ("elementwise", ORANGE))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8, length=0)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def transformer_graphs(seq=64, d=256, h=4, ff=1024):
    blk, npb, sh = make_transformer_block(seq, d, h, ff)
    g = tf.trace(blk, **sh)
    return g, tf.optimize(g, "fused"), blk, npb, sh


# ---------------------------------------------------------------------------
# 1. the transformer graph before / after fusion
# ---------------------------------------------------------------------------

def _epilogue_words(n):
    """Describe a fused_matmul epilogue in words, from its real operands."""
    words = []
    extra = n.inputs[2:]
    for op, operands in n.attrs["epilogue"]:
        if op == "add":
            ins = [o for o in operands if o[0] == "in"]
            if ins:
                x = extra[ins[0][1]]
                words.append("bias" if len(x.shape) == 1 else "residual")
            else:
                words.append("add")
        elif op == "mul" and any(o[0] == "c" for o in operands):
            words.append("scale")
        else:
            words.append(op)
    return words


def _label(n):
    if n.op == "fused_matmul":
        return "matmul + " + " + ".join(_epilogue_words(n))
    return n.op


def _layout(g):
    ops = [n for n in g.nodes if n.op not in ("input", "const")]
    depth = {}
    for n in ops:
        ds = [depth[i.id] + 1 for i in n.inputs if i.id in depth]
        depth[n.id] = max(ds, default=0)
    layers = {}
    for n in ops:
        layers.setdefault(depth[n.id], []).append(n)
    pos = {}
    for dpt in sorted(layers):
        row = layers[dpt]
        # barycentre of parents keeps the q/k/v branches apart and stable
        def bary(n):
            xs = [pos[i.id][0] for i in n.inputs if i.id in pos]
            return sum(xs) / len(xs) if xs else 0.0
        row.sort(key=bary)
        k = len(row)
        for j, n in enumerate(row):
            pos[n.id] = ((j - (k - 1) / 2) * 1.75, -dpt)
    return ops, pos


def _draw_graph(ax, g, title):
    ops, pos = _layout(g)
    for n in ops:
        x1, y1 = pos[n.id]
        for i in n.inputs:
            if i.id in pos:
                x0, y0 = pos[i.id]
                ax.annotate("", xy=(x1, y1 + 0.28), xytext=(x0, y0 - 0.28),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8,
                                            shrinkA=0, shrinkB=0, mutation_scale=7))
    for n in ops:
        x, y = pos[n.id]
        label = _label(n)
        color = category(n.op)[1]
        w = max(0.75, 0.072 * len(label) + 0.25)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.26), w, 0.52,
                                    boxstyle="round,pad=0.0,rounding_size=0.08",
                                    fc=color, ec=SURFACE, lw=1.5, zorder=3))
        if n.op == "fused_matmul":   # orange stripe: elementwise work folded in
            ax.add_patch(FancyBboxPatch((x + w / 2 - 0.16, y - 0.26), 0.16, 0.52,
                                        boxstyle="round,pad=0.0,rounding_size=0.06",
                                        fc=ORANGE, ec="none", zorder=3.5))
        dark = color in (BLUE,)
        ax.text(x - (0.08 if n.op == "fused_matmul" else 0), y, label, ha="center",
                va="center", fontsize=7.3, zorder=4,
                color="white" if dark else TEXT)
    ys = [p[1] for p in pos.values()]
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(min(ys) - 0.8, 0.8)
    ax.set_title(title, fontsize=10.5, color=TEXT, loc="left")
    ax.axis("off")


def fig_graph():
    g, opt, *_ = transformer_graphs()
    def counts(gr):
        ops = [n for n in gr.nodes if n.op not in ("input", "const")]
        return len(ops), len([n for n in ops if n.op != "reshape"])   # reshape is free
    (n0, k0), (n1, k1) = counts(g), counts(opt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 9.5), dpi=150,
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(SURFACE)
    _draw_graph(axes[0], g, f"as traced: {n0} ops, {k0} kernels")
    _draw_graph(axes[1], opt, f"after fusion: {n1} ops, {k1} kernels")
    # same vertical scale so the height difference is real
    lo = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    for ax in axes:
        ax.set_ylim(lo, 0.8)
    handles = [Rectangle((0, 0), 1, 1, fc=BLUE, ec="none", label="matmul"),
               Rectangle((0, 0), 1, 1, fc=ORANGE, ec="none", label="elementwise"),
               Rectangle((0, 0), 1, 1, fc=AQUA, ec="none", label="layernorm / softmax"),
               Rectangle((0, 0), 1, 1, fc=GRAY, ec="none", label="transpose / reshape")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, labelcolor=TEXT)
    fig.text(0.01, 0.985, "Transformer block (seq=64, d=256, 4 heads). Graph inputs and "
             "constants not drawn. Orange stripe = elementwise ops folded into the matmul.",
             fontsize=8.5, color=TEXT2, va="top")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, "fusion_graph.png")


# ---------------------------------------------------------------------------
# 2. animation: how the tiled matmul walks the problem
# ---------------------------------------------------------------------------

def fig_tiling():
    from PIL import Image
    M, N, K = 12, 16, 8          # toy problem, in elements
    MB, NB, KB, MR, NR = 6, 8, 4, 2, 4
    frames, durations = [], []
    acc = np.zeros((M, N))       # fraction of K accumulated
    done = np.zeros((M, N), bool)

    def draw(caption, a_hi=None, b_hi=None, c_hi=None, a_pack=None, b_pack=None):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.9), dpi=100,
                                 gridspec_kw={"width_ratios": [K, N, N]})
        fig.patch.set_facecolor(SURFACE)
        panels = [("A  (M x K)", M, K), ("B  (K x N)", K, N), ("C = A @ B  (M x N)", M, N)]
        for ax, (title, h, w) in zip(axes, panels):
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(title, fontsize=9, color=TEXT2, loc="left")
            ax.add_patch(Rectangle((0, 0), w, h, fc="white", ec=GRID, lw=1))
        # C: accumulated fraction, finished tiles in aqua
        for i in range(M):
            for j in range(N):
                if done[i, j]:
                    axes[2].add_patch(Rectangle((j, i), 1, 1, fc=AQUA, ec="white", lw=0.5))
                elif acc[i, j] > 0:
                    axes[2].add_patch(Rectangle((j, i), 1, 1, fc=BLUE, alpha=0.25 + 0.5 * acc[i, j],
                                                ec="white", lw=0.5))
        for ax, hi, col in ((axes[0], a_pack, YELLOW), (axes[1], b_pack, YELLOW),
                            (axes[0], a_hi, ORANGE), (axes[1], b_hi, ORANGE),
                            (axes[2], c_hi, ORANGE)):
            if hi is not None:
                r0, r1, c0, c1 = hi
                ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fc=col, alpha=0.35 if col == YELLOW else 0.6,
                                       ec=col, lw=2))
        for ax, (title, h, w) in zip(axes, panels):   # cell grid on top
            for x in range(w + 1):
                ax.plot([x, x], [0, h], color=GRID, lw=0.4)
            for y in range(h + 1):
                ax.plot([0, w], [y, y], color=GRID, lw=0.4)
        fig.text(0.015, 0.10, caption, fontsize=10, color=TEXT)
        fig.text(0.015, 0.03, "yellow = packed panel    orange = micro-kernel working set    "
                 "blue = partially accumulated    aqua = finished, epilogue applied",
                 fontsize=7.5, color=TEXT2)
        fig.tight_layout(rect=(0, 0.15, 1, 1))
        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                               fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)
        return img

    def add(img, ms):
        frames.append(img)
        durations.append(ms)

    add(draw("C[M,N] = A[M,K] @ B[K,N], blocked as MB x NB x KB with a 2x4 register micro-tile"), 1500)
    for j0 in range(0, N, NB):
        for k0 in range(0, K, KB):
            bp = (k0, k0 + KB, j0, j0 + NB)
            add(draw(f"pack B[{k0}:{k0 + KB}, {j0}:{j0 + NB}] into contiguous NR-wide panels",
                     b_pack=bp), 700)
            for i0 in range(0, M, MB):
                ap = (i0, i0 + MB, k0, k0 + KB)
                add(draw(f"pack A[{i0}:{i0 + MB}, {k0}:{k0 + KB}] into MR-tall panels",
                         a_pack=ap, b_pack=bp), 500)
                for ii in range(i0, i0 + MB, MR):
                    for jj in range(j0, j0 + NB, NR):
                        acc[ii:ii + MR, jj:jj + NR] = (k0 + KB) / K
                        add(draw(f"micro-kernel: C[{ii}:{ii + MR}, {jj}:{jj + NR}] += "
                                 f"A panel x B panel  (k {k0}..{k0 + KB - 1}, accumulators in registers)",
                                 a_hi=(ii, ii + MR, k0, k0 + KB), b_hi=(k0, k0 + KB, jj, jj + NR),
                                 c_hi=(ii, ii + MR, jj, jj + NR), a_pack=ap, b_pack=bp), 220)
                if k0 + KB >= K:
                    done[i0:i0 + MB, j0:j0 + NB] = True
                    add(draw("last K block done: epilogue (bias, gelu, residual...) runs on this "
                             "block while it is still in cache", c_hi=(i0, i0 + MB, j0, j0 + NB)), 900)
    add(draw("done: one pass over C, no intermediate tensors"), 2000)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "matmul_tiling.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)
    print("wrote", os.path.relpath(path, ROOT), f"({len(frames)} frames)")


# ---------------------------------------------------------------------------
# 3. where the time goes: per-kernel profile, unfused vs fused
# ---------------------------------------------------------------------------

def fig_profile(cfg=(128, 512, 8, 2048), calls=40):
    from tinyfuse.codegen import CodegenOptions
    blk, npb, sh = make_transformer_block(*cfg)
    xs = random_inputs(sh, seed=1)
    g = tf.trace(blk, **sh)
    mods = {p: tf.compile_graph(g, p, CodegenOptions(profile=True)) for p in ("unfused", "fused")}
    for m in mods.values():
        m(*xs)
        m.module.profile(reset=True)
    for _ in range(calls):            # interleaved
        for m in mods.values():
            m(*xs)
    cats = ["matmul", "elementwise", "layernorm / softmax", "transpose"]
    colors = [BLUE, ORANGE, AQUA, GRAY]
    rows = {}
    for p, m in mods.items():
        tot = dict.fromkeys(cats, 0.0)
        for name, _, sec in m.module.profile():
            op = name.split("_", 1)[1]
            c = {"fused_matmul": "matmul", "matmul": "matmul", "softmax": "layernorm / softmax",
                 "layernorm": "layernorm / softmax", "transpose": "transpose"}.get(op, "elementwise")
            tot[c] += sec / calls * 1e3
        rows[p] = tot
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    labels = {"unfused": "unfused (25 kernels)", "fused": "fused (15 kernels)"}
    for y, p in enumerate(["fused", "unfused"]):
        left = 0.0
        for c, col in zip(cats, colors):
            v = rows[p][c]
            if v <= 0:
                continue
            ax.barh(y, v, left=left, height=0.5, color=col, edgecolor=SURFACE, linewidth=2)
            if v > 0.45:
                ax.text(left + v / 2, y, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if col == BLUE else TEXT)
            left += v
        ax.text(left + 0.1, y, f"{left:.2f} ms", va="center", fontsize=8.5, color=TEXT)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([labels["fused"], labels["unfused"]], fontsize=9, color=TEXT)
    ax.set_xlabel("ms per call, single thread (from the built-in kernel profiler)", fontsize=8.5,
                  color=TEXT2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(sum(r.values()) for r in rows.values()) * 1.12)
    ax.set_title(f"Where the time goes: transformer block seq={cfg[0]} d={cfg[1]} "
                 f"heads={cfg[2]} ff={cfg[3]}", fontsize=10.5, color=TEXT, loc="left")
    handles = [Rectangle((0, 0), 1, 1, fc=c, label=n) for n, c in zip(cats, colors)]
    ax.legend(handles=handles, ncol=4, frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0, -0.32), labelcolor=TEXT)
    fig.tight_layout()
    save(fig, "kernel_profile.png")


# ---------------------------------------------------------------------------
# 4. memory planner: buffer lifetimes vs workspace offset
# ---------------------------------------------------------------------------

def fig_memory(cfg=(128, 512, 8, 2048)):
    from tinyfuse.codegen import plan_memory
    blk, _, sh = make_transformer_block(*cfg)
    g = tf.trace(blk, **sh)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    peak = 0
    for ax, p in zip(axes, ("unfused", "fused")):
        opt = tf.optimize(g, p)
        plan = plan_memory(opt)
        naive = plan.naive_ws_size * 4 / 1024
        producer = {}
        for n in opt.nodes:
            st = plan.node_storage[n.id]
            producer.setdefault(st.sid, n.op)
        style_axes(ax)
        for st in plan.storages:
            if st.kind != "ws":
                continue
            col = category(producer[st.sid])[1]
            ax.add_patch(Rectangle((st.first, st.offset * 4 / 1024), st.last - st.first + 1,
                                   st.size * 4 / 1024, fc=col, ec=SURFACE, lw=1.2))
        peak = max(peak, plan.ws_size * 4 / 1024)
        first_op = min(st.first for st in plan.storages if st.kind == "ws")
        ax.set_xlim(first_op - 0.5, len(opt.nodes) + 0.5)
        ax.set_title(f"{p}: {plan.ws_size * 4 / 1024:.0f} KiB workspace "
                     f"(without reuse: {naive:.0f} KiB)", fontsize=9.5, color=TEXT, loc="left")
        ax.set_xlabel("node index (program order)", fontsize=8.5, color=TEXT2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, peak * 1.05)
    axes[0].set_ylabel("workspace offset (KiB)", fontsize=8.5, color=TEXT2)
    handles = [Rectangle((0, 0), 1, 1, fc=c, label=n) for n, c in
               (("matmul output", BLUE), ("elementwise output", ORANGE),
                ("layernorm / softmax output", AQUA), ("transpose output", GRAY))]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=8, loc="lower center",
               labelcolor=TEXT)
    fig.text(0.01, 0.985, "Each box is one intermediate tensor: from the node that writes it to "
             "the last node that reads it. Boxes stacked at the same height share memory.",
             fontsize=8.5, color=TEXT2, va="top")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, "memory_plan.png")


# ---------------------------------------------------------------------------
# 5. terminal recording of examples/demo.py (real output, replayed as a GIF)
# ---------------------------------------------------------------------------

def fig_terminal():
    import tempfile
    from PIL import Image, ImageDraw, ImageFont
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1",
               TINYFUSE_CACHE=tempfile.mkdtemp(prefix="tf_demo_"))   # cold cache: real compile
    out = subprocess.run([sys.executable, "examples/demo.py"], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout.rstrip("\n").split("\n")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 13)
    W, LH, PAD = 1180, 19, 16
    H = PAD * 2 + 28 + LH * (len(out) + 3)
    bg, fg, dim, green, accent = (24, 24, 27), (228, 228, 231), (140, 140, 150), (110, 200, 140), (120, 170, 255)
    prompt = "~/tinyfuse $ "
    cmd = "python examples/demo.py"

    def render(typed, n_lines, cursor=True):
        im = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((0, 0, W - 1, 27), 8, fill=(40, 40, 44))
        for k, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse((14 + k * 20, 8, 26 + k * 20, 20), fill=c)
        d.text((W // 2 - 60, 6), "tinyfuse demo", font=font, fill=dim)
        y = 28 + PAD
        d.text((PAD, y), prompt, font=bold, fill=green)
        x = PAD + d.textlength(prompt, font=bold)
        d.text((x, y), typed, font=font, fill=fg)
        if cursor and n_lines == 0:
            cx = x + d.textlength(typed, font=font)
            d.rectangle((cx + 1, y + 2, cx + 8, y + 16), fill=fg)
        for i, line in enumerate(out[:n_lines]):
            y += LH
            colr = accent if line.startswith(("graph", "  %", "  return", "}")) else fg
            if line.startswith(("max abs", "tinyfuse:")):
                colr = green
            d.text((PAD, y), line, font=font, fill=colr)
        if n_lines >= len(out):
            y += LH * 1.5
            d.text((PAD, y), prompt, font=bold, fill=green)
        return im

    frames, dur = [render("", 0)], [700]
    for k in range(1, len(cmd) + 1):
        frames.append(render(cmd[:k], 0))
        dur.append(55)
    frames.append(render(cmd, 0, cursor=False))
    dur.append(500)
    # replay output with the real pauses: compile first, timing at the end
    for n in range(1, len(out) + 1):
        frames.append(render(cmd, n, cursor=False))
        line = out[n - 1]
        dur.append(900 if line.startswith("compiled") else 1200 if line.startswith("max abs")
                   else 140)
    dur[-1] = 4000
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "demo.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0,
                   optimize=True)
    print("wrote", os.path.relpath(path, ROOT), f"({len(frames)} frames)")


FIGS = {"graph": fig_graph, "tiling": fig_tiling, "profile": fig_profile,
        "memory": fig_memory, "terminal": fig_terminal}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for w in which:
        FIGS[w]()
