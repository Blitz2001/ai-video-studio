"""Matplotlib animation scenes for concept_explainer.

The portrait flag (`_portrait`) switches multi-panel layouts to vertical
stacks so the 9:16 short gets a native 4:5 render instead of a shrunken 16:9.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ...media import theme
from ...media.anim import bullet_card, ease, frame_header, phase
from . import sims

LR_COLOURS = [theme.SERIES[0], theme.SERIES[2], theme.SERIES[1]]  # small, good, too big


def _contour(ax, xlim=(-5, 5), ylim=(-2.4, 2.4)):
    xs = np.linspace(*xlim, 300)
    ys = np.linspace(*ylim, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = sims.loss(X, Y)
    from matplotlib.colors import LinearSegmentedColormap

    valley = LinearSegmentedColormap.from_list("valley", ["#1c5cab", "#184f95", "#173a63", theme.SURFACE_2])
    ax.contourf(X, Y, np.log1p(Z), levels=18, cmap=valley)
    ax.contour(X, Y, np.log1p(Z), levels=18, colors=["#3a4a5e"], linewidths=0.8)
    ax.plot([0], [0], marker="*", ms=22, color=theme.TEXT, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _partial(path: np.ndarray, k: float) -> np.ndarray:
    """First fraction k of a polyline, interpolating the last segment."""
    n = len(path) - 1
    pos = k * n
    i = int(pos)
    if i >= n:
        return path
    f = pos - i
    return np.vstack([path[: i + 1], path[i] * (1 - f) + path[i + 1] * f])


# ----------------------------------------------------------------- gradient descent
def contour_descent(p: dict):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        rect = [0.06, 0.24, 0.88, 0.56] if p.get("_portrait") else [0.05, 0.24, 0.62, 0.58]
        ax = fig.add_axes(rect)
        _contour(ax)
        path = np.array(p["path"])
        line, = ax.plot([], [], color=theme.HIGHLIGHT, lw=3, zorder=4)
        ball, = ax.plot([], [], "o", ms=22, color=theme.HIGHLIGHT, mec=theme.TEXT, mew=2.5, zorder=6)
        lx, ly = (0.06, 0.17) if p.get("_portrait") else (0.71, 0.62)
        step_t = fig.text(lx, ly, "", fontsize=34, color=theme.TEXT)
        loss_t = fig.text(lx, ly - 0.07, "", fontsize=34, color=theme.TEXT_2)
        if not p.get("_portrait"):
            fig.text(0.71, 0.72, "loss = x² + 6y²", fontsize=28, color=theme.MUTED)

        def update(t):
            k = ease(phase(t, 0.05, 0.9))
            part = _partial(path, k * p.get("show_fraction", 1.0))
            line.set_data(part[:, 0], part[:, 1])
            ball.set_data([part[-1, 0]], [part[-1, 1]])
            step_t.set_text(f"step {int(k * p.get('show_fraction', 1.0) * (len(path) - 1))}")
            loss_t.set_text(f"loss {sims.loss(*part[-1]):.3f}")
        return update
    return setup


def gradient_1d(p: dict):
    lr, x0, n = p.get("lr", 0.3), p.get("x0", -2.6), p.get("steps", 5)
    xs = [x0]
    for _ in range(n):
        xs.append(xs[-1] - lr * 2 * xs[-1])

    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        ax = fig.add_axes([0.10, 0.24, 0.80, 0.52] if p.get("_portrait") else [0.08, 0.24, 0.84, 0.56])
        grid = np.linspace(-3, 3, 300)
        ax.plot(grid, grid ** 2, color=theme.TEXT_2, lw=3)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-0.8, 9.5)
        ax.set_xlabel("parameter w")
        ax.set_ylabel("loss")
        tangent, = ax.plot([], [], color=theme.SERIES[0], lw=2.5, ls="--")
        dot, = ax.plot([], [], "o", ms=20, color=theme.HIGHLIGHT, mec=theme.TEXT, mew=2)
        trail, = ax.plot([], [], "o", ms=9, color=theme.HIGHLIGHT, alpha=0.4)
        label = ax.text(0, 0, "", fontsize=24, color=theme.TEXT)
        arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=30, color=theme.SERIES[2], lw=3)
        ax.add_patch(arrow)

        def update(t):
            pos = ease(phase(t, 0.05, 0.92)) * n
            i = min(int(pos), n - 1)
            f = pos - i
            # first half of each step: show slope; second half: move
            x = xs[i] if f < 0.5 else xs[i] + (xs[i + 1] - xs[i]) * ease((f - 0.5) * 2)
            slope = 2 * x
            tx = np.array([x - 1.2, x + 1.2])
            tangent.set_data(tx, x ** 2 + slope * (tx - x))
            dot.set_data([x], [x ** 2])
            trail.set_data(xs[: i + 1], [v ** 2 for v in xs[: i + 1]])
            label.set_position((x + 0.25, x ** 2 + 0.8))
            label.set_text(f"slope {slope:+.2f}")
            arrow.set_positions((x, -0.4), (x - lr * slope * 0.9, -0.4) if abs(slope) > 0.05 else (x, -0.4))
        return update
    return setup


def lr_compare(p: dict):
    runs = p["runs"]  # [{lr, path, label}]

    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        portrait = p.get("_portrait")
        axes = []
        for i in range(len(runs)):
            rect = [0.06, 0.62 - i * 0.205, 0.88, 0.18] if portrait else [0.04 + i * 0.32, 0.30, 0.29, 0.42]
            ax = fig.add_axes(rect)
            _contour(ax, xlim=(-6, 6), ylim=(-3, 3))
            axes.append(ax)
        arts = []
        for ax, run, col in zip(axes, runs, LR_COLOURS):
            path = np.array(run["path"])
            line, = ax.plot([], [], color=col, lw=2.5, zorder=4)
            ball, = ax.plot([], [], "o", ms=14, color=col, mec=theme.TEXT, mew=2, zorder=6)
            ax.set_title(run["label"], fontsize=24 if not portrait else 22, color=theme.TEXT, loc="left")
            txt = ax.text(0.02, 0.04, "", transform=ax.transAxes, fontsize=20, color=theme.TEXT)
            arts.append((path, line, ball, txt, run))

        def update(t):
            k = ease(phase(t, 0.05, 0.9))
            for path, line, ball, txt, run in arts:
                part = _partial(path, k)
                clipped = np.clip(part, [-6, -3], [6, 3])
                line.set_data(clipped[:, 0], clipped[:, 1])
                ball.set_data([clipped[-1, 0]], [clipped[-1, 1]])
                step = int(k * (len(path) - 1))
                loss = sims.loss(*part[-1])
                txt.set_text(f"step {step}   loss {loss:,.3f}" if loss < 1e3 else f"step {step}   loss {loss:,.0f}  ↑")
        return update
    return setup


def sgd_vs_gd(p: dict):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        ax = fig.add_axes([0.06, 0.24, 0.88, 0.50] if p.get("_portrait") else [0.12, 0.24, 0.76, 0.50])
        _contour(ax)
        full, noisy = np.array(p["full"]), np.array(p["noisy"])
        l1, = ax.plot([], [], color=theme.SERIES[0], lw=3, label="full-batch gradient")
        l2, = ax.plot([], [], color=theme.SERIES[1], lw=2.5, label="mini-batch (stochastic)")
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, labelcolor=theme.TEXT_2)

        def update(t):
            k = ease(phase(t, 0.05, 0.9))
            a, b = _partial(full, k), _partial(noisy, k)
            l1.set_data(a[:, 0], a[:, 1])
            l2.set_data(b[:, 0], b[:, 1])
        return update
    return setup


# ----------------------------------------------------------------- sorting
def sort_race(p: dict):
    arr = p["array"]
    bf, bc = sims.bubble_trace(arr)
    mf, mc = sims.merge_trace(arr)

    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        portrait = p.get("_portrait")
        rects = ([[0.08, 0.50, 0.84, 0.24], [0.08, 0.20, 0.84, 0.24]] if portrait
                 else [[0.05, 0.28, 0.42, 0.48], [0.53, 0.28, 0.42, 0.48]])
        panels = []
        for rect, name, frames, total in zip(rects, ["Bubble sort", "Merge sort"], [bf, mf], [bc, mc]):
            ax = fig.add_axes(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_ylim(0, len(arr) + 1)
            for sp in ax.spines.values():
                sp.set_visible(False)
            bars = ax.bar(range(len(arr)), arr, color=theme.MUTED, width=0.78)
            ax.set_title(name, loc="left", fontsize=28, color=theme.TEXT)
            cnt = ax.text(1.0, 1.02, "", transform=ax.transAxes, ha="right", fontsize=24, color=theme.TEXT_2)
            panels.append((bars, frames, total, cnt))
        longest = max(bc, mc)

        def update(t):
            comps_now = ease(phase(t, 0.04, 0.92)) * longest  # same clock: one comparison = one tick
            for bars, frames, total, cnt in panels:
                idx = 0
                for j, (_, _, c) in enumerate(frames):
                    if c <= comps_now:
                        idx = j
                state, (a, b), c = frames[idx]
                done = comps_now >= total
                for k, (bar, h) in enumerate(zip(bars, state)):
                    bar.set_height(h)
                    bar.set_color(theme.SERIES[2] if done else (theme.HIGHLIGHT if k in (a, b) else theme.MUTED))
                cnt.set_text(f"{min(int(comps_now), total)} comparisons" + ("  ✓" if done else ""))
        return update
    return setup


def big_o(p: dict):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        ax = fig.add_axes([0.13, 0.24, 0.80, 0.50] if p.get("_portrait") else [0.08, 0.24, 0.84, 0.54])
        n = np.linspace(2, 1000, 400)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000 ** 2 / 2 * 1.08)
        ax.set_xlabel("items to sort (n)")
        ax.set_ylabel("comparisons")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        l1, = ax.plot([], [], color=theme.SERIES[1], lw=3.5)
        l2, = ax.plot([], [], color=theme.SERIES[2], lw=3.5)
        t1 = ax.text(0, 0, "", color=theme.TEXT, fontsize=24)
        t2 = ax.text(0, 0, "", color=theme.TEXT, fontsize=24)

        def update(t):
            k = max(2, int(ease(phase(t, 0.05, 0.85)) * len(n)))
            x = n[:k]
            q, l = x * (x - 1) / 2, x * np.log2(x)
            l1.set_data(x, q)
            l2.set_data(x, l)
            t1.set_position((x[-1] - 250, q[-1] * 0.92))
            t1.set_text(f"bubble ~ n²/2: {q[-1]:,.0f}")
            t2.set_position((x[-1] - 250, l[-1] + 25000))
            t2.set_text(f"merge ~ n log n: {l[-1]:,.0f}")
        return update
    return setup


# ----------------------------------------------------------------- RAG
def rag_pipeline(p: dict):
    stages = p["stages"]

    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        portrait = p.get("_portrait")
        boxes = []
        n = len(stages)
        for i, (name, sub) in enumerate(stages):
            if portrait:
                x, y, w, h = 0.12, 0.74 - i * 0.095, 0.76, 0.075
            else:
                x, y, w, h = 0.04 + i * (0.92 / n), 0.46, 0.92 / n - 0.025, 0.2
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                 transform=fig.transFigure, facecolor=theme.SURFACE_2, edgecolor=theme.GRID, lw=2)
            fig.patches.append(box)
            t1 = fig.text(x + w / 2, y + h * (0.62 if not portrait else 0.6), name, ha="center", va="center",
                          fontsize=26 if not portrait else 24, color=theme.TEXT, fontweight="bold")
            t2 = fig.text(x + w / 2, y + h * (0.3 if not portrait else 0.25), sub, ha="center", va="center",
                          fontsize=17 if not portrait else 17, color=theme.TEXT_2)
            boxes.append((box, t1, t2, (x, y, w, h)))
        packet = FancyBboxPatch((0, 0), 0.012, 0.02, boxstyle="circle,pad=0.004", transform=fig.transFigure,
                                color=theme.HIGHLIGHT)
        fig.patches.append(packet)

        def update(t):
            for i, (box, t1, t2, _) in enumerate(boxes):
                a = ease(phase(t, 0.03 + i * 0.1, 0.12 + i * 0.1))
                for art in (box, t1, t2):
                    art.set_alpha(a)
                lit = phase(t, 0.6, 0.95) * n
                box.set_edgecolor(theme.ACCENT if i < lit else theme.GRID)
            k = phase(t, 0.6, 0.95) * (n - 1)
            i = min(int(k), n - 2)
            f = k - i
            (x0, y0, w0, h0), (x1, y1, w1, h1) = boxes[i][3], boxes[i + 1][3]
            if portrait:
                px, py = x0 + w0 + 0.02, (y0 + h0 / 2) * (1 - f) + (y1 + h1 / 2) * f
            else:
                px, py = (x0 + w0 / 2) * (1 - f) + (x1 + w1 / 2) * f, y0 - 0.05
            packet.set_bounds(px, py, 0.012, 0.02)
            packet.set_alpha(1.0 if t > 0.6 else 0.0)
        return update
    return setup


def retrieval_scores(p: dict):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        portrait = p.get("_portrait")
        fig.text(0.07 if portrait else 0.05, 0.78, f"Query: “{p['query']}”", fontsize=26, color=theme.TEXT)
        ax = fig.add_axes([0.36, 0.24, 0.56, 0.46] if portrait else [0.25, 0.24, 0.66, 0.46])
        docs, scores = p["docs"][::-1], np.array(p["scores"][::-1])
        top = p["docs"][int(np.argmax(p["scores"]))]
        ax.set_yticks(range(len(docs)), docs, fontsize=22, color=theme.TEXT_2)
        ax.set_xlim(0, max(scores) * 1.3 + 0.05)
        ax.set_xlabel("similarity to the query")
        ax.grid(axis="y", visible=False)
        bars = ax.barh(range(len(docs)), np.zeros(len(docs)), height=0.6,
                       color=[theme.SERIES[0] if d == top else theme.MUTED for d in docs])
        txt = [ax.text(0, i, "", va="center", fontsize=20, color=theme.TEXT) for i in range(len(docs))]

        def update(t):
            k = ease(phase(t, 0.1, 0.6))
            for b, s, tx, i in zip(bars, scores, txt, range(len(docs))):
                b.set_width(s * k)
                tx.set_position((s * k + 0.01, i))
                tx.set_text(f"{s * k:.2f}")
        return update
    return setup


SCENES = {"contour_descent": contour_descent, "gradient_1d": gradient_1d, "lr_compare": lr_compare,
          "sgd_vs_gd": sgd_vs_gd, "sort_race": sort_race, "big_o": big_o, "rag_pipeline": rag_pipeline,
          "retrieval_scores": retrieval_scores, "bullet_card": bullet_card}
