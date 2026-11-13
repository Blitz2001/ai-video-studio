"""Animated chart scenes for data_story. Each takes JSON params and returns a
`setup(fig) -> update(t)` for avs.media.anim.render."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...media import theme
from ...media.anim import bullet_card, ease, frame_header, phase

PLOT_RECT = [0.07, 0.25, 0.86, 0.50]  # leaves the bottom quarter for captions
PLOT_RECT_PORTRAIT = [0.13, 0.22, 0.80, 0.50]


def plot_rect(p: dict) -> list[float]:
    return PLOT_RECT_PORTRAIT if p.get("_portrait") else PLOT_RECT


def group_colours(names: list[str]) -> dict[str, str]:
    """Fixed order -> fixed colour, so an entity keeps its colour in every chart."""
    return {n: theme.SERIES[i % len(theme.SERIES)] for i, n in enumerate(sorted(names))}


def _fmt(v: float, unit: str) -> str:
    return f"{v:.1f}{unit}" if unit == "%" else f"{v:.1f}"


# --------------------------------------------------------------------------- 1. headline counter
def stat_count(p: dict):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        years = [int(y) for y in p["world"]]
        vals = [p["world"][str(y)] if str(y) in p["world"] else p["world"][y] for y in years]
        portrait = p.get("_portrait")
        top = 0.70 if portrait else 0.62
        big = fig.text(0.07, top, "", fontsize=170, fontweight="bold", color=theme.TEXT, va="center")
        unit = fig.text(0.07, top - 0.13 if portrait else 0.45, p["unit_label"], fontsize=34, color=theme.TEXT_2, va="center")
        yr = fig.text(0.07, top - 0.19 if portrait else 0.36, "", fontsize=30, color=theme.MUTED, va="center")
        ax = fig.add_axes([0.12, 0.22, 0.80, 0.24] if portrait else [0.52, 0.30, 0.42, 0.42])
        ax.set_xlim(years[0] - 1, years[-1] + 1)
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.15)
        ax.grid(axis="x", visible=False)
        line, = ax.plot([], [], color=theme.ACCENT, lw=4, solid_capstyle="round")
        dot, = ax.plot([], [], "o", color=theme.ACCENT, ms=14, mec=theme.SURFACE, mew=3)
        unit.set_alpha(1)

        def update(t):
            k = ease(phase(t, 0.05, 0.85))
            xs = np.linspace(years[0], years[-1], 200)
            ys = np.interp(xs, years, vals)
            n = max(1, int(round(k * 199)) + 1)
            line.set_data(xs[:n], ys[:n])
            dot.set_data([xs[n - 1]], [ys[n - 1]])
            big.set_text(f"{ys[n - 1]:.1f}")
            yr.set_text(f"{p['label']} · {int(round(xs[n - 1]))}")
        return update
    return setup


# --------------------------------------------------------------------------- 2. group lines
def group_lines(p: dict):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        ax = fig.add_axes(plot_rect(p))
        groups = p["groups"]
        colours = group_colours(list(groups))
        years = sorted({int(y) for s in groups.values() for y in s})
        allv = [v for s in groups.values() for v in s.values()]
        ax.set_xlim(years[0], years[-1] + (years[-1] - years[0]) * 0.16)
        ax.set_ylim(min(allv) - 4, max(allv) + 4)
        ax.set_ylabel(p.get("ylabel", ""))
        ax.grid(axis="x", visible=False)
        ax.spines["bottom"].set_color(theme.GRID)
        lines, labels = {}, {}
        for name in sorted(groups):
            lines[name], = ax.plot([], [], color=colours[name], lw=3.5, solid_capstyle="round", label=name)
            labels[name] = ax.text(0, 0, "", color=theme.TEXT_2, fontsize=20, va="center")
        ax.legend(loc="lower left", ncol=3 if p.get("_portrait") else len(groups), bbox_to_anchor=(0, 1.01), handlelength=1.2,
                  labelcolor=theme.TEXT_2)
        series = {n: (np.array([int(y) for y in s]), np.array(list(s.values()))) for n, s in groups.items()}

        def update(t):
            k = ease(phase(t, 0.03, 0.8))
            x_end = years[0] + k * (years[-1] - years[0])
            ends = []
            for name, (xs, ys) in series.items():
                order = np.argsort(xs)
                xs, ys = xs[order], ys[order]
                grid = np.linspace(years[0], x_end, 150)
                yy = np.interp(grid, xs, ys)
                lines[name].set_data(grid, yy)
                ends.append([name, yy[-1]])
            # direct labels, nudged apart vertically
            ends.sort(key=lambda e: e[1])
            for i in range(1, len(ends)):
                if ends[i][1] - ends[i - 1][1] < 2.2:
                    ends[i][1] = ends[i - 1][1] + 2.2
            for name, y in ends:
                labels[name].set_position((x_end + 0.8, y))
                labels[name].set_text(f"{name} {lines[name].get_ydata()[-1]:.1f}")
        return update
    return setup


# --------------------------------------------------------------------------- 3. dumbbell
def dumbbell(p: dict):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        ax = fig.add_axes([0.24, 0.22, 0.66, 0.50] if p.get("_portrait") else [0.16, 0.25, 0.74, 0.50])
        items = sorted(p["groups"].items(), key=lambda kv: kv[1][1])
        colours = group_colours([k for k, _ in items])
        allv = [v for _, pair in items for v in pair]
        ax.set_xlim(min(allv) - 6, max(allv) + 8)
        ax.set_ylim(-0.7, len(items) - 0.3)
        ax.set_yticks(range(len(items)), [k for k, _ in items], fontsize=24, color=theme.TEXT_2)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(p.get("xlabel", ""))
        arts = []
        for i, (name, (a, b)) in enumerate(items):
            ln, = ax.plot([], [], color=colours[name], lw=6, solid_capstyle="round", alpha=0.55)
            d0, = ax.plot([a], [i], "o", ms=18, color=theme.MUTED, mec=theme.SURFACE, mew=3)
            d1, = ax.plot([], [], "o", ms=22, color=colours[name], mec=theme.SURFACE, mew=3)
            tx = ax.text(0, i, "", fontsize=21, color=theme.TEXT, va="center")
            arts.append((a, b, i, ln, d1, tx))
        ax.text(0.0, 1.04, f"● {p['y0']}", color=theme.MUTED, transform=ax.transAxes, fontsize=22)
        ax.text(0.12, 1.04, f"● {p['y1']}", color=theme.TEXT, transform=ax.transAxes, fontsize=22)

        def update(t):
            for j, (a, b, i, ln, d1, tx) in enumerate(arts):
                k = ease(phase(t, 0.08 + j * 0.08, 0.45 + j * 0.08))
                x = a + (b - a) * k
                ln.set_data([a, x], [i, i])
                d1.set_data([x], [i])
                tx.set_position((max(a, x) + 1.2, i))
                tx.set_text(f"{a:.1f} → {x:.1f}" if k > 0 else "")
        return update
    return setup


# --------------------------------------------------------------------------- 4. bars
def bar_grow(p: dict):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        ax = fig.add_axes([0.34, 0.22, 0.56, 0.52] if p.get("_portrait") else [0.24, 0.25, 0.66, 0.50])
        items = p["items"][::-1]
        names = [n for n, _ in items]
        vals = np.array([v for _, v in items])
        ax.set_yticks(range(len(items)), names, fontsize=21, color=theme.TEXT_2)
        ax.set_xlim(min(0, vals.min()) * 1.25 - 1, vals.max() * 1.18)
        ax.set_ylim(-0.6, len(items) - 0.4)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=theme.MUTED, lw=1.5)
        ax.set_xlabel(p.get("xlabel", ""))
        colours = [theme.SERIES[1] if v < 0 else theme.SERIES[0] for v in vals]
        bars = ax.barh(range(len(items)), np.zeros(len(items)), color=colours, height=0.62)
        texts = [ax.text(0, i, "", va="center", fontsize=20, color=theme.TEXT) for i in range(len(items))]

        def update(t):
            for i, (b, v, tx) in enumerate(zip(bars, vals, texts)):
                k = ease(phase(t, 0.04 + 0.03 * (len(items) - i), 0.5 + 0.03 * (len(items) - i)))
                w = v * k
                b.set_width(w)
                tx.set_position((w + 0.6 if v >= 0 else 0.6, i))  # negatives labelled right of zero, clear of names
                tx.set_ha("left")
                tx.set_text(f"{w:+.1f}" if k > 0 else "")
        return update
    return setup


# --------------------------------------------------------------------------- 5. bubbles over time
def bubble_time(p: dict, data: pd.DataFrame):
    def setup(fig):
        frame_header(fig, p["title"], p.get("source"))
        ax = fig.add_axes(plot_rect(p))
        ax.set_xscale("log")
        df = data.dropna(subset=["x"])
        years = sorted(df["year"].unique())
        wide = {c: df.pivot_table(index="entity", columns="year", values=c) for c in ("x", "value", "weight")}
        ents = wide["x"].dropna().index
        grp = df.drop_duplicates("entity").set_index("entity")["group"].reindex(ents)
        colours = group_colours(sorted(df["group"].unique()))
        highlight = set(p.get("highlight", []))
        col = [colours[g] if g in highlight else theme.MUTED for g in grp]
        alpha = np.array([0.85 if g in highlight else 0.35 for g in grp])
        ax.set_xlim(df["x"].min() * 0.7, df["x"].max() * 1.4)
        ax.set_ylim(df["value"].min() - 3, df["value"].max() + 3)
        ax.set_xlabel(p.get("xlabel", ""))
        ax.set_ylabel(p.get("ylabel", ""))
        year_txt = ax.text(0.98, 0.06, "", transform=ax.transAxes, ha="right", fontsize=110,
                           color=theme.GRID, fontweight="bold", zorder=0)
        sc = ax.scatter([], [], s=[], zorder=3, edgecolors=theme.SURFACE, linewidths=1.5)
        handles = [ax.scatter([], [], s=200, color=colours[g], label=g) for g in sorted(highlight)]
        handles.append(ax.scatter([], [], s=200, color=theme.MUTED, label="Other continents"))
        ax.legend(handles=handles, loc="lower left", ncol=len(handles), bbox_to_anchor=(0, 1.01),
                  labelcolor=theme.TEXT_2)
        ax.grid(which="minor", visible=False)
        X = wide["x"].loc[ents, years].to_numpy()
        Y = wide["value"].loc[ents, years].to_numpy()
        S = wide["weight"].loc[ents, years].to_numpy()
        smax = np.nanmax(S)

        def update(t):
            pos = ease(phase(t, 0.04, 0.92)) * (len(years) - 1)
            i = min(int(pos), len(years) - 2)
            f = pos - i
            x = np.exp(np.log(X[:, i]) * (1 - f) + np.log(X[:, i + 1]) * f)
            y = Y[:, i] * (1 - f) + Y[:, i + 1] * f
            s = S[:, i] * (1 - f) + S[:, i + 1] * f
            sc.set_offsets(np.c_[x, y])
            sc.set_sizes(30 + 4200 * (s / smax))
            sc.set_facecolors([(*_rgb(c), a) for c, a in zip(col, alpha)])
            year_txt.set_text(str(int(round(years[i] + f * (years[i + 1] - years[i])))))
        return update
    return setup


def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


SCENES = {"stat_count": stat_count, "group_lines": group_lines, "dumbbell": dumbbell,
          "bar_grow": bar_grow, "source_card": bullet_card}
