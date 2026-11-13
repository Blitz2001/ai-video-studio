"""Generic country-year panel analysis. Works for any dataset that can be
shaped as entity/year/value (+ optional group, weight, x). Every number that
can appear in narration is produced here, once, with one rounding."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PanelFindings:
    first: int
    last: int
    n_entities: int
    world: dict[int, float]                       # year -> (weighted) mean
    groups: dict[str, dict[int, float]] = field(default_factory=dict)
    gains: list[tuple[str, float]] = field(default_factory=list)   # sorted desc
    declines: list[tuple[str, float]] = field(default_factory=list)  # entities that went down
    stall: tuple[str, int, int, float, float] | None = None   # group, y0, y1, lo, hi
    corr_log_x: float | None = None
    threshold: float | None = None
    above_first: int = 0
    above_last: int = 0
    weighted: bool = False


def _wmean(g: pd.DataFrame) -> float:
    if "weight" in g and g["weight"].notna().all():
        return float(np.average(g["value"], weights=g["weight"]))
    return float(g["value"].mean())


def analyse(df: pd.DataFrame) -> PanelFindings:
    counts = df.groupby("year")["entity"].nunique()
    good_years = counts[counts >= 0.6 * counts.max()].index
    df = df[df["year"].isin(good_years)]
    years = [int(y) for y in sorted(df["year"].unique())]
    first, last = int(years[0]), int(years[-1])
    weighted = "weight" in df.columns
    world = {int(y): round(_wmean(g), 1) for y, g in df.groupby("year")}

    groups: dict[str, dict[int, float]] = {}
    if "group" in df.columns:
        for (grp, y), g in df.groupby(["group", "year"]):
            groups.setdefault(str(grp), {})[int(y)] = round(_wmean(g), 1)

    wide = df.pivot_table(index="entity", columns="year", values="value")
    both = wide[[first, last]].dropna()
    delta = (both[last] - both[first]).round(1).sort_values(ascending=False)
    gains = [(str(e), float(v)) for e, v in delta.items()]
    declines = [(e, v) for e, v in reversed(gains) if v < 0]

    stall = None
    if groups and len(years) >= 3:
        best = None
        for grp, series in groups.items():
            ys = [y for y in years if y in series]
            for i in range(len(ys) - 2):
                window = [series[ys[i]], series[ys[i + 1]], series[ys[i + 2]]]
                spread = max(window) - min(window)
                if best is None or spread < best[0]:
                    best = (spread, grp, ys[i], ys[i + 2], min(window), max(window))
        if best and best[0] <= 0.5:
            stall = (str(best[1]), int(best[2]), int(best[3]), float(best[4]), float(best[5]))

    corr = None
    if "x" in df.columns:
        d = df[df["year"] == last].dropna(subset=["x"])
        d = d[d["x"] > 0]
        if len(d) > 10:
            corr = round(float(np.corrcoef(np.log10(d["x"]), d["value"])[0, 1]), 2)

    median_last = float(both[last].median())
    threshold = float(round(median_last, -1)) if median_last > 10 else None
    above_first = int((both[first] >= threshold).sum()) if threshold is not None else 0
    above_last = int((both[last] >= threshold).sum()) if threshold is not None else 0

    return PanelFindings(first, last, int(len(both)), world, groups, gains, declines, stall, corr,
                         threshold, above_first, above_last, weighted)
