from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ...media.anim import estimate_seconds, render
from ...models import Asset, CheckResult, Fact, ResearchPack, Script, Source
from ..base import FormatPlugin, RunContext
from ..writing import Beat
from .analysis import PanelFindings, analyse
from .scenes import SCENES, bubble_time
from .sources import file_sha256, load_dataset


def _registry(folder: Path) -> dict[str, dict]:
    return yaml.safe_load((folder / "datasets.yaml").read_text())


def pick_dataset(topic: str, registry: dict[str, dict]) -> tuple[str, dict]:
    t = topic.lower().strip()
    if t in registry:
        return t, registry[t]
    best, score = None, 0
    for key, spec in registry.items():
        words = " ".join([spec["title"], *spec.get("keywords", [])]).lower()
        s = sum(1 for w in t.split() if len(w) > 2 and w in words)
        if s > score:
            best, score = key, s
    if best is None:
        raise ValueError(f"no dataset matches topic {topic!r}; add one to datasets.yaml")
    return best, registry[best]


class DataStory(FormatPlugin):
    name = "data_story"
    description = "Analyse a real open dataset (OWID, World Bank, Kaggle) and animate the findings."
    default_topic = "gapminder"
    visual_kinds = ("stat_count", "group_lines", "dumbbell", "bar_grow", "bubble_time", "source_card")

    # ------------------------------------------------------------------ research
    def research(self, topic: str, ctx: RunContext) -> ResearchPack:
        key, spec = pick_dataset(topic, _registry(self.folder))
        df, path = load_dataset(spec, ctx.settings.root)
        tidy = ctx.assets_dir / "data.csv"
        df.to_csv(tidy, index=False)
        f = analyse(df)
        sha = file_sha256(path)
        unit = spec.get("unit", "")
        label = spec.get("metric_label", "value")
        u = (lambda v: f"{v:.1f}%") if unit == "%" else (lambda v: f"{v:.1f} {unit}".strip())

        sources = [
            Source(id="dataset", title=spec["title"], kind="dataset", url=spec["url"], publisher=spec["publisher"],
                   licence=spec["licence"], reference=f"snapshot sha256 {sha[:16]}"),
            Source(id="analysis", title=f"Our analysis of the {spec['publisher']} data", kind="computation",
                   reference="avs/formats/data_story/analysis.py" + (" (population-weighted means)" if f.weighted else "")),
        ]
        w0, w1 = f.world[f.first], f.world[f.last]
        facts = [
            Fact(id="coverage", source_id="dataset", text=f"The dataset covers {f.n_entities} countries from {f.first} to {f.last}."),
            Fact(id="world", source_id="analysis",
                 text=f"{'Population-weighted average' if f.weighted else 'Average'} {label} was {u(w0)} in {f.first} and {u(w1)} in {f.last}."),
            Fact(id="world_change", source_id="analysis", text=f"The average changed by {w1 - w0:+.1f} between {f.first} and {f.last}."),
        ]
        g_first = {g: s[f.first] for g, s in f.groups.items() if f.first in s and f.last in s}
        g_last = {g: f.groups[g][f.last] for g in g_first}
        extra: dict = {}
        if g_first:
            for g in g_first:
                facts.append(Fact(id=f"group_{g}", source_id="analysis",
                                  text=f"{g}: {u(g_first[g])} in {f.first}, {u(g_last[g])} in {f.last}."))
            gains = {g: round(g_last[g] - g_first[g], 1) for g in g_first}
            best = max(gains, key=gains.get)
            low0, low1 = min(g_first, key=g_first.get), min(g_last, key=g_last.get)
            top1 = max(g_last, key=g_last.get)
            gap0, gap1 = round(g_first[top1] - g_first[best], 1), round(g_last[top1] - g_last[best], 1)
            facts += [
                Fact(id="group_best", source_id="analysis",
                     text=f"{best} had the largest gain of any group: from {u(g_first[best])} to {u(g_last[best])}."),
                Fact(id="group_gap", source_id="analysis",
                     text=f"The gap between {best} and {top1} went from {u(gap0)} in {f.first} to {u(gap1)} in {f.last}."),
            ]
            if all(v > 0 for v in gains.values()):
                facts.append(Fact(id="all_up", source_id="analysis",
                                  text=f"Every group's average was higher in {f.last} than in {f.first}."))
            if low0 == low1:
                facts.append(Fact(id="group_low", source_id="analysis",
                                  text=f"{low0} was lowest in {f.first} ({u(g_first[low0])}) and lowest in {f.last} ({u(g_last[low0])})."))
            extra.update(best=best, top1=top1, low=low0 if low0 == low1 else None, gap0=gap0, gap1=gap1,
                         best_v=[g_first[best], g_last[best]], low_v=[g_first[low0], g_last[low0]])
            if f.stall:
                g, y0, y1, lo, hi = f.stall
                facts.append(Fact(id="stall", source_id="analysis",
                                  text=f"{g}'s average stayed between {lo:.1f} and {hi:.1f} from {y0} to {y1}."))
        for i, (e, v) in enumerate(f.gains[:3]):
            facts.append(Fact(id=f"gain_{i}", source_id="analysis",
                              text=f"{e} changed by {v:+.1f} ({abs(v):.1f}) between {f.first} and {f.last}"
                                   + (", the largest gain of any country." if i == 0 else ".")))
        if f.declines:
            e, v = f.declines[0]
            facts.append(Fact(id="decline", source_id="analysis",
                              text=f"{e} was {abs(v):.1f} lower in {f.last} than in {f.first}, the largest fall of any country."))
        if f.corr_log_x is not None:
            facts.append(Fact(id="corr", source_id="analysis",
                              text=f"In {f.last} the correlation between log {spec.get('x_label', 'x')} and {label} was {f.corr_log_x:.2f}."))
        if f.threshold is not None:
            facts.append(Fact(id="threshold", source_id="analysis",
                              text=f"{f.above_first} of {f.n_entities} countries were at {f.threshold:.0f} or more in {f.first}; "
                                   f"{f.above_last} were in {f.last}."))

        src_line = f"Source: {spec['publisher']} ({spec['licence'].replace('-', ' ')}) · analysis: {ctx.brand}"
        storyboard = self._storyboard(f, spec, extra, src_line, tidy)
        return ResearchPack(format=self.name, topic=key, angle=spec.get("angle", ""), sources=sources, facts=facts,
                            data={"dataset_key": key, "spec": spec, "sha256": sha, "tidy_csv": "assets/data.csv",
                                  "storyboard": storyboard, "findings": {
                                      "first": f.first, "last": f.last, "n": f.n_entities, "w0": w0, "w1": w1,
                                      "weighted": f.weighted, "gains": f.gains[:3], "declines": f.declines[:1],
                                      "stall": f.stall, "corr": f.corr_log_x, "threshold": f.threshold,
                                      "above_first": f.above_first, "above_last": f.above_last, **extra}})

    def _storyboard(self, f: PanelFindings, spec: dict, extra: dict, src_line: str, tidy: Path) -> list[dict]:
        label, unit = spec.get("metric_label", "value"), spec.get("unit", "")
        unit_label = f"{label} ({unit})" if unit else label
        beats = [{"id": "hook", "on_screen_text": f"{label.capitalize()}, {f.first} → {f.last}", "in_short": True,
                  "facts": ["coverage", "world", "world_change"],
                  "visual": {"kind": "stat_count", "params": {
                      "title": f"Average {label}", "source": src_line, "unit_label": unit_label,
                      "label": "population-weighted" if f.weighted else "average",
                      "world": {str(k): v for k, v in f.world.items()}}}}]
        if f.groups:
            beats.append({"id": "trend", "on_screen_text": "Every continent, side by side", "in_short": True,
                          "facts": ["all_up", "group_best", "stall"],
                          "visual": {"kind": "group_lines", "params": {
                              "title": f"{label.capitalize()} by {spec.get('group', 'group')}", "source": src_line,
                              "ylabel": unit_label, "groups": {g: {str(y): v for y, v in s.items()} for g, s in f.groups.items()}}}})
            g_pairs = {g: [s[f.first], s[f.last]] for g, s in f.groups.items() if f.first in s and f.last in s}
            beats.append({"id": "gap", "on_screen_text": "The gap is closing", "in_short": False,
                          "facts": ["group_gap", "group_low"],
                          "visual": {"kind": "dumbbell", "params": {
                              "title": f"{f.first} vs {f.last}", "source": src_line, "groups": g_pairs,
                              "y0": f.first, "y1": f.last, "xlabel": unit_label}}})
        items = [[e, v] for e, v in f.gains[:8]] + [[e, v] for e, v in f.declines[:3]]
        beats.append({"id": "countries", "on_screen_text": "Biggest movers", "in_short": True,
                      "facts": ["gain_0", "gain_1", "gain_2", "decline"],
                      "visual": {"kind": "bar_grow", "params": {
                          "title": f"Change in {label}, {f.first}–{f.last}", "source": src_line, "items": items,
                          "xlabel": f"change ({unit})" if unit else "change"}}})
        if f.corr_log_x is not None:
            beats.append({"id": "wealth", "on_screen_text": "Is it just money?", "in_short": False,
                          "facts": ["corr", "threshold"],
                          "visual": {"kind": "bubble_time", "params": {
                              "title": f"Income vs {label}", "source": src_line, "xlabel": spec.get("x_label", ""),
                              "ylabel": unit_label, "highlight": [x for x in [extra.get("best"), extra.get("low")] if x]},
                              }, "hero": True})
        beats.append({"id": "caveats", "on_screen_text": "Read this before you share it", "in_short": False,
                      "facts": [],
                      "visual": {"kind": "source_card", "params": {
                          "title": "What this data can and can't tell us",
                          "lines": [f"Averages hide differences inside each country",
                                    "Correlation is not causation",
                                    f"Data: {spec['publisher']} ({spec['licence']})",
                                    "Dataset link + analysis code in the description"]}}})
        return beats

    # ------------------------------------------------------------------ offline writer
    def fallback_script(self, research: ResearchPack, ctx: RunContext) -> Script:
        d = research.data["findings"]
        spec = research.data["spec"]
        label, unit = spec.get("metric_label", "value"), spec.get("unit", "")
        u = (lambda v: f"{v:.1f}%") if unit == "%" else (lambda v: f"{v:.1f} {unit}".strip())
        weighted = " weighted by population," if d["weighted"] else ""
        has = {f.id for f in research.facts}
        segs = []
        for beat in research.data["storyboard"]:
            b = Beat(beat)
            if beat["id"] == "hook":
                b.say(f"This is data on {d['n']} countries from {d['first']} to {d['last']}.", "dataset")
                b.say(f"In {d['first']}, average {label},{weighted} was {u(d['w0'])}.", "analysis")
                b.say(f"By {d['last']} it was {u(d['w1'])}, a change of {d['w1'] - d['w0']:+.1f}.", "analysis")
                b.say("So where did that change come from, and who got left behind?")
            elif beat["id"] == "trend":
                if "all_up" in has:
                    b.say(f"Split by continent, every continent's average was higher in {d['last']} than in {d['first']}.", "analysis")
                g0, g1 = d["best_v"]
                b.say(f"{d['best']} had the largest gain of any continent: from {u(g0)} to {u(g1)}.", "analysis")
                if "stall" in has:
                    g, y0, y1, lo, hi = d["stall"]
                    b.say(f"Meanwhile, {g}'s average stayed between {lo:.1f} and {hi:.1f} from {y0} to {y1}.", "analysis")
                    b.say("For a decade, it barely moved.")
            elif beat["id"] == "gap":
                b.say("Put the first year next to the last, and the gaps tell a different story.")
                b.say(f"The gap between {d['best']} and {d['top1']} went from {u(d['gap0'])} to {u(d['gap1'])}.", "analysis")
                if d.get("low") and "group_low" in has:
                    v0, v1 = d["low_v"]
                    b.say(f"But {d['low']} was lowest at the start, at {v0:.1f}, and still lowest at the end, at {v1:.1f}.", "analysis")
            elif beat["id"] == "countries":
                (e0, v0), (e1, v1), (e2, v2) = d["gains"][:3]
                b.say(f"Country by country, {e0} gained the most: {abs(v0):.1f} {unit}.".replace(" .", "."), "analysis")
                b.say(f"{e1} gained {abs(v1):.1f}, and {e2} {abs(v2):.1f}.", "analysis")
                if d["declines"]:
                    e, v = d["declines"][0]
                    b.say("Not everyone moved forward.")
                    b.say(f"In {e}, {label} was {abs(v):.1f} {unit} lower in {d['last']} than in {d['first']}.".replace(" .", "."), "analysis")
            elif beat["id"] == "wealth":
                b.say("Each bubble is a country, sized by population, and they all drift up and to the right.")
                b.say(f"In {d['last']}, the correlation between log income per person and {label} was {d['corr']:.2f}.", "analysis")
                if d.get("threshold") is not None:
                    b.say(f"In {d['first']}, {d['above_first']} of {d['n']} countries were at {d['threshold']:.0f} {unit} or more; "
                          f"in {d['last']}, {d['above_last']} were.".replace(" or more", " or more"), "analysis")
                b.say("That link is strong, but a correlation alone can't tell us what causes what.")
            elif beat["id"] == "caveats":
                b.say(f"These are averages, so they hide big differences inside each country.")
                b.say(f"The data comes from the {spec['publisher']}, and the dataset link and the analysis code are in the description.")
                b.say(f"I'm {ctx.brand}. See you in the next one.")
            segs.append(b.segment())
        title_change = d["w1"] - d["w0"]
        title = (f"How the World Added {title_change:.0f} Years of Life ({d['first']}–{d['last']})"
                 if unit == "years" else f"{label.capitalize()}: {d['first']}–{d['last']} in Data")
        return Script(format=self.name, topic=research.topic, title=title,
                      thumbnail_text=f"+{title_change:.0f} {unit} of life" if unit == "years" else label,
                      summary=(f"An animated look at {d['n']} countries in the {spec['publisher']} data: who gained, "
                               f"who stalled, and what income does and doesn't explain."),
                      tags=["data visualization", "data science", label, "gapminder", "open data", "python", "matplotlib"],
                      segments=segs, written_by="offline-template")

    # ------------------------------------------------------------------ assets
    def generate_assets(self, script: Script, research: ResearchPack, ctx: RunContext) -> list[Asset]:
        tidy = pd.read_csv(ctx.run.path / research.data["tidy_csv"])
        fps = int(ctx.settings.get("render.fps", 30))
        beats = {b["id"]: b for b in research.data["storyboard"]}
        out = []
        make_short = ctx.settings.get("render.make_short", True)
        for seg in script.segments:
            kind = seg.visual.kind
            dur = estimate_seconds(seg.narration)
            variants = [("landscape", (1920, 1080))] + ([("portrait", (1080, 1350))] if make_short and seg.in_short else [])
            for aspect, size in variants:
                params = dict(seg.visual.params, _portrait=aspect == "portrait")
                setup = bubble_time(params, tidy) if kind == "bubble_time" else SCENES[kind](params)
                path = ctx.assets_dir / f"{seg.id}{'_portrait' if aspect == 'portrait' else ''}.mp4"
                render(setup, path, dur, fps=fps, size=size)
                ctx.log.info("rendered %s %s (%s, %.1fs)", seg.id, aspect, kind, dur)
                out.append(Asset(id=f"anim_{seg.id}_{aspect}", segment_id=seg.id, kind="animation",
                                 path=str(path.relative_to(ctx.run.path)), licence="self-generated",
                                 origin=f"matplotlib {kind} scene from {research.data['dataset_key']} data",
                                 duration=dur, meta={"hero": beats[seg.id].get("hero", False) and aspect == "landscape",
                                                     "aspect": aspect, "source_line": params.get("source"),
                                                     "data_source": "dataset"}))
        return out

    # ------------------------------------------------------------------ format checks
    def checks(self):
        def charts_show_source(ctx) -> CheckResult:
            missing = [a.id for a in ctx.assets if a.kind == "animation" and a.segment_id != "caveats"
                       and not a.meta.get("source_line")]
            return CheckResult(check="data_story:charts_show_source", passed=not missing,
                               details=[f"chart without on-screen source: {m}" for m in missing])

        def dataset_pinned(ctx) -> CheckResult:
            sha = (ctx.research.data or {}).get("sha256") if ctx.research else None
            return CheckResult(check="data_story:dataset_snapshot_hashed", passed=bool(sha),
                               details=[f"sha256 {sha}"] if sha else ["dataset snapshot hash missing"])
        return [charts_show_source, dataset_pinned]
