from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from ...media.anim import bullet_card, estimate_seconds, render
from ...models import Asset, CheckResult, Fact, ResearchPack, Script, Source
from ..base import FormatPlugin, RunContext
from ..writing import Beat
from .builder import build_project
from .studio import record, studio_page

SECRET_PATTERNS = {
    "Google API key": r"AIza[0-9A-Za-z_\-]{35}",
    "OpenAI key": r"sk-[A-Za-z0-9_\-]{20,}",
    "Anthropic key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "AWS key": r"AKIA[0-9A-Z]{16}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}


def scan_for_secrets(paths: list[Path]) -> list[str]:
    hits = []
    for p in paths:
        try:
            text = p.read_text(errors="ignore")
        except (IsADirectoryError, FileNotFoundError):
            continue
        for name, pat in SECRET_PATTERNS.items():
            if re.search(pat, text):
                hits.append(f"{name} pattern in {p.name}")
    return hits


class BuildLog(FormatPlugin):
    name = "build_log"
    description = "An agent builds a small real project; Playwright records the screen; narration explains what and why."
    default_topic = "focus_timer"
    visual_kinds = ("screen_recording", "app_demo", "bullet_card")

    def _recipe_dir(self, topic: str) -> Path:
        key = topic.lower().strip().replace(" ", "_").replace("-", "_")
        recipes = self.folder / "recipes"
        if (recipes / key).exists():
            return recipes / key
        for d in recipes.iterdir():
            rec = yaml.safe_load((d / "recipe.yaml").read_text())
            if any(w in rec["title"].lower() for w in topic.lower().split() if len(w) > 3):
                return d
        raise ValueError(f"no recipe for {topic!r}; add a folder under {recipes}")

    # ------------------------------------------------------------------ research = do the build
    def research(self, topic: str, ctx: RunContext) -> ResearchPack:
        recipe_dir = self._recipe_dir(topic)
        build_dir = ctx.run.path / "build"
        m = build_project(recipe_dir, build_dir, ctx.llm, ctx.log)
        fin = m["final"]
        last_tests = m["steps"][-1]["tests"]
        sources = [
            Source(id="build_repo", title=f"Build repository for {m['title']}", kind="build",
                   reference=f"git commits {', '.join(fin['commits'])} (build/project in the run folder)"),
            Source(id="build_tests", title="Playwright browser test log", kind="build",
                   reference="build/build_log.json (tests run headless in Chromium after every step)"),
            Source(id="recipe", title=f"Project brief: {m['title']}", kind="build",
                   reference=f"avs/formats/build_log/recipes/{m['recipe']}/recipe.yaml"),
        ]
        facts = [
            Fact(id="final_size", source_id="build_repo",
                 text=f"The finished app is {fin['lines']} lines in {fin['files']} files across {len(m['steps'])} commits: "
                      f"{fin['by_type']['.html']} HTML, {fin['by_type']['.css']} CSS, {fin['by_type']['.js']} JavaScript."),
            Fact(id="tests", source_id="build_tests",
                 text=f"{last_tests['passed']} of {last_tests['total']} browser tests passed on the final build, "
                      f"with {len(last_tests['external_requests'])} external network requests."),
            Fact(id="discipline", source_id="build_tests",
                 text="Every step ran the full browser test suite before it was committed."),
        ]
        storyboard = [{"id": "demo", "on_screen_text": "What we built", "in_short": True, "facts": ["final_size"],
                       "visual": {"kind": "app_demo", "params": {"step": m["steps"][-1]["id"]}}}]
        for i, s in enumerate(m["steps"], 1):
            facts += [
                Fact(id=f"{s['id']}_goal", source_id="recipe", text=s["goal"]),
                Fact(id=f"{s['id']}_why", source_id="recipe", text=s["why"]),
                Fact(id=f"{s['id']}_diff", source_id="build_repo",
                     text=f"Step {i} (commit {s['commit']}) changed {len(s['files'])} files: +{s['added']} / -{s['removed']} lines; "
                          f"{s['tests']['passed']} of {s['tests']['total']} tests passed; written by {s['mode']}."),
            ]
            storyboard.append({"id": s["id"], "on_screen_text": f"Step {i}: {s['title']}", "in_short": i == len(m["steps"]),
                               "facts": [f"{s['id']}_goal", f"{s['id']}_why", f"{s['id']}_diff"],
                               "visual": {"kind": "screen_recording", "params": {"step": s["id"], "index": i}}})
        storyboard.append({"id": "wrap", "on_screen_text": "Build stats", "in_short": False, "facts": ["tests", "final_size"],
                           "visual": {"kind": "bullet_card", "params": {"title": "Build log", "lines": [
                               f"{len(m['steps'])} commits, {fin['lines']} lines, no dependencies",
                               f"{last_tests['passed']}/{last_tests['total']} browser tests passing",
                               f"{len(last_tests['external_requests'])} external network requests",
                               "Every step: code → tests → commit"]}}})
        return ResearchPack(format=self.name, topic=m["recipe"], angle=m["pitch"], sources=sources, facts=facts,
                            data={"manifest_path": "build/build_log.json", "storyboard": storyboard})

    def _manifest(self, ctx: RunContext, research: ResearchPack) -> dict:
        return json.loads((ctx.run.path / research.data["manifest_path"]).read_text())

    # ------------------------------------------------------------------ offline writer
    def fallback_script(self, research: ResearchPack, ctx: RunContext) -> Script:
        m = self._manifest(ctx, research)
        fin, last = m["final"], m["steps"][-1]["tests"]
        n = len(m["steps"])
        segs = []
        for beat in research.data["storyboard"]:
            b = Beat(beat)
            if beat["id"] == "demo":
                b.say(f"This is {m['title']}, built from an empty folder by a build agent, while I recorded the screen.")
                b.say(f"The finished app is {fin['lines']} lines in {fin['files']} files across {n} commits.", "build_repo")
                b.say("Here's how it came together, step by step.")
            elif beat["id"] == "wrap":
                b.say(f"{last['passed']} of {last['total']} browser tests passed on the final build, "
                      f"with {len(last['external_requests'])} external network requests.", "build_tests")
                b.say("Every step ran the full browser test suite before it was committed.", "build_tests")
                b.say(f"I'm {ctx.brand}. See you in the next build.")
            else:
                i = [s["id"] for s in m["steps"]].index(beat["id"])
                s = m["steps"][i]
                lead = "First" if i == 0 else ("Finally" if i == n - 1 else "Next")
                b.say(f"{lead}: {s['title'].lower()}.")
                b.say("The goal: " + s["goal"][0].lower() + s["goal"][1:].rstrip(".") + ".", "recipe")
                b.say(s["why"].rstrip(".") + ".", "recipe")
                b.say(f"This commit changed {len(s['files'])} files, adding {s['added']} lines and removing {s['removed']}.",
                      "build_repo")
            segs.append(b.segment())
        return Script(format=self.name, topic=research.topic, title=f"Build Log: {m['title'][0].upper() + m['title'][1:]}, Step by Step",
                      thumbnail_text="Built from zero", summary=f"{m['pitch']} Built step by step with a test gate on "
                      f"every commit, recorded with Playwright.",
                      tags=["build log", "javascript", "web development", "playwright", "ai agent", "coding"],
                      segments=segs)

    # ------------------------------------------------------------------ assets = screen recordings
    def generate_assets(self, script: Script, research: ResearchPack, ctx: RunContext) -> list[Asset]:
        m = self._manifest(ctx, research)
        steps = {s["id"]: s for s in m["steps"]}
        order = [s["id"] for s in m["steps"]]
        studio_dir = ctx.assets_dir / "studio"
        studio_dir.mkdir(exist_ok=True)
        fps = int(ctx.settings.get("render.fps", 30))
        make_short = ctx.settings.get("render.make_short", True)
        out = []
        for seg in script.segments:
            kind, params = seg.visual.kind, seg.visual.params
            dur = estimate_seconds(seg.narration)
            variants = [("landscape", (1920, 1080))] + ([("portrait", (1080, 1350))] if make_short and seg.in_short else [])
            for aspect, size in variants:
                suffix = "_portrait" if aspect == "portrait" else ""
                path = ctx.assets_dir / f"{seg.id}{suffix}.mp4"
                if kind == "bullet_card":
                    render(bullet_card(params), path, dur, fps=fps, size=size)
                    origin = "matplotlib summary card"
                else:
                    step_id = params["step"]
                    snap = f"build/snapshots/{step_id}"
                    if kind == "app_demo":
                        page = studio_page(None, None, None, ctx.brand, "demo", len(order), 0)
                        actions, tests, typing = m["demo"], None, False
                    else:
                        s = steps[step_id]
                        i = order.index(step_id)
                        prev = f"/build/snapshots/{order[i - 1]}/index.html" if i > 0 else None
                        page = studio_page(s, ctx.run.path / snap, prev, ctx.brand, "build", len(order), i + 1)
                        actions = self._step_actions(s, research)
                        tests, typing = s["tests"]["tests"], True
                    page_path = studio_dir / f"{seg.id}{suffix}.html"
                    page_path.write_text(page)
                    record(ctx.run.path, str(page_path.relative_to(ctx.run.path)), path, size, dur,
                           f"{snap}/index.html", actions, tests, typing)
                    origin = "Playwright screen recording of the locally built app"
                ctx.log.info("recorded %s %s (%.1fs)", seg.id, aspect, dur)
                out.append(Asset(id=f"rec_{seg.id}_{aspect}", segment_id=seg.id,
                                 kind="animation" if kind == "bullet_card" else "screen_recording",
                                 path=str(path.relative_to(ctx.run.path)), licence="self-generated", origin=origin,
                                 duration=dur, meta={"aspect": aspect, "hero": seg.id == "demo" and aspect == "landscape",
                                                     "hero_at": 0.6}))
        return out

    def _step_actions(self, step: dict, research: ResearchPack) -> list[dict]:
        """Interactions shown during a step recording: replay that step's own test actions (minus navigation)."""
        recipe = yaml.safe_load((self._recipe_dir(research.topic) / "recipe.yaml").read_text())
        spec = next(s for s in recipe["steps"] if s["id"] == step["id"])
        actions = []
        for t in spec["tests"]:
            for a in t["do"]:
                verb = next(iter(a))
                if verb in ("click", "press") or (verb == "wait" and actions):
                    actions.append(a)
                elif verb == "goto" and "speed=" in a["goto"] and not actions:
                    actions.append({"goto": a["goto"].replace("speed=3000", "speed=600")})
        return actions[:6]

    # ------------------------------------------------------------------ format checks
    def checks(self):
        def tests_green(ctx) -> CheckResult:
            m = json.loads((ctx.run.path / ctx.research.data["manifest_path"]).read_text())
            bad = [f"{s['id']}: {t['name']} ({t['error']})" for s in m["steps"] for t in s["tests"]["tests"] if not t["passed"]]
            return CheckResult(check="build_log:tests_pass", passed=not bad, details=bad)

        def clean_console(ctx) -> CheckResult:
            m = json.loads((ctx.run.path / ctx.research.data["manifest_path"]).read_text())
            errs = sorted({e for s in m["steps"] for e in s["tests"]["console_errors"]})
            ext = sorted({u for s in m["steps"] for u in s["tests"]["external_requests"]} | set(m["final"]["external_urls"]))
            return CheckResult(check="build_log:no_console_errors_or_external_requests", passed=not errs and not ext,
                               details=[f"console: {e}" for e in errs] + [f"external: {u}" for u in ext])

        def no_secrets(ctx) -> CheckResult:
            root = ctx.run.path
            files = [p for p in (root / "build").rglob("*") if p.is_file() and ".git" not in p.parts]
            files += list((root / "assets" / "studio").glob("*.html"))
            hits = scan_for_secrets(files)
            return CheckResult(check="build_log:no_secrets_on_screen", passed=not hits, details=hits)
        return [tests_green, clean_console, no_secrets]
