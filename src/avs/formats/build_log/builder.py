"""The build agent.

For each recipe step it produces the project files, runs that step's browser
tests plus every earlier step's tests (regression), and commits to a real git
repo. With an LLM configured, the LLM writes each step from the goal + tests;
if its code fails the tests it gets one repair attempt, then the step falls
back to the recipe's reference implementation and the build log says so.
Offline, the reference implementation is replayed. Generated code is only
ever static HTML/CSS/JS served on localhost and run inside headless Chromium:
nothing an LLM writes is executed by Python.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

from .browser import run_tests, serve

ALLOWED_SUFFIXES = {".html", ".css", ".js"}
MAX_FILE_BYTES = 40_000

LLM_STEP_PROMPT = """You are building {title} step by step. Stack: {stack}. No frameworks, no CDNs,
no network requests, no external assets.

Current files:
{files}

Step {step_id}: {goal}
Why this step: {why}

These browser tests must pass (ids and texts matter):
{tests}
{repair}
Return JSON: {{"files": {{"<filename>": "<full file content>"}}, "commit_message": "<short imperative>"}}.
Only return files you create or change. Allowed file types: .html .css .js"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _read_tree(root: Path) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted(root.iterdir()) if p.is_file() and p.suffix in ALLOWED_SUFFIXES}


def _safe_files(files: dict) -> dict[str, str]:
    out = {}
    for name, content in files.items():
        if "/" in name or "\\" in name or name.startswith(".") or Path(name).suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"refusing file name {name!r}")
        if len(content.encode()) > MAX_FILE_BYTES:
            raise ValueError(f"{name} too large")
        out[name] = content
    return out


def _discard_uncommitted(project: Path) -> None:
    has_head = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=project, capture_output=True).returncode == 0
    if has_head:
        _git(project, "checkout", "--", ".")
        _git(project, "clean", "-fdq")
    else:
        for p in project.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()


def _llm_step(llm, recipe: dict, step: dict, project: Path, repair: str = "") -> tuple[dict[str, str], str]:
    files = "\n\n".join(f"--- {n} ---\n{c}" for n, c in _read_tree(project).items()) or "(none yet)"
    prompt = LLM_STEP_PROMPT.format(title=recipe["title"], stack=recipe["stack"], files=files, step_id=step["id"],
                                    goal=step["goal"], why=step["why"], tests=json.dumps(step["tests"], indent=1),
                                    repair=f"\nYour last attempt failed: {repair}\n" if repair else "")
    raw = llm.complete_json(prompt)
    return _safe_files(raw["files"]), raw.get("commit_message", step["title"])


def build_project(recipe_dir: Path, build_dir: Path, llm, log) -> dict:
    recipe = yaml.safe_load((recipe_dir / "recipe.yaml").read_text())
    project = build_dir / "project"
    snapshots = build_dir / "snapshots"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    project.mkdir(parents=True)
    snapshots.mkdir()
    _git(project, "init", "-q", "-b", "main")
    _git(project, "config", "user.email", "build-agent@ai-video-studio.local")
    _git(project, "config", "user.name", "AVS build agent")

    manifest = {"recipe": recipe["name"], "title": recipe["title"], "stack": recipe["stack"], "pitch": recipe["pitch"],
                "steps": []}
    all_tests: list[dict] = []
    for step in recipe["steps"]:
        all_tests = all_tests + step["tests"]
        mode, message, result = "recipe", f"{step['id']}: {step['title']}", None
        if llm is not None and not llm.is_offline:
            repair = ""
            for attempt in range(2):
                try:
                    files, message = _llm_step(llm, recipe, step, project, repair)
                    for n, c in files.items():
                        (project / n).write_text(c)
                    with serve(project) as url:
                        result = run_tests(url, all_tests)
                    if result["passed"] == result["total"] and not result["external_requests"]:
                        mode = "llm"
                        break
                    repair = "; ".join(f"{t['name']}: {t['error']}" for t in result["tests"] if not t["passed"])
                except Exception as exc:
                    repair = str(exc)
                log.warning("LLM attempt %d for %s failed: %s", attempt + 1, step["id"], repair[:200])
            if mode != "llm":
                _discard_uncommitted(project)
                mode = "recipe-fallback"
        if mode != "llm":
            for f in (recipe_dir / step["id"]).iterdir():
                shutil.copy(f, project / f.name)
            with serve(project) as url:
                result = run_tests(url, all_tests)
        _git(project, "add", "-A")
        _git(project, "commit", "-q", "-m", message)
        commit = _git(project, "rev-parse", "--short", "HEAD")
        numstat = _git(project, "show", "--numstat", "--format=", "HEAD")
        added = removed = 0
        changed = []
        for line in numstat.splitlines():
            a, r, name = line.split("\t")
            added, removed = added + int(a), removed + int(r)
            changed.append({"file": name, "added": int(a), "removed": int(r)})
        snap = snapshots / step["id"]
        shutil.copytree(project, snap, ignore=shutil.ignore_patterns(".git"))
        diffs = {}
        for c in changed:
            try:
                old = _git(project, "show", f"HEAD~1:{c['file']}")
            except subprocess.CalledProcessError:
                old = ""
            diffs[c["file"]] = old
        manifest["steps"].append({**{k: step[k] for k in ("id", "title", "goal", "why")}, "mode": mode, "commit": commit,
                                  "message": message, "added": added, "removed": removed, "files": changed,
                                  "previous": diffs, "tests": result})
        log.info("built %s via %s: +%d/-%d, tests %d/%d", step["id"], mode, added, removed, result["passed"], result["total"])

    final = _read_tree(project)
    manifest["final"] = {
        "files": len(final), "lines": sum(len(c.splitlines()) for c in final.values()),
        "by_type": {s: sum(len(c.splitlines()) for n, c in final.items() if n.endswith(s)) for s in (".html", ".css", ".js")},
        "external_urls": sorted({u for c in final.values() for u in re.findall(r"https?://[^\s\"')]+", c)}),
        "commits": [s["commit"] for s in manifest["steps"]],
    }
    manifest["demo"] = recipe.get("demo", [])
    (build_dir / "build_log.json").write_text(json.dumps(manifest, indent=2))
    return manifest
