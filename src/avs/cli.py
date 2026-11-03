"""avs: one command per job, so any of them can go in cron / Task Scheduler.

    avs run --format data_story [--topic gapminder] [--llm gemini]   full pipeline -> review queue
    avs formats                                                        list format plugins
    avs resume <run_id> --step edit                                    re-run from a step
    avs runs [--status awaiting_review]                                list runs
    avs check <run_id> [--stage pre_upload]                            re-run a compliance stage
    avs dashboard                                                      open the review dashboard
    avs youtube-auth                                                   one-time OAuth consent
    avs publish [--run <id>] [--dry-run]                               upload approved runs (private)
    avs record-sheet <run_id>                                          lines to record for voice=recorded
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import load_settings


def _settings(args):
    overrides = {}
    if getattr(args, "voice", None):
        overrides.setdefault("voice", {})["backend"] = args.voice
    if getattr(args, "no_short", False):
        overrides.setdefault("render", {})["make_short"] = False
    return load_settings(overrides=overrides)


def cmd_run(args) -> int:
    from .pipeline import Pipeline

    settings = _settings(args)
    pipe = Pipeline(settings, llm_provider=args.llm)
    run = pipe.run(args.format, args.topic)
    info = run.info
    print(f"\nrun:    {run.id}\nstatus: {info['status']}\nfolder: {run.path}")
    if info.get("error"):
        print(f"error:  {info['error']}")
    return 0 if info["status"] == "awaiting_review" else 2


def cmd_resume(args) -> int:
    from .pipeline import Pipeline

    pipe = Pipeline(_settings(args), llm_provider=args.llm)
    run = pipe.resume(args.run_id, args.step)
    print(f"\nrun:    {run.id}\nstatus: {run.info['status']}")
    return 0 if run.info["status"] == "awaiting_review" else 2


def cmd_formats(args) -> int:
    from .formats import discover

    for name, plugin in sorted(discover().items()):
        print(f"{name:20s} {plugin.description}\n{'':20s} default topic: {plugin.default_topic}")
    return 0


def cmd_runs(args) -> int:
    from .store import Store

    store = Store(_settings(args))
    for run in store.list_runs({args.status} if args.status else None):
        i = run.info
        print(f"{run.id:60s} {i.get('status', '?'):16s} {i.get('title', '')}")
    return 0


def cmd_check(args) -> int:
    from .agents.publisher import load_check_context
    from .compliance import run_stage
    from .formats import get_format
    from .store import Store

    settings = _settings(args)
    store = Store(settings)
    run = store.get(args.run_id)
    report = run_stage(args.stage, load_check_context(run, settings, store), get_format(run.info["format"]).checks())
    print(report.summary())
    return 0 if report.passed else 1


def cmd_publish(args) -> int:
    from .agents.publisher import publish_run
    from .store import Store

    settings = _settings(args)
    store = Store(settings)
    runs = [store.get(args.run)] if args.run else store.list_runs({"approved"})
    if not runs:
        print("nothing approved; approve videos in the dashboard (avs dashboard)")
        return 0
    rc = 0
    for run in runs:
        report, video_id = publish_run(run, settings, store, dry_run=args.dry_run)
        print(report.summary())
        if video_id:
            print(f"uploaded {run.id} as PRIVATE video {video_id}")
        elif not report.passed:
            rc = 1
            if any(r.check == "daily_upload_cap" and not r.passed for r in report.results):
                break
    return rc


def cmd_dashboard(args) -> int:
    settings = _settings(args)
    app = settings.root / "dashboard" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"])


def cmd_youtube_auth(args) -> int:
    from .agents.publisher import youtube_auth

    print(f"token saved to {youtube_auth(_settings(args))}")
    return 0


def cmd_record_sheet(args) -> int:
    from .models import Script
    from .store import Store

    settings = _settings(args)
    run = Store(settings).get(args.run_id)
    script = run.load("script.json", Script)
    folder = settings.path(settings.get("voice.recorded_dir", "recordings")) / run.id
    print(f"Record one WAV per line into {folder}/<id>.wav, then: avs run ... --voice recorded\n")
    for seg in script.segments:
        print(f"[{seg.id}]\n{seg.narration}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "matplotlib", "PIL", "phonemizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="avs", description="AI Video Studio")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full pipeline for one format")
    r.add_argument("--format", "-f", required=True)
    r.add_argument("--topic", "-t")
    r.add_argument("--llm", choices=["gemini", "claude", "openai", "offline"])
    r.add_argument("--voice", choices=["kokoro", "piper", "recorded", "silent"])
    r.add_argument("--no-short", action="store_true")
    r.set_defaults(fn=cmd_run)

    rr = sub.add_parser("resume", help="re-run an existing run from a given step")
    rr.add_argument("run_id")
    rr.add_argument("--step", required=True)
    rr.add_argument("--llm", choices=["gemini", "claude", "openai", "offline"])
    rr.add_argument("--voice", choices=["kokoro", "piper", "recorded", "silent"])
    rr.set_defaults(fn=cmd_resume)

    sub.add_parser("formats").set_defaults(fn=cmd_formats)
    rs = sub.add_parser("runs")
    rs.add_argument("--status")
    rs.set_defaults(fn=cmd_runs)
    c = sub.add_parser("check")
    c.add_argument("run_id")
    c.add_argument("--stage", default="pre_review", choices=["script_gate", "pre_render", "pre_review", "pre_upload"])
    c.set_defaults(fn=cmd_check)
    pub = sub.add_parser("publish")
    pub.add_argument("--run")
    pub.add_argument("--dry-run", action="store_true")
    pub.set_defaults(fn=cmd_publish)
    sub.add_parser("dashboard").set_defaults(fn=cmd_dashboard)
    sub.add_parser("youtube-auth").set_defaults(fn=cmd_youtube_auth)
    rec = sub.add_parser("record-sheet")
    rec.add_argument("run_id")
    rec.set_defaults(fn=cmd_record_sheet)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
