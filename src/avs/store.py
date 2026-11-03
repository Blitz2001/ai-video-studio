"""File-backed persistence. A run is one folder under runs/<run_id>/:

    run.json            status, format, topic, timestamps, errors
    research.json       ResearchPack
    script.json         Script
    assets.json         list[Asset]
    voice.json          VoiceTrack
    render.json         RenderOutput
    metadata.json       Metadata
    compliance/*.json   one ComplianceReport per stage
    licence_log.json    every asset + licence + attribution
    review.json         Review (written by the dashboard only)
    assets/ voice/ out/ working files

Global state lives in data/state/: uploads.jsonl (publisher log) and
history.jsonl (past videos for the similarity check)."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import Settings
from .models import Review, utcnow

M = TypeVar("M", bound=BaseModel)


def slugify(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-") or "run"


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def id(self) -> str:
        return self.path.name

    def sub(self, name: str) -> Path:
        p = self.path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- json helpers
    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, BaseModel):
            text = obj.model_dump_json(indent=2)
        elif isinstance(obj, list) and obj and isinstance(obj[0], BaseModel):
            text = json.dumps([o.model_dump() for o in obj], indent=2)
        else:
            text = json.dumps(obj, indent=2, default=str)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(p)
        return p

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.path / name
        if not p.exists():
            return default
        return json.loads(p.read_text())

    def load(self, name: str, model: type[M]) -> M | None:
        data = self.read_json(name)
        return model.model_validate(data) if data is not None else None

    def load_list(self, name: str, model: type[M]) -> list[M]:
        return [model.model_validate(d) for d in (self.read_json(name) or [])]

    # ---- run.json
    @property
    def info(self) -> dict:
        return self.read_json("run.json", {})

    def update_info(self, **kw: Any) -> dict:
        info = self.info
        info.update(kw)
        info["updated_at"] = utcnow()
        self.write_json("run.json", info)
        return info

    @property
    def status(self) -> str:
        return self.info.get("status", "unknown")

    def review(self) -> Review:
        return self.load("review.json", Review) or Review()

    def content_hash(self) -> str:
        """Hash of everything a reviewer looked at: rendered video, thumbnail and
        metadata. Approval is bound to this hash, so editing after approval voids it."""
        h = hashlib.sha256()
        render = self.read_json("render.json", {}) or {}
        meta = self.read_json("metadata.json", {}) or {}
        for rel in [render.get("landscape"), render.get("portrait"), meta.get("thumbnail")]:
            if rel and (self.path / rel).exists():
                with open(self.path / rel, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
        h.update(json.dumps(meta, sort_keys=True).encode())
        return h.hexdigest()


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.runs_dir = settings.runs_dir
        self.state_dir = settings.state_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def new_run(self, fmt: str, topic: str) -> Run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{fmt}-{slugify(topic, 30)}"
        path, n = self.runs_dir / base, 1
        while path.exists():  # two runs in the same second (scheduler, tests)
            n += 1
            path = self.runs_dir / f"{base}-{n}"
        run = Run(path)
        run.path.mkdir(parents=True, exist_ok=False)
        run.write_json("run.json", {
            "id": run.id, "format": fmt, "topic": topic, "status": "running",
            "created_at": utcnow(), "updated_at": utcnow(), "steps": [],
        })
        return run

    def get(self, run_id: str) -> Run:
        p = self.runs_dir / run_id
        if not p.exists():
            raise FileNotFoundError(f"no run {run_id} in {self.runs_dir}")
        return Run(p)

    def list_runs(self, statuses: set[str] | None = None) -> list[Run]:
        runs = [Run(p) for p in sorted(self.runs_dir.iterdir(), reverse=True)
                if p.is_dir() and (p / "run.json").exists()]
        if statuses:
            runs = [r for r in runs if r.status in statuses]
        return runs

    # ---- uploads log (daily cap)
    @property
    def uploads_file(self) -> Path:
        return self.state_dir / "uploads.jsonl"

    def log_upload(self, record: dict) -> None:
        with open(self.uploads_file, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def uploads(self) -> list[dict]:
        if not self.uploads_file.exists():
            return []
        return [json.loads(line) for line in self.uploads_file.read_text().splitlines() if line.strip()]

    # ---- history of past videos (similarity check)
    @property
    def history_file(self) -> Path:
        return self.state_dir / "history.jsonl"

    def add_history(self, run_id: str, title: str, text: str) -> None:
        with open(self.history_file, "a") as fh:
            fh.write(json.dumps({"run_id": run_id, "title": title, "text": text, "at": utcnow()}) + "\n")

    def past_scripts(self, exclude_run: str | None = None) -> list[dict]:
        """Past videos = imported history + every run that got past the script gate
        and was not rejected (pending review, approved, published)."""
        items: list[dict] = []
        if self.history_file.exists():
            items += [json.loads(l) for l in self.history_file.read_text().splitlines() if l.strip()]
        seen = {i["run_id"] for i in items}
        for run in self.list_runs({"awaiting_review", "approved", "published"}):
            if run.id == exclude_run or run.id in seen:
                continue
            script = run.read_json("script.json")
            if script:
                text = "\n".join(s["narration"] for s in script["segments"])
                items.append({"run_id": run.id, "title": script["title"], "text": text})
        return [i for i in items if i["run_id"] != exclude_run]
