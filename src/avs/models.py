"""Typed data passed between agents. Everything is JSON-serialisable so a run
can be inspected, resumed or reviewed from its folder alone."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- research
class Source(BaseModel):
    """Something a claim can point at. `kind` decides what counts as proof:
    - web / dataset / book: an external reference (url or citation required)
    - computation: a number we computed ourselves; `reference` names the code + inputs
    - build: an artifact of the build_log project (repo path, test log)"""

    id: str
    title: str
    kind: Literal["web", "dataset", "book", "computation", "build"] = "web"
    url: str | None = None
    publisher: str | None = None
    licence: str | None = None
    reference: str | None = None  # citation / code path / commit hash
    retrieved_at: str = Field(default_factory=utcnow)

    def is_citable(self) -> bool:
        if self.kind in ("web", "dataset"):
            return bool(self.url)
        if self.kind == "book":
            return bool(self.reference or self.url)
        return bool(self.reference)


class Fact(BaseModel):
    id: str
    text: str
    source_id: str
    value: Any = None


class ResearchPack(BaseModel):
    format: str
    topic: str
    angle: str = ""
    sources: list[Source] = []
    facts: list[Fact] = []
    data: dict[str, Any] = {}  # format-specific payload (paths, specs, parameters)

    def source(self, sid: str) -> Source | None:
        return next((s for s in self.sources if s.id == sid), None)

    def fact(self, fid: str) -> Fact | None:
        return next((f for f in self.facts if f.id == fid), None)


# --------------------------------------------------------------------------- script
class Claim(BaseModel):
    text: str  # the factual statement as it appears (or nearly) in the narration
    source_id: str


class Visual(BaseModel):
    kind: str  # format-specific: "line_race", "bubble", "sort_animation", "screen_recording" ...
    params: dict[str, Any] = {}


class Segment(BaseModel):
    id: str
    narration: str
    visual: Visual
    claims: list[Claim] = []
    on_screen_text: str | None = None
    in_short: bool = True


class Script(BaseModel):
    format: str
    topic: str
    title: str
    thumbnail_text: str
    segments: list[Segment]
    tags: list[str] = []
    summary: str = ""
    written_by: str = "offline-template"

    def full_text(self) -> str:
        return "\n".join(s.narration for s in self.segments)


# --------------------------------------------------------------------------- assets / media
class Asset(BaseModel):
    id: str
    segment_id: str | None
    kind: Literal["animation", "image", "screen_recording", "music", "voice", "font", "thumbnail"]
    path: str
    licence: str | None
    origin: str  # how it was made, e.g. "matplotlib render of gapminder snapshot"
    attribution: str | None = None
    realistic_person: bool = False  # true if it depicts a realistic synthetic human
    duration: float | None = None
    meta: dict[str, Any] = {}


class VoiceLine(BaseModel):
    segment_id: str
    path: str
    duration: float


class VoiceTrack(BaseModel):
    backend: str
    voice: str | None = None
    lines: list[VoiceLine] = []
    realistic: bool = False


class RenderOutput(BaseModel):
    landscape: str | None = None
    portrait: str | None = None
    srt: str | None = None
    duration: float = 0.0
    chapters: list[tuple[float, str]] = []


class Metadata(BaseModel):
    title: str
    description: str
    tags: list[str]
    category_id: str
    default_language: str = "en"
    privacy_status: Literal["private"] = "private"
    contains_synthetic_media: bool = False
    made_for_kids: bool = False
    thumbnail: str | None = None


# --------------------------------------------------------------------------- compliance
class CheckResult(BaseModel):
    check: str
    passed: bool
    severity: Literal["block", "warn"] = "block"
    details: list[str] = []


class ComplianceReport(BaseModel):
    stage: str
    results: list[CheckResult] = []
    created_at: str = Field(default_factory=utcnow)

    @property
    def passed(self) -> bool:
        return all(r.passed or r.severity == "warn" for r in self.results)

    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "block"]

    def summary(self) -> str:
        lines = [f"[{self.stage}] {'PASS' if self.passed else 'FAIL'}"]
        for r in self.results:
            mark = "ok " if r.passed else ("WARN" if r.severity == "warn" else "FAIL")
            lines.append(f"  {mark} {r.check}")
            if not r.passed:
                lines.extend(f"       - {d}" for d in r.details[:8])
        return "\n".join(lines)


# --------------------------------------------------------------------------- review
RunStatus = Literal[
    "running", "failed", "blocked", "awaiting_review", "approved", "rejected", "published"
]


class Review(BaseModel):
    status: Literal["pending", "approved", "rejected"] = "pending"
    reviewer: str | None = None
    approved_via: str | None = None  # must be "dashboard" for the publisher to accept it
    decided_at: str | None = None
    note: str | None = None
    content_hash: str | None = None  # hash of video+metadata at approval time
