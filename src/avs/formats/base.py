"""The format plugin contract.

A format is one folder containing `__init__.py` that exposes `PLUGIN`, an
instance of a `FormatPlugin` subclass, plus a `template.md` prompt. The engine
calls, in order:

    research(topic, ctx)               -> ResearchPack with sources, facts and a storyboard
    write_script(research, ctx)        -> Script   (LLM fills the template; offline -> fallback_script)
    generate_assets(script, research, ctx) -> list[Asset]
    checks()                           -> extra compliance checks for this format

The storyboard is the key idea: the plugin decides *what is shown* (beats with
a visual and the facts available to that beat); the LLM only decides *what is
said*, and must cite a fact's source for every factual sentence.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..models import Asset, Claim, ResearchPack, Script, Segment, Visual

if TYPE_CHECKING:
    from ..compliance.base import CheckContext
    from ..config import Settings
    from ..llm import LLM
    from ..models import CheckResult
    from ..store import Run

Check = Callable[["CheckContext"], "CheckResult"]


@dataclass
class RunContext:
    settings: "Settings"
    run: "Run"
    llm: "LLM"
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("avs"))
    feedback: list[str] = field(default_factory=list)  # compliance failures fed back to the writer

    @property
    def assets_dir(self) -> Path:
        return self.run.sub("assets")

    @property
    def brand(self) -> str:
        return self.settings.channel_name


SCRIPT_JSON_SHAPE = """{
  "title": "string, <= 90 chars, no clickbait",
  "thumbnail_text": "2-4 punchy words",
  "summary": "1-2 sentence description",
  "tags": ["5-12 lowercase tags"],
  "segments": [
    {
      "id": "beat id from the storyboard, same order",
      "narration": "spoken text for this beat (25-60 words)",
      "on_screen_text": "optional short caption heading or null",
      "claims": [{"text": "the exact factual sentence", "source_id": "id of the source backing it"}]
    }
  ]
}"""

SYSTEM_PROMPT = (
    "You write narration for short educational YouTube videos on the channel '{channel}' "
    "({niche}). Voice: clear, curious, precise, no hype. Rules you must follow:\n"
    "1. Use ONLY the facts provided. Never invent numbers, dates, names or rankings.\n"
    "2. Every sentence that states a fact (any number, date, ranking, comparison or attribution) "
    "must appear in that segment's `claims` with the source_id of the fact it came from.\n"
    "3. Do not give medical, financial or legal advice, and do not tell the viewer what to do "
    "about their health, money or legal situation.\n"
    "4. Keep the storyboard order and ids. Output JSON only."
)


class FormatPlugin(ABC):
    name: str = "base"
    description: str = ""
    default_topic: str = ""
    visual_kinds: tuple[str, ...] = ()

    # ------------------------------------------------------------------ plugin folder
    @property
    def folder(self) -> Path:
        import inspect

        return Path(inspect.getfile(type(self))).parent

    def template(self) -> str:
        p = self.folder / "template.md"
        return p.read_text() if p.exists() else ""

    # ------------------------------------------------------------------ steps
    @abstractmethod
    def research(self, topic: str, ctx: RunContext) -> ResearchPack: ...

    @abstractmethod
    def fallback_script(self, research: ResearchPack, ctx: RunContext) -> Script:
        """Deterministic, template-based script used when no LLM is configured."""

    @abstractmethod
    def generate_assets(self, script: Script, research: ResearchPack, ctx: RunContext) -> list[Asset]: ...

    def checks(self) -> list[Check]:
        return []

    # ------------------------------------------------------------------ LLM script writing
    def build_prompt(self, research: ResearchPack, ctx: RunContext) -> str:
        storyboard = research.data.get("storyboard", [])
        facts = {f.id: {"text": f.text, "source_id": f.source_id} for f in research.facts}
        sources = [{"id": s.id, "title": s.title, "kind": s.kind} for s in research.sources]
        body = self.template().format(
            topic=research.topic,
            angle=research.angle,
            channel=ctx.brand,
            storyboard=json.dumps(storyboard, indent=2),
            facts=json.dumps(facts, indent=2),
            sources=json.dumps(sources, indent=2),
        )
        prompt = f"{body}\n\nReturn JSON with exactly this shape:\n{SCRIPT_JSON_SHAPE}"
        if ctx.feedback:
            prompt += ("\n\nYour previous draft failed these compliance checks. Fix every item "
                       "(cite the fact's source_id, or remove the sentence):\n- " + "\n- ".join(ctx.feedback))
        return prompt

    def write_script(self, research: ResearchPack, ctx: RunContext) -> Script:
        if ctx.llm.is_offline:
            return self.fallback_script(research, ctx)
        system = SYSTEM_PROMPT.format(channel=ctx.brand, niche=ctx.settings.get("channel.niche", ""))
        try:
            raw = ctx.llm.complete_json(self.build_prompt(research, ctx), system=system)
            return self.script_from_llm(raw, research, ctx)
        except Exception as exc:  # malformed JSON, refusal, network ...
            ctx.log.warning("LLM script failed (%s); using template writer", exc)
            if ctx.settings.get("llm.fallback_to_offline", True):
                return self.fallback_script(research, ctx)
            raise

    def script_from_llm(self, raw: dict[str, Any], research: ResearchPack, ctx: RunContext) -> Script:
        """Merge LLM words with plugin-owned visuals. The LLM can never change a visual."""
        beats = {b["id"]: b for b in research.data.get("storyboard", [])}
        segments: list[Segment] = []
        for seg in raw.get("segments", []):
            beat = beats.get(seg.get("id"))
            if beat is None:
                continue
            segments.append(Segment(
                id=beat["id"],
                narration=seg["narration"].strip(),
                visual=Visual(**beat["visual"]),
                claims=[Claim(**c) for c in seg.get("claims", [])],
                on_screen_text=seg.get("on_screen_text") or beat.get("on_screen_text"),
                in_short=beat.get("in_short", True),
            ))
        if len(segments) != len(beats):
            raise ValueError(f"LLM returned {len(segments)} of {len(beats)} storyboard beats")
        return Script(
            format=self.name, topic=research.topic, title=raw["title"][:95],
            thumbnail_text=raw.get("thumbnail_text", research.topic)[:40],
            summary=raw.get("summary", ""), tags=[t.lower() for t in raw.get("tags", [])][:15],
            segments=segments, written_by=ctx.llm.name,
        )
