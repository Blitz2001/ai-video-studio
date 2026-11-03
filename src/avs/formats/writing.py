"""Helpers for deterministic (offline) script writers: every factual sentence
is written together with its claim, so template scripts are compliant by construction."""
from __future__ import annotations

from ..models import Claim, Segment, Visual


class Beat:
    def __init__(self, beat: dict):
        self.beat = beat
        self.sentences: list[str] = []
        self.claims: list[Claim] = []

    def say(self, text: str, source: str | None = None) -> "Beat":
        text = text.strip()
        self.sentences.append(text)
        if source:
            self.claims.append(Claim(text=text, source_id=source))
        return self

    def segment(self) -> Segment:
        return Segment(
            id=self.beat["id"], narration=" ".join(self.sentences), visual=Visual(**self.beat["visual"]),
            claims=self.claims, on_screen_text=self.beat.get("on_screen_text"),
            in_short=self.beat.get("in_short", True),
        )


def fmt_num(x: float, decimals: int = 1) -> str:
    return f"{x:,.{decimals}f}"
