from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..models import Asset, Metadata, RenderOutput, ResearchPack, Script, VoiceTrack

if TYPE_CHECKING:
    from ..config import Settings
    from ..llm import LLM
    from ..store import Run, Store


@dataclass
class CheckContext:
    settings: "Settings"
    research: ResearchPack | None = None
    script: Script | None = None
    assets: list[Asset] = field(default_factory=list)
    voice: VoiceTrack | None = None
    render: RenderOutput | None = None
    metadata: Metadata | None = None
    run: "Run | None" = None
    store: "Store | None" = None
    llm: "LLM | None" = None
    now: Any = None  # datetime override for tests

    def cfg(self, dotted: str, default: Any = None) -> Any:
        return self.settings.get(dotted, default)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
