from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from avs.compliance.base import CheckContext
from avs.config import load_settings
from avs.models import (Asset, Claim, Fact, Metadata, RenderOutput, ResearchPack, Script, Segment, Source,
                        Visual, VoiceLine, VoiceTrack)
from avs.store import Store

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / "config" / "studio.yaml", tmp_path / "config" / "studio.yaml")
    return tmp_path


@pytest.fixture
def settings(root):
    return load_settings(root)


@pytest.fixture
def store(settings):
    return Store(settings)


def research_pack() -> ResearchPack:
    return ResearchPack(
        format="data_story", topic="test",
        sources=[
            Source(id="dataset", title="Gapminder", kind="dataset", url="https://www.gapminder.org/data/",
                   publisher="Gapminder Foundation", licence="CC-BY-4.0"),
            Source(id="analysis", title="our analysis", kind="computation", reference="analysis.py"),
            Source(id="rumour", title="a blog with no link", kind="web"),
        ],
        facts=[
            Fact(id="world", source_id="analysis", text="Average life expectancy was 48.9 years in 1952 and 68.9 years in 2007."),
            Fact(id="cov", source_id="dataset", text="The dataset covers 142 countries from 1952 to 2007."),
        ],
        data={"sha256": "0" * 64},
    )


def segment(text: str, claims: list[tuple[str, str]] = (), sid: str = "s1") -> Segment:
    return Segment(id=sid, narration=text, visual=Visual(kind="x"),
                   claims=[Claim(text=t, source_id=s) for t, s in claims])


def script(*segments: Segment, title: str = "How the world got healthier", topic: str = "gapminder") -> Script:
    return Script(format="data_story", topic=topic, title=title, thumbnail_text="x", segments=list(segments))


@pytest.fixture
def make_ctx(settings, store):
    def _make(**kw) -> CheckContext:
        kw.setdefault("research", research_pack())
        return CheckContext(settings=settings, store=store, **kw)
    return _make


@pytest.fixture
def run_with_video(store, settings):
    """A run folder that looks like it finished the pipeline: tiny fake video, thumbnail, metadata."""
    def _make(synthetic: bool = True):
        run = store.new_run("data_story", "test")
        (run.sub("out") / "video_16x9.mp4").write_bytes(b"fake-video")
        from PIL import Image
        Image.new("RGB", (64, 36), (30, 30, 30)).save(run.sub("out") / "thumbnail.png")
        (run.sub("out") / "captions.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        (run.sub("assets") / "a.mp4").write_bytes(b"x")
        seg = segment("The dataset covers 142 countries from 1952 to 2007.",
                      [("The dataset covers 142 countries from 1952 to 2007.", "dataset")])
        run.write_json("research.json", research_pack())
        run.write_json("script.json", script(seg))
        run.write_json("assets.json", [Asset(id="a", segment_id="s1", kind="animation", path="assets/a.mp4",
                                             licence="self-generated", origin="test",
                                             meta={"source_line": "Source: Gapminder"}).model_dump()])
        run.write_json("voice.json", VoiceTrack(backend="kokoro" if synthetic else "recorded",
                                                lines=[VoiceLine(segment_id="s1", path="voice/s1.wav", duration=3.0)]))
        run.write_json("render.json", RenderOutput(landscape="out/video_16x9.mp4", srt="out/captions.srt", duration=3.0))
        desc = "Sources: Gapminder Foundation. Credits." + (" AI-generated narration: synthetic voice." if synthetic else "")
        run.write_json("metadata.json", Metadata(title="t", description=desc, tags=["a"], category_id="28",
                                                 contains_synthetic_media=synthetic, thumbnail="out/thumbnail.png"))
        run.update_info(status="awaiting_review")
        return run
    return _make
