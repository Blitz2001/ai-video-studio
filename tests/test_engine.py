"""Engine-level tests: stage runner, plugin discovery, and an end-to-end
pipeline run with a throwaway plugin (silent voice, no LLM, real ffmpeg)."""
import shutil
import textwrap

import pytest

from avs.compliance import STAGES, run_stage
from avs.formats import discover
from avs.models import CheckResult

from .conftest import script, segment


def test_crashing_check_counts_as_failure(make_ctx):
    def boom(ctx):
        raise RuntimeError("bug")
    report = run_stage("pre_render", make_ctx(script=script(segment("Hello there."))), [boom])
    assert not report.passed
    assert any("crashed" in d for r in report.results for d in r.details)


def test_upload_stage_includes_every_gate():
    names = {fn.__name__ for fn in STAGES["pre_upload"]}
    assert {"check_claims", "check_topics", "check_licences", "check_disclosure", "check_private",
            "check_approval", "check_daily_cap"} <= names
    assert "check_similarity" in {fn.__name__ for fn in STAGES["script_gate"]}


def test_format_checks_run_only_after_script_gate(make_ctx):
    called = []

    def extra(ctx):
        called.append(1)
        return CheckResult(check="extra", passed=True)
    run_stage("script_gate", make_ctx(script=script(segment("Hi."))), [extra])
    assert not called
    run_stage("pre_render", make_ctx(script=script(segment("Hi."))), [extra])
    assert called


def test_builtin_formats_are_discovered():
    assert {"data_story", "concept_explainer", "build_log"} <= set(discover())


PLUGIN_SRC = textwrap.dedent('''
    from avs.formats.base import FormatPlugin
    from avs.formats.writing import Beat
    from avs.media.anim import bullet_card, render
    from avs.models import Asset, CheckResult, Fact, ResearchPack, Script, Source


    class Hello(FormatPlugin):
        name = "hello_format"
        description = "test plugin"
        default_topic = "hello"

        def research(self, topic, ctx):
            return ResearchPack(format=self.name, topic=topic,
                sources=[Source(id="calc", title="arithmetic", kind="computation", reference="2+2")],
                facts=[Fact(id="f", source_id="calc", text="Two plus two is 4.")],
                data={"storyboard": [{"id": "only", "visual": {"kind": "card", "params": {}}}]})

        def fallback_script(self, research, ctx):
            b = Beat(research.data["storyboard"][0])
            b.say("Two plus two is 4.", "calc").say(SAY_EXTRA)
            return Script(format=self.name, topic=research.topic, title="Hello", thumbnail_text="Hi",
                          segments=[b.segment()])

        def generate_assets(self, script, research, ctx):
            p = ctx.assets_dir / "only.mp4"
            render(bullet_card({"title": "Hello", "lines": ["one line"]}), p, 1.5, fps=10, size=(640, 360))
            return [Asset(id="a", segment_id="only", kind="animation", path="assets/only.mp4",
                          licence="self-generated", origin="test")]

        def checks(self):
            return [lambda ctx: CheckResult(check="hello:ok", passed=True)]


    PLUGIN = Hello()
''')


@pytest.fixture
def hello_plugin(tmp_path, monkeypatch):
    def make(extra_sentence: str):
        folder = tmp_path / "plugins" / "hello_format"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text(PLUGIN_SRC.replace("SAY_EXTRA", repr(extra_sentence)))
        monkeypatch.setenv("AVS_EXTRA_FORMATS", str(tmp_path / "plugins"))
        return folder
    return make


def test_adding_a_folder_adds_a_format(hello_plugin):
    hello_plugin("Bye.")
    assert "hello_format" in discover()


def _pipeline(settings):
    from avs.pipeline import Pipeline

    settings.raw["voice"]["backend"] = "silent"
    settings.raw["render"].update({"make_short": False, "landscape": [640, 360], "fps": 10})
    return Pipeline(settings, llm_provider="offline")


def test_pipeline_end_to_end_reaches_review_queue(hello_plugin, settings):
    hello_plugin("That is all for today.")
    run = _pipeline(settings).run("hello_format")
    info = run.info
    assert info["status"] == "awaiting_review", info.get("error")
    assert (run.path / "out" / "video_16x9.mp4").stat().st_size > 1000
    assert (run.path / "out" / "thumbnail.png").exists()
    assert (run.path / "licence_log.json").exists()
    meta = run.read_json("metadata.json")
    assert meta["privacy_status"] == "private" and meta["contains_synthetic_media"] is False
    assert run.review().status == "pending"  # nothing is approved automatically


def test_pipeline_stops_at_script_gate_on_uncited_claim(hello_plugin, settings):
    hello_plugin("It is the fastest way to learn maths.")  # uncited superlative
    run = _pipeline(settings).run("hello_format")
    assert run.info["status"] == "blocked"
    assert run.info["blocked_at"] == "script_gate"
    assert not (run.path / "assets").exists() or not any((run.path / "assets").glob("*.mp4"))


def test_pipeline_blocks_health_advice(hello_plugin, settings):
    hello_plugin("You should take 500mg of ibuprofen every day.")
    run = _pipeline(settings).run("hello_format")
    assert run.info["status"] == "blocked"
    assert "no_health_finance_legal_advice" in run.info["error"]


class ScriptedLLM:
    """Stands in for Gemini: first draft has an uncited number, second draft (after
    compliance feedback is appended to the prompt) cites it."""
    name = "fake-llm"
    is_offline = False

    def __init__(self):
        self.prompts = []

    def complete_json(self, prompt, system=None):
        if "content-policy reviewer" in prompt:      # the topic check's second opinion
            return {"advice": False, "domain": "none", "evidence": ""}
        self.prompts.append(prompt)
        cite = "failed these compliance checks" in prompt
        seg = {"id": "only", "narration": "Two plus two is 4. Nice and simple.",
               "claims": [{"text": "Two plus two is 4.", "source_id": "calc"}] if cite else []}
        return {"title": "Hello", "thumbnail_text": "Hi", "summary": "s", "tags": ["a"], "segments": [seg]}

    def embed(self, texts):
        raise RuntimeError("no embeddings in tests")


def test_llm_rewrites_after_compliance_feedback(hello_plugin, settings):
    from avs.pipeline import Pipeline

    hello_plugin("unused")
    settings.raw["voice"]["backend"] = "silent"
    settings.raw["render"].update({"make_short": False, "landscape": [640, 360], "fps": 10})
    llm = ScriptedLLM()
    run = Pipeline(settings, llm=llm).run("hello_format")
    assert len(llm.prompts) == 2, "the gate should have sent the script back exactly once"
    assert "uncited factual sentence" in llm.prompts[1]
    assert run.info["status"] == "awaiting_review", run.info.get("error")
    assert run.read_json("script.json")["written_by"] == "fake-llm"
