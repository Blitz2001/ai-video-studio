import json

from avs.compliance.disclosure import check_disclosure
from avs.compliance.licences import check_licences
from avs.models import Asset, Metadata, VoiceTrack


def asset(**kw) -> Asset:
    base = dict(id="a1", segment_id="s1", kind="animation", path="assets/a1.mp4", licence="self-generated", origin="test")
    base.update(kw)
    return Asset(**base)


def meta(desc: str = "Gapminder Foundation", synthetic: bool = False) -> Metadata:
    return Metadata(title="t", description=desc, tags=[], category_id="28", contains_synthetic_media=synthetic)


def test_self_generated_assets_pass_and_log_is_written(make_ctx, store):
    run = store.new_run("data_story", "t")
    (run.sub("assets") / "a1.mp4").write_bytes(b"x")
    r = check_licences(make_ctx(assets=[asset()], run=run, metadata=meta()))
    assert r.passed, r.details
    log = json.loads((run.path / "licence_log.json").read_text())
    assert {row["id"] for row in log} == {"a1", "dataset"}
    assert (store.state_dir / "licence_log.csv").exists()


def test_missing_licence_fails(make_ctx):
    assert not check_licences(make_ctx(assets=[asset(licence=None)])).passed


def test_unlisted_licence_fails(make_ctx):
    r = check_licences(make_ctx(assets=[asset(licence="All rights reserved")]))
    assert not r.passed and "allow-list" in r.details[0]


def test_cc_by_needs_attribution_text(make_ctx):
    assert not check_licences(make_ctx(assets=[asset(licence="CC-BY-4.0")])).passed


def test_attribution_must_reach_the_description(make_ctx):
    a = asset(licence="CC-BY-4.0", attribution="Photo by Jane Doe")
    r = check_licences(make_ctx(assets=[a], metadata=meta("Gapminder Foundation only")))
    assert not r.passed and "Jane Doe" in " ".join(r.details)
    assert check_licences(make_ctx(assets=[a], metadata=meta("Gapminder Foundation. Photo by Jane Doe"))).passed


def test_dataset_licence_must_be_allowed(make_ctx):
    ctx = make_ctx(assets=[asset()])
    ctx.research.sources[0].licence = "proprietary"
    assert not check_licences(ctx).passed


def test_missing_file_fails(make_ctx, store):
    run = store.new_run("data_story", "t")
    r = check_licences(make_ctx(assets=[asset()], run=run))
    assert not r.passed and "file missing" in " ".join(r.details)


def test_no_assets_fails(make_ctx):
    assert not check_licences(make_ctx(assets=[])).passed


# --------------------------------------------------------------------------- disclosure
def test_realistic_voice_requires_flag(make_ctx):
    ctx = make_ctx(voice=VoiceTrack(backend="kokoro"), metadata=meta("AI-generated narration", synthetic=False))
    r = check_disclosure(ctx)
    assert not r.passed and "containsSyntheticMedia" in r.details[0]


def test_realistic_voice_requires_description_line(make_ctx):
    ctx = make_ctx(voice=VoiceTrack(backend="piper"), metadata=meta("nothing here", synthetic=True))
    assert not check_disclosure(ctx).passed


def test_disclosed_realistic_voice_passes(make_ctx):
    ctx = make_ctx(voice=VoiceTrack(backend="kokoro"), metadata=meta("AI-generated narration: Kokoro", synthetic=True))
    assert check_disclosure(ctx).passed


def test_own_voice_needs_no_flag(make_ctx):
    assert check_disclosure(make_ctx(voice=VoiceTrack(backend="recorded"), metadata=meta())).passed


def test_realistic_synthetic_person_requires_flag(make_ctx):
    ctx = make_ctx(voice=VoiceTrack(backend="recorded"), assets=[asset(realistic_person=True)], metadata=meta())
    assert not check_disclosure(ctx).passed


def test_over_disclosure_is_only_a_warning(make_ctx):
    r = check_disclosure(make_ctx(voice=VoiceTrack(backend="recorded"), metadata=meta(synthetic=True)))
    assert not r.passed and r.severity == "warn"
