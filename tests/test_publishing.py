from datetime import datetime, timedelta, timezone

import pytest

from avs.agents.publisher import load_check_context, publish_run, upload_body
from avs.compliance.publishing import check_approval, check_daily_cap
from avs.models import Metadata, Review, utcnow


def approve(run, via="dashboard"):
    run.write_json("review.json", Review(status="approved", reviewer="me", approved_via=via, decided_at=utcnow(),
                                         content_hash=run.content_hash()))
    run.update_info(status="approved")


class FakeRequest:
    def __init__(self, response):
        self.response = response

    def next_chunk(self):
        return None, self.response

    def execute(self):
        return self.response


class FakeYouTube:
    def __init__(self):
        self.inserted, self.thumbs, self.caption_tracks = [], [], []

    def videos(self):
        yt = self

        class V:
            def insert(self, part, body, media_body):
                yt.inserted.append(body)
                return FakeRequest({"id": f"vid{len(yt.inserted)}"})
        return V()

    def thumbnails(self):
        yt = self

        class T:
            def set(self, videoId, media_body):
                yt.thumbs.append(videoId)
                return FakeRequest({})
        return T()

    def captions(self):
        yt = self

        class C:
            def insert(self, part, body, media_body):
                yt.caption_tracks.append(body)
                return FakeRequest({})
        return C()


def media(path, mimetype, resumable=True):
    return {"path": str(path), "mimetype": mimetype}


SLST_NOON = datetime(2026, 11, 3, 6, 30, tzinfo=timezone.utc)  # 12:00 in Colombo


# --------------------------------------------------------------------------- approval gate
def test_unapproved_run_cannot_publish(run_with_video, settings, store):
    run = run_with_video()
    yt = FakeYouTube()
    report, vid = publish_run(run, settings, store, youtube=yt, media_factory=media, now=SLST_NOON)
    assert vid is None and not yt.inserted
    assert any(r.check == "human_approval" and not r.passed for r in report.results)


def test_cli_approval_does_not_count(run_with_video, settings, store):
    run = run_with_video()
    approve(run, via="cli")
    r = check_approval(load_check_context(run, settings, store))
    assert not r.passed and "dashboard" in r.details[0]


def test_changing_content_after_approval_voids_it(run_with_video, settings, store):
    run = run_with_video()
    approve(run)
    meta = run.load("metadata.json", Metadata)
    meta.title = "a sneaky new title"
    run.write_json("metadata.json", meta)
    r = check_approval(load_check_context(run, settings, store))
    assert not r.passed and "changed after approval" in r.details[0]


def test_rejected_run_cannot_publish(run_with_video, settings, store):
    run = run_with_video()
    run.write_json("review.json", Review(status="rejected", approved_via="dashboard"))
    assert not check_approval(load_check_context(run, settings, store)).passed


# --------------------------------------------------------------------------- upload
def test_approved_run_uploads_private_with_disclosure(run_with_video, settings, store):
    run = run_with_video(synthetic=True)
    approve(run)
    yt = FakeYouTube()
    report, vid = publish_run(run, settings, store, youtube=yt, media_factory=media, now=SLST_NOON)
    assert report.passed, report.summary()
    assert vid == "vid1"
    body = yt.inserted[0]
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["containsSyntheticMedia"] is True
    assert body["status"]["selfDeclaredMadeForKids"] is False
    assert yt.thumbs == ["vid1"] and yt.caption_tracks
    assert run.info["status"] == "published"
    assert store.uploads()[0]["video_id"] == "vid1"


def test_dry_run_uploads_nothing(run_with_video, settings, store):
    run = run_with_video()
    approve(run)
    yt = FakeYouTube()
    report, vid = publish_run(run, settings, store, youtube=yt, dry_run=True, media_factory=media, now=SLST_NOON)
    assert report.passed and vid is None and not yt.inserted


def test_upload_body_refuses_public():
    meta = Metadata(title="t", description="d", tags=[], category_id="28")
    object.__setattr__(meta, "privacy_status", "public")  # bypass validation the way a bug might
    with pytest.raises(ValueError):
        upload_body(meta)


# --------------------------------------------------------------------------- daily cap
def test_daily_cap_blocks_second_upload_same_day(run_with_video, settings, store):
    first, second = run_with_video(), run_with_video()
    approve(first)
    approve(second)
    yt = FakeYouTube()
    assert publish_run(first, settings, store, youtube=yt, media_factory=media, now=SLST_NOON)[1] == "vid1"
    report, vid = publish_run(second, settings, store, youtube=yt, media_factory=media, now=SLST_NOON + timedelta(hours=1))
    assert vid is None and len(yt.inserted) == 1
    assert any(r.check == "daily_upload_cap" and not r.passed for r in report.results)


def test_cap_resets_at_midnight_sri_lanka_time(make_ctx, store):
    # 23:00 SLST on 3 Nov = 17:30 UTC; 00:30 SLST on 4 Nov = 19:00 UTC (still 3 Nov in UTC)
    store.log_upload({"run_id": "r1", "video_id": "v", "status": "uploaded",
                      "uploaded_at": datetime(2026, 11, 3, 17, 30, tzinfo=timezone.utc).isoformat()})
    same_day = make_ctx(now=datetime(2026, 11, 3, 18, 0, tzinfo=timezone.utc))   # 23:30 SLST
    next_day = make_ctx(now=datetime(2026, 11, 3, 19, 0, tzinfo=timezone.utc))   # 00:30 SLST next day
    assert not check_daily_cap(same_day).passed
    assert check_daily_cap(next_day).passed


def test_cap_is_configurable(make_ctx, store, settings):
    settings.raw["publish"]["daily_upload_cap"] = 2
    store.log_upload({"run_id": "r1", "video_id": "v", "status": "uploaded", "uploaded_at": SLST_NOON.isoformat()})
    assert check_daily_cap(make_ctx(now=SLST_NOON)).passed
