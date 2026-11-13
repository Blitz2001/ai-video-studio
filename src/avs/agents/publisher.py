"""Publisher: YouTube Data API v3. Uploads as PRIVATE, only after the
pre_upload compliance stage passes (which includes the dashboard approval and
the daily cap). You flip videos public yourself in YouTube Studio."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..compliance import CheckContext, run_stage
from ..config import Settings
from ..formats import get_format
from ..models import Asset, ComplianceReport, Metadata, RenderOutput, ResearchPack, Script, VoiceTrack
from ..store import Run, Store

log = logging.getLogger("avs.publisher")

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # thumbnails.set + captions.insert
]


def youtube_client(settings: Settings):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    token = settings.path(os.environ.get("YOUTUBE_TOKEN_FILE", "secrets/youtube_token.json"))
    if not token.exists():
        raise RuntimeError(f"no YouTube token at {token}; run `avs youtube-auth` first")
    creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def youtube_auth(settings: Settings) -> Path:
    """One-time OAuth consent in the browser; stores a refresh token locally."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = settings.path(os.environ.get("YOUTUBE_CLIENT_SECRETS", "secrets/client_secret.json"))
    token = settings.path(os.environ.get("YOUTUBE_TOKEN_FILE", "secrets/youtube_token.json"))
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    token.parent.mkdir(parents=True, exist_ok=True)
    token.write_text(creds.to_json())
    return token


def _media(path: Path, mimetype: str, resumable: bool = True):
    from googleapiclient.http import MediaFileUpload

    return MediaFileUpload(str(path), mimetype=mimetype, chunksize=8 * 1024 * 1024, resumable=resumable)


def load_check_context(run: Run, settings: Settings, store: Store, llm=None, now=None) -> CheckContext:
    assets = run.load_list("assets.json", Asset)
    return CheckContext(
        settings=settings, research=run.load("research.json", ResearchPack), script=run.load("script.json", Script),
        assets=assets, voice=run.load("voice.json", VoiceTrack), render=run.load("render.json", RenderOutput),
        metadata=run.load("metadata.json", Metadata), run=run, store=store, llm=llm, now=now,
    )


def upload_body(meta: Metadata) -> dict[str, Any]:
    if meta.privacy_status != "private":  # belt and braces; the model type already enforces it
        raise ValueError("refusing to upload anything that is not private")
    return {
        "snippet": {
            "title": meta.title, "description": meta.description, "tags": meta.tags,
            "categoryId": meta.category_id, "defaultLanguage": meta.default_language,
            "defaultAudioLanguage": meta.default_language,
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": meta.made_for_kids,
            "containsSyntheticMedia": meta.contains_synthetic_media,
            "embeddable": True,
        },
    }


def publish_run(run: Run, settings: Settings, store: Store, youtube=None, dry_run: bool = False,
                media_factory: Callable = _media, now=None) -> tuple[ComplianceReport, str | None]:
    ctx = load_check_context(run, settings, store, now=now)
    extra = get_format(run.info["format"]).checks()
    report = run_stage("pre_upload", ctx, extra)
    if not report.passed:
        log.warning("pre_upload compliance failed for %s\n%s", run.id, report.summary())
        return report, None
    if dry_run:
        return report, None

    youtube = youtube or youtube_client(settings)
    meta = ctx.metadata
    video_path = run.path / ctx.render.landscape
    request = youtube.videos().insert(part="snippet,status", body=upload_body(meta),
                                      media_body=media_factory(video_path, "video/mp4"))
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status is not None:
            log.info("upload %s: %d%%", run.id, int(status.progress() * 100))
    video_id = response["id"]

    if meta.thumbnail:
        youtube.thumbnails().set(videoId=video_id, media_body=media_factory(run.path / meta.thumbnail, "image/png", False)).execute()
    if settings.get("publish.upload_captions", True) and ctx.render.srt:
        youtube.captions().insert(
            part="snippet",
            body={"snippet": {"videoId": video_id, "language": meta.default_language, "name": "English", "isDraft": False}},
            media_body=media_factory(run.path / ctx.render.srt, "application/octet-stream", False),
        ).execute()

    when = (now or datetime.now(timezone.utc)).isoformat()
    store.log_upload({"run_id": run.id, "video_id": video_id, "uploaded_at": when, "status": "uploaded",
                      "privacy": "private", "synthetic": meta.contains_synthetic_media})
    store.add_history(run.id, meta.title, ctx.script.full_text())
    run.update_info(status="published", youtube_id=video_id, published_at=when,
                    youtube_url=f"https://studio.youtube.com/video/{video_id}/edit")
    return report, video_id
