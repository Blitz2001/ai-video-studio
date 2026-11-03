"""Upload gates: human approval in the dashboard, private-only, daily cap."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from ..models import CheckResult
from .base import CheckContext


def check_approval(ctx: CheckContext) -> CheckResult:
    if ctx.run is None:
        return CheckResult(check="human_approval", passed=False, details=["no run"])
    review = ctx.run.review()
    details = []
    if review.status != "approved":
        details.append(f"review status is {review.status!r}; approve it in the dashboard first")
    elif review.approved_via != "dashboard":
        details.append(f"approval came from {review.approved_via!r}; only dashboard approvals count")
    elif review.content_hash != ctx.run.content_hash():
        details.append("video, thumbnail or metadata changed after approval; re-approve in the dashboard")
    return CheckResult(check="human_approval", passed=not details,
                       details=details or [f"approved by {review.reviewer} at {review.decided_at}"])


def check_private(ctx: CheckContext) -> CheckResult:
    ok = ctx.metadata is not None and ctx.metadata.privacy_status == "private"
    return CheckResult(check="upload_as_private", passed=ok,
                       details=[] if ok else ["privacy_status must be 'private'"])


def uploads_today(ctx: CheckContext) -> int:
    tz = ZoneInfo(ctx.cfg("channel.timezone", "UTC"))
    now = ctx.now or datetime.now(timezone.utc)
    today = now.astimezone(tz).date()
    count = 0
    for rec in (ctx.store.uploads() if ctx.store else []):
        when = datetime.fromisoformat(rec["uploaded_at"]).astimezone(tz).date()
        if when == today and rec.get("status") == "uploaded":
            count += 1
    return count


def check_daily_cap(ctx: CheckContext) -> CheckResult:
    cap = int(ctx.cfg("publish.daily_upload_cap", 1))
    n = uploads_today(ctx)
    tzname = ctx.cfg("channel.timezone", "UTC")
    if n >= cap:
        return CheckResult(check="daily_upload_cap", passed=False,
                           details=[f"{n} upload(s) today ({tzname}) and the cap is {cap}"])
    return CheckResult(check="daily_upload_cap", passed=True, details=[f"{n}/{cap} uploads today ({tzname})"])
