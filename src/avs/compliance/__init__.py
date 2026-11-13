"""Compliance layer. Stages and the checks they run:

    script_gate  right after the script: claims, advice topics, similarity
    pre_render   before the editor renders: + asset licences + format checks
    pre_review   before the run enters the review queue: + disclosure, private, metadata attributions
    pre_upload   in the publisher, right before upload: + human approval + daily cap

Any blocking failure stops the pipeline; nothing reaches YouTube without
passing pre_upload, and pre_upload requires a dashboard approval.
"""
from __future__ import annotations

from typing import Callable

from ..models import CheckResult, ComplianceReport
from .base import CheckContext
from .claims import check_claims
from .disclosure import check_disclosure
from .licences import check_licences
from .publishing import check_approval, check_daily_cap, check_private
from .similarity import check_similarity
from .topics import check_topics

CheckFn = Callable[[CheckContext], CheckResult]

STAGES: dict[str, list[CheckFn]] = {
    "script_gate": [check_claims, check_topics, check_similarity],
    "pre_render": [check_claims, check_topics, check_similarity, check_licences],
    "pre_review": [check_claims, check_topics, check_licences, check_disclosure, check_private],
    "pre_upload": [check_claims, check_topics, check_licences, check_disclosure, check_private,
                   check_approval, check_daily_cap],
}
FORMAT_CHECK_STAGES = {"pre_render", "pre_review", "pre_upload"}


def run_stage(stage: str, ctx: CheckContext, extra_checks: list[CheckFn] | None = None) -> ComplianceReport:
    checks = list(STAGES[stage])
    if stage in FORMAT_CHECK_STAGES:
        checks += extra_checks or []
    report = ComplianceReport(stage=stage)
    for fn in checks:
        try:
            result = fn(ctx)
        except Exception as exc:  # a crashing check is a failing check
            result = CheckResult(check=getattr(fn, "__name__", "check"), passed=False,
                                 details=[f"check crashed: {exc!r}"])
        report.results.append(result)
    if ctx.run is not None:
        ctx.run.write_json(f"compliance/{stage}.json", report)
    return report


__all__ = ["CheckContext", "run_stage", "STAGES"]
