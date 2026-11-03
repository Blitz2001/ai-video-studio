"""YouTube altered/synthetic content disclosure.

YouTube asks creators to disclose realistic content made with AI that viewers
could mistake for a real person, place or event. A realistic neural voice
counts, so we set `status.containsSyntheticMedia = true` whenever a realistic
TTS backend or a realistic synthetic person is used, and say so in the
description. Your own recorded voice, or an obviously robotic one, needs no flag.
"""
from __future__ import annotations

from ..models import CheckResult
from .base import CheckContext

DISCLOSURE_MARKER = "AI-generated narration"


def requires_disclosure(ctx: CheckContext) -> tuple[bool, list[str]]:
    reasons = []
    realistic = set(ctx.cfg("compliance.realistic_voice_backends", ["kokoro", "piper"]))
    if ctx.voice and (ctx.voice.realistic or ctx.voice.backend in realistic):
        reasons.append(f"realistic synthetic voice ({ctx.voice.backend}/{ctx.voice.voice})")
    for a in ctx.assets:
        if a.realistic_person:
            reasons.append(f"realistic synthetic person in asset {a.id}")
    return bool(reasons), reasons


def check_disclosure(ctx: CheckContext) -> CheckResult:
    needed, reasons = requires_disclosure(ctx)
    if ctx.metadata is None:
        return CheckResult(check="synthetic_content_disclosure", passed=False, details=["no metadata yet"])
    details = []
    if needed and not ctx.metadata.contains_synthetic_media:
        details.append("containsSyntheticMedia must be true: " + "; ".join(reasons))
    if needed and DISCLOSURE_MARKER.lower() not in ctx.metadata.description.lower():
        details.append(f"description must include the disclosure line ('{DISCLOSURE_MARKER}')")
    if not needed and ctx.metadata.contains_synthetic_media:
        # over-disclosing is allowed, but flag it so it is a choice
        return CheckResult(check="synthetic_content_disclosure", passed=False, severity="warn",
                           details=["flag set although no realistic synthetic media was detected"])
    return CheckResult(check="synthetic_content_disclosure", passed=not details,
                       details=details or [("disclosed: " + "; ".join(reasons)) if needed else "no disclosure needed"])
