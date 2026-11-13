"""Asset licence log: only self-generated or properly licensed visuals/music.

Every asset (and every dataset source) needs a licence from the allow-list.
Licences that require attribution must carry attribution text, and once
metadata exists that text must appear in the video description.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..models import Asset, CheckResult
from .base import CheckContext


def licence_rows(ctx: CheckContext) -> list[dict]:
    rows = []
    for a in ctx.assets:
        rows.append({"id": a.id, "kind": a.kind, "path": a.path, "licence": a.licence or "",
                     "origin": a.origin, "attribution": a.attribution or ""})
    if ctx.research:
        for s in ctx.research.sources:
            if s.kind == "dataset":
                rows.append({"id": s.id, "kind": "dataset", "path": s.url or "", "licence": s.licence or "",
                             "origin": s.publisher or s.title, "attribution": f"{s.title} - {s.publisher}"})
    return rows


def write_licence_log(ctx: CheckContext) -> Path | None:
    if ctx.run is None:
        return None
    rows = licence_rows(ctx)
    ctx.run.write_json("licence_log.json", rows)
    # global append-only CSV for the whole channel
    if ctx.store is not None:
        log_path = ctx.store.state_dir / "licence_log.csv"
        new = not log_path.exists()
        with open(log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_id", "id", "kind", "path", "licence", "origin", "attribution"])
            if new:
                w.writeheader()
            for r in rows:
                w.writerow({"run_id": ctx.run.id, **r})
    return ctx.run.path / "licence_log.json"


def _check_asset(a: Asset, allowed: set[str], needs_attr: set[str], base: Path | None) -> list[str]:
    errs = []
    if not a.licence:
        errs.append(f"asset {a.id} ({a.kind}) has no licence recorded")
    elif a.licence not in allowed:
        errs.append(f"asset {a.id} licence {a.licence!r} is not on the allow-list {sorted(allowed)}")
    if a.licence in needs_attr and not a.attribution:
        errs.append(f"asset {a.id} is {a.licence} but has no attribution text")
    if base is not None and not (base / a.path).exists():
        errs.append(f"asset {a.id} file missing: {a.path}")
    return errs


def check_licences(ctx: CheckContext) -> CheckResult:
    allowed = set(ctx.cfg("compliance.allowed_licences", ["self-generated", "CC0-1.0", "CC-BY-4.0", "public-domain"]))
    needs_attr = set(ctx.cfg("compliance.attribution_required", ["CC-BY-4.0"]))
    base = ctx.run.path if ctx.run else None
    details: list[str] = []
    if not ctx.assets:
        details.append("no assets recorded; every visual and audio file must be logged")
    for a in ctx.assets:
        details += _check_asset(a, allowed, needs_attr, base)

    attributions = [a.attribution for a in ctx.assets if a.licence in needs_attr and a.attribution]
    if ctx.research:
        for s in ctx.research.sources:
            if s.kind != "dataset":
                continue
            if s.licence not in allowed:
                details.append(f"dataset {s.id} licence {s.licence!r} is not on the allow-list")
            elif s.licence in needs_attr:
                attributions.append(s.publisher or s.title)
    if ctx.metadata is not None:
        desc = ctx.metadata.description.lower()
        for attr in set(attributions):
            if attr.lower() not in desc:
                details.append(f"attribution {attr!r} missing from the video description")

    write_licence_log(ctx)
    return CheckResult(check="asset_licences", passed=not details, details=details)
