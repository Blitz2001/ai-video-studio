"""Copy the reviewable parts of a run into samples/<format>/ (videos, thumbnail,
captions, metadata, script with claims, compliance reports, licence log).

    python scripts/export_sample.py <run_id>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ["out/video_16x9.mp4", "out/short_9x16.mp4", "out/thumbnail.png", "out/captions.srt", "metadata.json",
        "script.json", "research.json", "licence_log.json", "render.json", "voice.json", "run.json",
        "compliance/script_gate.json", "compliance/pre_render.json", "compliance/pre_review.json",
        "build/build_log.json"]


def main(run_id: str) -> int:
    run = ROOT / "runs" / run_id
    info = json.loads((run / "run.json").read_text())
    dest = ROOT / "samples" / info["format"]
    if dest.exists():
        shutil.rmtree(dest)
    for rel in KEEP:
        src = run / rel
        if src.exists():
            out = dest / (Path(rel).name if rel.startswith("out/") else rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, out)
    print(f"exported {run_id} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
