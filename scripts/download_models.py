"""Download the free CPU voice model (Kokoro-82M, Apache-2.0) into models/kokoro/.

    python scripts/download_models.py

Files come from the kokoro-onnx project's release assets on GitHub. The int8
model is ~90 MB and runs faster than real time on a laptop CPU.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
FILES = {
    "kokoro-v1.0.int8.onnx": f"{BASE}/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": f"{BASE}/voices-v1.0.bin",
}


def main() -> int:
    dest = Path(__file__).resolve().parents[1] / "models" / "kokoro"
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        out = dest / name
        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"ok      {out}")
            continue
        print(f"fetch   {url}")
        urllib.request.urlretrieve(url, out)
        print(f"saved   {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print("\nSet voice.backend: kokoro in config/studio.yaml (the default).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
