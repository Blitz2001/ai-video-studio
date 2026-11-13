"""Frame-by-frame matplotlib animation piped straight into ffmpeg.

A scene is `setup(fig) -> update(t)` where t runs 0..1 over the clip. This is
CPU-only, deterministic and fast enough on a laptop (no Manim/LaTeX needed).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import theme  # noqa: E402

Update = Callable[[float], None]
Setup = Callable[[plt.Figure], Update]


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def ease(t: float) -> float:
    """easeInOutCubic"""
    t = min(1.0, max(0.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def phase(t: float, start: float, end: float) -> float:
    """Map global t to 0..1 within [start, end]."""
    if end <= start:
        return 1.0 if t >= end else 0.0
    return min(1.0, max(0.0, (t - start) / (end - start)))


def render(setup: Setup, out: Path, duration: float, fps: int = 30, size=(1920, 1080)) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = size
    with plt.rc_context(theme.mpl_style()):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        update = setup(fig)
        n = max(2, int(round(duration * fps)))
        cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
               "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            for i in range(n):
                update(i / (n - 1))
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                proc.stdin.write(buf.tobytes())
        finally:
            proc.stdin.close()
            proc.wait()
            plt.close(fig)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {out}")
    return out


def frame_header(fig: plt.Figure, title: str, source: str | None = None) -> None:
    """Title top-left, source line under it. Bottom ~22% is kept clear for captions."""
    fig.text(0.05, 0.93, title, fontsize=40, fontweight="bold", color=theme.TEXT, va="top")
    if source:
        fig.text(0.05, 0.855, source, fontsize=19, color=theme.MUTED, va="top")


def estimate_seconds(narration: str, wpm: float = 165) -> float:
    return max(3.0, len(narration.split()) / wpm * 60 + 0.6)


def bullet_card(p: dict):
    """Title + bullets fading in one by one. Shared by formats for recap/caveat beats."""
    from matplotlib.patches import FancyBboxPatch

    def setup(fig):
        frame_header(fig, p["title"], None)
        items, y = [], 0.72
        for line in p["lines"]:
            items.append(fig.text(0.09, y, line, fontsize=30, color=theme.TEXT, va="top"))
            fig.patches.append(FancyBboxPatch((0.065, y - 0.028), 0.012, 0.022, boxstyle="round,pad=0.002",
                                              transform=fig.transFigure, color=theme.ACCENT, figure=fig))
            y -= 0.1
        for tx in items:
            tx.set_alpha(0)

        def update(t):
            for i, tx in enumerate(items):
                tx.set_alpha(ease(phase(t, 0.05 + i * 0.12, 0.2 + i * 0.12)))
        return update
    return setup
