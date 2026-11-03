"""Channel look: one dark theme shared by charts, captions, cards and thumbnails.

Series colours are the dark-mode steps of a CVD-validated categorical palette,
used in fixed order (never cycled). Scatter/bubble charts use at most 3 slots.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

SURFACE = "#1a1a19"
SURFACE_2 = "#242422"
GRID = "#3a3a37"
TEXT = "#f0efec"
TEXT_2 = "#c3c2b7"
MUTED = "#8d8c84"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
ACCENT = SERIES[0]
HIGHLIGHT = SERIES[1]

_INTER_DIRS = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/inter",
               str(Path.home() / ".fonts"), str(Path.home() / ".local/share/fonts"),
               "/Library/Fonts", "C:/Windows/Fonts"]


@lru_cache(maxsize=None)
def font_path(weight: str = "bold") -> tuple[str, str]:
    """Return (path, licence). Prefers Inter (OFL-1.1), falls back to the DejaVu
    Sans that ships inside matplotlib, so it works on any laptop."""
    names = {"bold": ["Inter-Bold.otf", "Inter-Bold.ttf", "InterDisplay-Bold.otf"],
             "semibold": ["Inter-SemiBold.otf", "Inter-SemiBold.ttf"],
             "regular": ["Inter-Regular.otf", "Inter-Regular.ttf"]}[weight]
    for d in _INTER_DIRS:
        for n in names:
            p = Path(d) / n
            if p.exists():
                return str(p), "OFL-1.1"
    import matplotlib

    data = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fallback = {"bold": "DejaVuSans-Bold.ttf", "semibold": "DejaVuSans-Bold.ttf", "regular": "DejaVuSans.ttf"}[weight]
    return str(data / fallback), "Bitstream-Vera"


def mpl_style() -> dict:
    """rcParams for matplotlib renders."""
    from matplotlib import font_manager

    path, _ = font_path("regular")
    bold, _ = font_path("bold")
    for p in (path, bold):
        try:
            font_manager.fontManager.addfont(p)
        except Exception:
            pass
    family = font_manager.FontProperties(fname=path).get_name()
    return {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "font.family": family, "font.size": 20, "axes.titlesize": 30, "axes.labelsize": 20,
        "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
        "legend.frameon": False, "lines.linewidth": 3.0,
    }
