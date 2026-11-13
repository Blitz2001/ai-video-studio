"""Pillow text rendering for captions, title cards and thumbnails."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import theme


def font(size: int, weight: str = "bold") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(theme.font_path(weight)[0], size)


def hex_rgba(h: str, a: int = 255) -> tuple[int, int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a


def wrap(text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if fnt.getlength(trial) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def caption_image(text: str, max_width: int, size: int = 54, pad: int = 22) -> np.ndarray:
    """White text on a translucent rounded plate; returns an RGBA array."""
    fnt = font(size, "bold")
    lines = wrap(text, fnt, max_width - 2 * pad)
    line_h = int(size * 1.25)
    w = int(max(fnt.getlength(l) for l in lines)) + 2 * pad
    h = line_h * len(lines) + 2 * pad - int(size * 0.2)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=18, fill=(10, 10, 10, 190))
    for i, line in enumerate(lines):
        lw = fnt.getlength(line)
        d.text(((w - lw) / 2, pad + i * line_h - size * 0.1), line, font=fnt, fill=hex_rgba(theme.TEXT))
    return np.array(img)


def card_image(size: tuple[int, int], title: str, kicker: str, footer: str = "") -> np.ndarray:
    """Full-frame title / outro card."""
    W, H = size
    img = Image.new("RGB", (W, H), hex_rgba(theme.SURFACE)[:3])
    d = ImageDraw.Draw(img)
    portrait = H > W
    margin = int(W * 0.08)
    k_font = font(int(H * (0.028 if portrait else 0.045)), "semibold")
    t_font = font(int(H * (0.05 if portrait else 0.085)), "bold")
    f_font = font(int(H * (0.02 if portrait else 0.03)), "regular")
    y = int(H * (0.36 if portrait else 0.30))
    # accent bar + kicker
    d.rectangle([margin, y, margin + int(W * 0.06), y + 10], fill=theme.ACCENT)
    y += 40
    d.text((margin, y), kicker.upper(), font=k_font, fill=theme.TEXT_2)
    y += int(k_font.size * 1.8)
    for line in wrap(title, t_font, W - 2 * margin):
        d.text((margin, y), line, font=t_font, fill=theme.TEXT)
        y += int(t_font.size * 1.18)
    if footer:
        fy = H - margin - f_font.size * 2
        for line in wrap(footer, f_font, W - 2 * margin)[:3]:
            d.text((margin, fy), line, font=f_font, fill=theme.MUTED)
            fy += int(f_font.size * 1.4)
    return np.array(img)
