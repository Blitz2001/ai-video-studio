"""Editor agent.

Builds a 16:9 video (intro card -> segments -> outro card) and, optionally, a
9:16 short from the segments marked `in_short`, with burned-in captions (also
exported as .srt for YouTube), self-generated music under the voice, and
crossfade transitions. Visuals are time-fitted to the narration: sped up (up
to 2.5x) when too long, held on the last frame when too short.

Implementation note: an early version composited every frame in moviepy. On a
CPU laptop that was ~10x slower than real time, so all compositing now happens
inside ffmpeg filtergraphs (overlay / xfade / acrossfade / amix), which keeps a
2-minute 1080p render to a few minutes. Pillow renders the caption PNGs.
"""
from __future__ import annotations

import re
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..formats.base import RunContext
from ..media import theme
from ..media.anim import ffmpeg_bin
from ..media.music import generate_music
from ..media.text import caption_image, card_image, font, wrap
from ..models import Asset, RenderOutput, Script, VoiceTrack

XFADE = 0.35
VIDEO_EXT = {".mp4", ".webm", ".mov", ".mkv", ".gif"}
AUDIO = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


@dataclass
class Cue:
    start: float
    end: float
    text: str


# ----------------------------------------------------------------------------- captions
def chunk_text(text: str, max_words: int = 7) -> list[str]:
    chunks: list[str] = []
    for sentence in re.split(r"(?<=[.!?;:])\s+", " ".join(text.split())):
        words = sentence.split()
        if not words:
            continue
        n = max(1, int(np.ceil(len(words) / max_words)))
        size = int(np.ceil(len(words) / n))
        chunks += [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    return chunks


def caption_cues(text: str, speech_seconds: float, offset: float) -> list[Cue]:
    chunks = chunk_text(text)
    weights = np.array([len(c) + 6 for c in chunks], dtype=float)
    bounds = np.concatenate([[0], np.cumsum(weights / weights.sum() * speech_seconds)])
    return [Cue(offset + bounds[i], offset + bounds[i + 1], c) for i, c in enumerate(chunks)]


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[Cue], path: Path) -> Path:
    lines = []
    for i, c in enumerate(cues, 1):
        lines += [str(i), f"{srt_time(c.start)} --> {srt_time(c.end)}", c.text, ""]
    path.write_text("\n".join(lines))
    return path


# ----------------------------------------------------------------------------- ffmpeg helpers
def ff(args: list[str]) -> None:
    proc = subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-2000:]}")


def media_duration(path: Path) -> float:
    proc = subprocess.run([ffmpeg_bin(), "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", proc.stderr)
    if not m:
        raise RuntimeError(f"cannot read duration of {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _png(arr: np.ndarray, path: Path) -> Path:
    Image.fromarray(arr).save(path)
    return path


def _heading(text: str, W: int) -> np.ndarray:
    from PIL import ImageDraw

    fnt = font(64, "bold")
    lines = wrap(text, fnt, int(W * 0.86))[:3]
    h = int(len(lines) * 64 * 1.2) + 20
    img = Image.new("RGBA", (W, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text(((W - fnt.getlength(line)) / 2, i * 64 * 1.2), line, font=fnt, fill=theme.TEXT)
    return np.array(img)


def _segment_asset(assets: list[Asset], seg_id: str, portrait: bool = False) -> Asset | None:
    """Prefer a native portrait render for the short; fall back to the landscape one."""
    visual = [a for a in assets if a.segment_id == seg_id and a.kind in ("animation", "image", "screen_recording")]
    want = "portrait" if portrait else "landscape"
    exact = [a for a in visual if a.meta.get("aspect", "landscape") == want]
    if exact:
        return exact[0]
    landscape = [a for a in visual if a.meta.get("aspect", "landscape") == "landscape"]
    return (landscape or visual or [None])[0]


# ----------------------------------------------------------------------------- building blocks
def build_segment(seg, voice_path: Path, asset: Asset | None, ctx: RunContext, size, portrait: bool,
                  title: str, work: Path) -> tuple[Path, float, list[Cue]]:
    W, H = size
    fps = int(ctx.settings.get("render.fps", 30))
    D = media_duration(voice_path)
    inputs: list[str] = []
    graph: list[str] = []
    surface = theme.SURFACE.replace("#", "0x")

    # [0] the visual: native 4:5 render if the plugin made one, else the 16:9 one
    native = asset is not None and asset.meta.get("aspect") == "portrait"
    vis_box = ((W, int(W * 1.25)) if native else (W, 2 * round(W * 9 / 32))) if portrait else (W, H)
    if asset is None:
        inputs += ["-f", "lavfi", "-t", f"{D}", "-i", f"color=c={surface}:s={vis_box[0]}x{vis_box[1]}:r={fps}"]
        fit = "null"
    else:
        p = ctx.run.path / asset.path
        if p.suffix.lower() in VIDEO_EXT:
            vd = media_duration(p)
            factor = min(vd / D, 2.5) if vd > D else 1.0
            inputs += ["-i", str(p)]
            fit = f"setpts=PTS/{factor:.4f},"
        else:
            inputs += ["-loop", "1", "-t", f"{D}", "-i", str(p)]
            fit = ""
        fit += (f"fps={fps},scale={vis_box[0]}:{vis_box[1]}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                f"pad={vis_box[0]}:{vis_box[1]}:(ow-iw)/2:(oh-ih)/2:color={surface},"
                f"tpad=stop_mode=clone:stop_duration={D:.3f},trim=duration={D:.3f},setsar=1")
    graph.append(f"[0:v]{fit}[vis]")
    n_in = 1
    if portrait:
        inputs += ["-f", "lavfi", "-t", f"{D}", "-i", f"color=c={surface}:s={W}x{H}:r={fps}"]
        head = _png(_heading(seg.on_screen_text or title, W), work / f"{seg.id}_head.png")
        inputs += ["-i", str(head)]
        vis_y = int(H * 0.135) if native else int(H * 0.30)
        graph.append(f"[{n_in}:v][vis]overlay=0:{vis_y}[b0]")
        graph.append(f"[b0][{n_in + 1}:v]overlay=0:{int(H * (0.035 if native else 0.12))}[v0]")
        n_in += 2
        cap_w, cap_size = int(W * 0.86), 58
        cap_y = vis_y + vis_box[1] - 230 if native else vis_y + vis_box[1] + 90
    else:
        graph.append("[vis]null[v0]")
        cap_w, cap_size, cap_y = int(W * 0.72), 50, None

    cues = caption_cues(seg.narration, max(0.5, D - 0.35), 0.0)
    last = "v0"
    for k, c in enumerate(cues):
        img = caption_image(c.text, cap_w, size=cap_size)
        png = _png(img, work / f"{seg.id}_{'p' if portrait else 'l'}_cap{k}.png")
        inputs += ["-i", str(png)]
        y = cap_y if cap_y is not None else H - img.shape[0] - 48
        graph.append(f"[{last}][{n_in}:v]overlay=x=(W-w)/2:y={y}:enable='between(t,{c.start:.3f},{c.end:.3f})'[c{k}]")
        last = f"c{k}"
        n_in += 1
    inputs += ["-i", str(voice_path)]
    graph.append(f"[{n_in}:a]aresample=48000,aformat=channel_layouts=stereo[aout]")
    out = work / f"{seg.id}_{'p' if portrait else 'l'}.mp4"
    ff([*inputs, "-filter_complex", ";".join(graph), "-map", f"[{last}]", "-map", "[aout]",
        "-t", f"{D:.3f}", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", *AUDIO, str(out)])
    return out, D, cues


def build_card(img: np.ndarray, seconds: float, ctx: RunContext, path: Path) -> Path:
    fps = int(ctx.settings.get("render.fps", 30))
    png = _png(img, path.with_suffix(".png"))
    ff(["-loop", "1", "-t", f"{seconds}", "-i", str(png), "-f", "lavfi", "-t", f"{seconds}",
        "-i", "anullsrc=r=48000:cl=stereo", "-r", str(fps), "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", *AUDIO, "-shortest", str(path)])
    return path


def join(clips: list[tuple[Path, float]], music: Path, music_gain: float, ctx: RunContext, out: Path) -> float:
    inputs, graph = [], []
    for p, _ in clips:
        inputs += ["-i", str(p)]
    vlast, alast, offset = "0:v", "0:a", clips[0][1]
    for i in range(1, len(clips)):
        offset -= XFADE
        graph.append(f"[{vlast}][{i}:v]xfade=transition=fade:duration={XFADE}:offset={offset:.3f}[xv{i}]")
        graph.append(f"[{alast}][{i}:a]acrossfade=d={XFADE}[xa{i}]")
        vlast, alast = f"xv{i}", f"xa{i}"
        offset += clips[i][1]
    total = offset
    inputs += ["-i", str(music)]
    m = len(clips)
    graph.append(f"[{m}:a]atrim=0:{total:.3f},volume={music_gain:.4f},afade=t=in:d=1.5,"
                 f"afade=t=out:st={max(0, total - 2.5):.3f}:d=2.5,aformat=channel_layouts=stereo[mus]")
    graph.append(f"[{alast}][mus]amix=inputs=2:duration=first:normalize=0[aout]")
    s = ctx.settings
    ff([*inputs, "-filter_complex", ";".join(graph), "-map", f"[{vlast}]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(s.get("render.crf", 23)), "-pix_fmt", "yuv420p",
        *AUDIO, "-movflags", "+faststart", str(out)])
    return total


# ----------------------------------------------------------------------------- orchestration
def render_video(script: Script, assets: list[Asset], voice: VoiceTrack, ctx: RunContext,
                 portrait: bool = False) -> tuple[Path, list[Cue], list[tuple[float, str]], float]:
    s = ctx.settings
    size = tuple(s.get("render.portrait" if portrait else "render.landscape"))
    lines = {l.segment_id: l for l in voice.lines}
    segments = list(script.segments)
    if portrait:
        budget, chosen, total = float(s.get("render.short_max_seconds", 58)) - 2.5, [], 0.0
        for seg in segments:
            if seg.in_short and total + lines[seg.id].duration <= budget:
                chosen.append(seg)
                total += lines[seg.id].duration
        segments = chosen or segments[:1]
    work = ctx.run.sub("work")
    tag = "p" if portrait else "l"

    clips: list[tuple[Path, float]] = []
    cues: list[Cue] = []
    chapters: list[tuple[float, str]] = []
    t = 0.0
    if not portrait:
        intro_d = 2.6
        clips.append((build_card(card_image(size, script.title, ctx.brand), intro_d, ctx, work / "intro_l.mp4"), intro_d))
        t = intro_d - XFADE
    for seg in segments:
        path, dur, seg_cues = build_segment(seg, ctx.run.path / lines[seg.id].path, _segment_asset(assets, seg.id, portrait),
                                            ctx, size, portrait, script.title, work)
        chapters.append((t if clips else 0.0, seg.on_screen_text or seg.id.replace("_", " ").title()))
        cues += [Cue(c.start + t, c.end + t, c.text) for c in seg_cues]
        clips.append((path, dur))
        t += dur - XFADE
        ctx.log.info("segment %s (%s) %.1fs", seg.id, tag, dur)
    disclosure = "Narration: AI voice (Kokoro TTS)" if voice.realistic else ""
    footer = " · ".join(x for x in [disclosure, "Visuals generated with code", "Sources in the description"] if x)
    outro_d = 2.5 if portrait else 4.0
    outro = card_image(size, f"Subscribe to {ctx.brand}", s.get("channel.tagline", ""), footer)
    clips.append((build_card(outro, outro_d, ctx, work / f"outro_{tag}.mp4"), outro_d))

    music = ctx.run.sub("assets") / "music.wav"
    total_est = sum(d for _, d in clips)
    generate_music(total_est + 2, music, seed=zlib.crc32(script.title.encode()) % 97)
    gain = 10 ** (float(s.get("render.music_volume_db", -26)) / 20)
    out = ctx.run.sub("out") / ("short_9x16.mp4" if portrait else "video_16x9.mp4")
    total = join(clips, music, gain, ctx, out)
    return out, cues, chapters, total


def run_editor(script: Script, assets: list[Asset], voice: VoiceTrack, ctx: RunContext) -> tuple[RenderOutput, list[Asset]]:
    out, cues, chapters, duration = render_video(script, assets, voice, ctx, portrait=False)
    srt = write_srt(cues, ctx.run.sub("out") / "captions.srt")
    portrait = None
    if ctx.settings.get("render.make_short", True):
        portrait, _, _, _ = render_video(script, assets, voice, ctx, portrait=True)
    rel = lambda p: str(Path(p).relative_to(ctx.run.path)) if p else None  # noqa: E731
    music = Asset(id="music", segment_id=None, kind="music", path="assets/music.wav", licence="self-generated",
                  origin="numpy-synthesised ambient pad (avs.media.music)")
    font_path, font_licence = theme.font_path("bold")
    font_asset = Asset(id="font", segment_id=None, kind="font", path=font_path, licence=font_licence,
                       origin="caption and card typeface", meta={"external_path": True})
    return RenderOutput(landscape=rel(out), portrait=rel(portrait), srt=rel(srt), duration=duration,
                        chapters=chapters), [music, font_asset]
