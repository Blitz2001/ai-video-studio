"""Voice agent: one wav per segment.

Backends (all free, all CPU):
- kokoro   : Kokoro-82M ONNX (Apache-2.0) via kokoro-onnx. Realistic -> disclosure flag.
- piper    : Piper ONNX voices via piper-tts. Realistic -> disclosure flag.
- recorded : your own voice. Put recordings/<run_id>/<segment_id>.wav (or
             recordings/<segment_id>.wav). `avs record-sheet <run_id>` prints the lines.
- silent   : timed silence (tests / CI / previewing visuals).
"""
from __future__ import annotations

import re
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..formats.base import RunContext
from ..media.music import wav_duration, write_wav
from ..models import Script, VoiceLine, VoiceTrack

TAIL_SILENCE = 0.35


def tts_text(text: str) -> str:
    """Make written text speakable without changing what the captions show."""
    t = text.replace("%", " percent").replace("&", " and ").replace("—", ", ").replace("–", " to ")
    t = re.sub(r"\bO\(n\^?2\)|O\(n²\)", "O of n squared", t)
    t = re.sub(r"\bO\(n log n\)", "O of n log n", t)
    t = re.sub(r"\bO\(n\)", "O of n", t)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = t.replace("e.g.", "for example").replace("i.e.", "that is").replace(" vs ", " versus ")
    return t


@lru_cache(maxsize=2)
def _kokoro(model: str, voices: str):
    from kokoro_onnx import Kokoro

    return Kokoro(model, voices)


def _pad(samples: np.ndarray, sr: int) -> np.ndarray:
    return np.concatenate([samples, np.zeros(int(TAIL_SILENCE * sr), dtype=samples.dtype)])


def synth_kokoro(text: str, out: Path, ctx: RunContext) -> None:
    s = ctx.settings
    k = _kokoro(str(s.path(s.get("voice.kokoro_model"))), str(s.path(s.get("voice.kokoro_voices"))))
    samples, sr = k.create(tts_text(text), voice=s.get("voice.kokoro_voice", "am_michael"),
                           speed=float(s.get("voice.kokoro_speed", 1.0)), lang="en-us")
    write_wav(out, _pad(np.asarray(samples, dtype=np.float32), sr), sr)


def synth_piper(text: str, out: Path, ctx: RunContext) -> None:
    from piper import PiperVoice

    voice = PiperVoice.load(str(ctx.settings.path(ctx.settings.get("voice.piper_model"))))
    with wave.open(str(out), "wb") as w:
        voice.synthesize_wav(tts_text(text), w)
    with wave.open(str(out), "rb") as r:
        sr = r.getframerate()
        pcm = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).astype(np.float32) / 32767
    write_wav(out, _pad(pcm, sr), sr)


def use_recording(segment_id: str, out: Path, ctx: RunContext) -> None:
    base = ctx.settings.path(ctx.settings.get("voice.recorded_dir", "recordings"))
    for cand in (base / ctx.run.id / f"{segment_id}.wav", base / f"{segment_id}.wav"):
        if cand.exists():
            from moviepy import AudioFileClip

            clip = AudioFileClip(str(cand))
            arr = clip.to_soundarray(fps=24000)
            mono = arr.mean(axis=1) if arr.ndim == 2 else arr
            write_wav(out, _pad(mono.astype(np.float32), 24000), 24000)
            return
    raise FileNotFoundError(f"no recording for segment {segment_id} in {base}/{ctx.run.id}/ or {base}/")


def synth_silent(text: str, out: Path) -> None:
    seconds = max(1.5, len(text.split()) / 2.6)  # ~156 wpm
    write_wav(out, np.zeros(int(seconds * 24000), dtype=np.float32), 24000)


def run_voice(script: Script, ctx: RunContext) -> VoiceTrack:
    backend = ctx.settings.get("voice.backend", "kokoro")
    if backend == "kokoro" and not ctx.settings.path(ctx.settings.get("voice.kokoro_model", "")).exists():
        ctx.log.warning("Kokoro model not found; falling back to silent voice. See README -> Voice setup.")
        backend = "silent"
    realistic = backend in set(ctx.settings.get("compliance.realistic_voice_backends", []))
    out_dir = ctx.run.sub("voice")
    lines = []
    for seg in script.segments:
        out = out_dir / f"{seg.id}.wav"
        if backend == "kokoro":
            synth_kokoro(seg.narration, out, ctx)
        elif backend == "piper":
            synth_piper(seg.narration, out, ctx)
        elif backend == "recorded":
            use_recording(seg.id, out, ctx)
        elif backend == "silent":
            synth_silent(seg.narration, out)
        else:
            raise ValueError(f"unknown voice backend {backend!r}")
        lines.append(VoiceLine(segment_id=seg.id, path=str(out.relative_to(ctx.run.path)), duration=wav_duration(out)))
        ctx.log.info("voice %s: %.1fs", seg.id, lines[-1].duration)
    voice_name = ctx.settings.get("voice.kokoro_voice") if backend == "kokoro" else None
    return VoiceTrack(backend=backend, voice=voice_name, lines=lines, realistic=realistic)
