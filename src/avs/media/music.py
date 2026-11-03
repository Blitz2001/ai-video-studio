"""Self-generated background music: a slow ambient chord pad synthesised with
numpy. No samples, no third-party audio, so its licence is simply
'self-generated'. Deterministic per seed, so every video gets a stable bed."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 44100
# i - vi - IV - V in C major, voiced as (root, third, fifth, octave) in Hz
PROGRESSIONS = [
    [(130.81, 164.81, 196.00, 261.63), (110.00, 130.81, 164.81, 220.00),
     (87.31, 110.00, 130.81, 174.61), (98.00, 123.47, 146.83, 196.00)],
    [(110.00, 130.81, 164.81, 220.00), (87.31, 110.00, 130.81, 174.61),
     (130.81, 164.81, 196.00, 261.63), (98.00, 123.47, 146.83, 196.00)],
]


def _pad_tone(freq: float, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    detune = 1 + rng.uniform(-0.003, 0.003, size=3)
    tone = sum(np.sin(2 * np.pi * freq * d * t + rng.uniform(0, 6.28)) for d in detune) / 3
    tone += 0.25 * np.sin(2 * np.pi * freq * 2 * t)  # soft 2nd harmonic
    return tone


def generate_music(duration: float, out: Path, seed: int = 7, bar_seconds: float = 4.0) -> Path:
    rng = np.random.default_rng(seed)
    prog = PROGRESSIONS[seed % len(PROGRESSIONS)]
    n_bar = int(bar_seconds * SR)
    bars = int(np.ceil(duration / bar_seconds)) + 1
    xfade = int(0.8 * SR)
    total = np.zeros(bars * n_bar + xfade)
    env = np.ones(n_bar + xfade)
    env[:xfade] = np.linspace(0, 1, xfade)
    env[-xfade:] = np.linspace(1, 0, xfade)
    for b in range(bars):
        chord = prog[b % len(prog)]
        seg = sum(_pad_tone(f, n_bar + xfade, rng) * (0.9 if i == 0 else 0.6) for i, f in enumerate(chord))
        total[b * n_bar: b * n_bar + n_bar + xfade] += seg * env
    # gentle low-pass via moving average and a slow tremolo
    k = 24
    total = np.convolve(total, np.ones(k) / k, mode="same")
    t = np.arange(len(total)) / SR
    total *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.08 * t)
    total = total[: int(duration * SR)]
    fade = int(min(2.0, duration / 4) * SR)
    total[:fade] *= np.linspace(0, 1, fade)
    total[-fade:] *= np.linspace(1, 0, fade)
    total /= max(1e-9, np.abs(total).max())
    total *= 0.9
    write_wav(out, total, SR)
    return out


def write_wav(path: Path, samples: np.ndarray, sr: int, channels: int = 1) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())
