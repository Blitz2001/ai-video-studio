"""Blocks scripts that are too close to a past video.

Embedders:
- local  : hashed word 1-2 gram + char 3-5 gram TF vectors (scikit-learn, no download,
           deterministic, CPU-cheap). Good at catching re-runs and light rewrites.
- gemini : Gemini text embeddings (semantic) when a key is configured.
`auto` picks gemini when the configured LLM can embed, else local. Each
embedder has its own threshold because their cosine scales differ.
"""
from __future__ import annotations

import logging

import numpy as np

from ..models import CheckResult
from .base import CheckContext

log = logging.getLogger("avs.similarity")

DEFAULT_THRESHOLDS = {"local": 0.80, "gemini": 0.93, "openai": 0.90}


class LocalEmbedder:
    name = "local"

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.word = HashingVectorizer(analyzer="word", ngram_range=(1, 2), n_features=2**18,
                                      alternate_sign=False, norm=None, stop_words="english")
        self.char = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**18,
                                      alternate_sign=False, norm=None)

    def embed(self, texts: list[str]) -> np.ndarray:
        from scipy.sparse import hstack
        from sklearn.preprocessing import normalize

        w = normalize(self.word.transform(texts).log1p())
        c = normalize(self.char.transform(texts).log1p())
        return normalize(hstack([w, c]).tocsr()).toarray()


class LLMEmbedder:
    def __init__(self, llm):
        self.llm = llm
        self.name = llm.name

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(self.llm.embed(texts), dtype=float)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def get_embedder(ctx: CheckContext):
    choice = ctx.cfg("compliance.embedder", "auto")
    if choice in ("auto", "gemini", "openai") and ctx.llm is not None and not ctx.llm.is_offline:
        try:
            emb = LLMEmbedder(ctx.llm)
            emb.embed(["probe"])
            return emb
        except Exception as exc:
            log.warning("remote embedder unavailable (%s); using local embedder", exc)
    return LocalEmbedder()


def most_similar(candidate: str, past: list[dict], embedder) -> tuple[float, dict | None]:
    if not past:
        return 0.0, None
    vecs = embedder.embed([candidate] + [p["text"] for p in past])
    sims = vecs[1:] @ vecs[0]
    i = int(np.argmax(sims))
    return float(sims[i]), past[i]


def check_similarity(ctx: CheckContext) -> CheckResult:
    if ctx.script is None:
        return CheckResult(check="not_too_similar_to_past_videos", passed=False, details=["no script"])
    past = ctx.store.past_scripts(exclude_run=ctx.run.id if ctx.run else None) if ctx.store else []
    embedder = get_embedder(ctx)
    thresholds = {**DEFAULT_THRESHOLDS, **(ctx.cfg("compliance.similarity_threshold", {}) or {})}
    threshold = thresholds.get(embedder.name, DEFAULT_THRESHOLDS["local"])
    score, match = most_similar(ctx.script.full_text(), past, embedder)
    detail = f"max cosine {score:.3f} vs threshold {threshold:.2f} ({embedder.name} embedder, {len(past)} past videos)"
    if match is not None and score >= threshold:
        return CheckResult(check="not_too_similar_to_past_videos", passed=False,
                           details=[detail, f"too close to {match['run_id']}: {match.get('title', '')!r}"])
    return CheckResult(check="not_too_similar_to_past_videos", passed=True, details=[detail])
