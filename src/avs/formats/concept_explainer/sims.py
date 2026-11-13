"""Seeded simulations. Their outputs are both the animation data and the
'computation' facts the narration can cite, so the numbers on screen and the
numbers spoken are the same numbers."""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------- gradient descent
A, B = 1.0, 6.0          # f(x, y) = A x^2 + B y^2  (an elongated bowl)
START = (-4.0, 1.6)
TOL = 1e-3


def loss(x, y):
    return A * x ** 2 + B * y ** 2


def grad(x, y):
    return np.array([2 * A * x, 2 * B * y])


def descend(lr: float, steps: int = 80, start=START, noise: float = 0.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.array(start, dtype=float)
    path = [p.copy()]
    for _ in range(steps):
        g = grad(*p)
        if noise:
            g = g + rng.normal(0, noise, 2) * np.abs(g).mean()
        p = p - lr * g
        path.append(p.copy())
        if loss(*p) > 1e4:
            break
    return np.array(path)


def steps_to_converge(path: np.ndarray) -> int | None:
    for i, (x, y) in enumerate(path):
        if loss(x, y) < TOL:
            return i
    return None


def gd_summary(lrs=(0.02, 0.08, 0.17)) -> dict:
    out = {}
    for lr in lrs:
        path = descend(lr)
        losses = [float(loss(*p)) for p in path]
        out[str(lr)] = {"path": path.round(4).tolist(), "steps": steps_to_converge(path),
                        "final_loss": losses[-1], "start_loss": losses[0],
                        "diverged": losses[-1] > losses[0]}
    return out


# ----------------------------------------------------------------- sorting
def bubble_trace(arr: list[int]) -> tuple[list[tuple], int]:
    a, frames, comps = list(arr), [], 0
    n = len(a)
    for i in range(n):
        for j in range(n - 1 - i):
            comps += 1
            swapped = a[j] > a[j + 1]
            if swapped:
                a[j], a[j + 1] = a[j + 1], a[j]
            frames.append((list(a), (j, j + 1), comps))
    return frames, comps


def merge_trace(arr: list[int]) -> tuple[list[tuple], int]:
    a, frames = list(arr), []
    comps = 0

    def sort(lo, hi):
        nonlocal comps
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        sort(lo, mid)
        sort(mid, hi)
        left, right, merged = a[lo:mid], a[mid:hi], []
        i = j = 0
        while i < len(left) and j < len(right):
            comps += 1
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
            frames.append((a[:lo] + merged + left[i:] + right[j:] + a[hi:], (lo + len(merged) - 1, hi - 1), comps))
        merged += left[i:] + right[j:]
        a[lo:hi] = merged
        frames.append((list(a), (lo, hi - 1), comps))

    sort(0, len(a))
    return frames, comps


def sorting_summary(n: int = 24, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    arr = rng.permutation(np.arange(1, n + 1)).tolist()
    bf, bc = bubble_trace(arr)
    mf, mc = merge_trace(arr)
    return {"n": n, "array": arr, "bubble_comparisons": bc, "merge_comparisons": mc,
            "ratio": round(bc / mc, 1)}


# ----------------------------------------------------------------- RAG toy retrieval
DOCS = [
    ("Returns policy", "Items can be returned within 30 days with the original receipt for a full refund."),
    ("Shipping", "Standard shipping takes 3 to 5 business days. Express shipping arrives the next day."),
    ("Warranty", "Laptops include a one year warranty covering hardware faults, not accidental damage."),
    ("Store hours", "Our stores open at 9am and close at 8pm from Monday to Saturday."),
    ("Gift cards", "Gift cards never expire and can be used online or in any store."),
]
QUERY = "What is your returns policy for items?"


def rag_summary() -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    m = vec.fit_transform([f"{d[0]}. {d[1]}" for d in DOCS] + [QUERY])
    sims = (m[:-1] @ m[-1].T).toarray().ravel()
    order = np.argsort(-sims)
    return {"query": QUERY, "docs": [d[0] for d in DOCS], "texts": [d[1] for d in DOCS],
            "scores": [round(float(s), 2) for s in sims], "top": int(order[0]),
            "top_title": DOCS[int(order[0])][0], "top_score": round(float(sims[order[0]]), 2),
            "second_score": round(float(sims[order[1]]), 2)}
