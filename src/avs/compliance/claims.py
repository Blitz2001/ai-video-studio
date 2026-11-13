"""Every factual claim needs a source.

Three layers, all blocking:
1. citation integrity  - each Claim names a source that exists in the research
                         pack and is citable (url / dataset / code reference).
2. coverage            - each narration sentence that *looks* factual (numbers,
                         dates, rankings, comparisons, attributions) must be
                         covered by a Claim in the same segment, and every number
                         in the sentence must appear in that claim.
3. grounding           - every number in a claim must match a number in a fact
                         from the cited source (to the claim's precision), so a
                         model cannot cite a real source for an invented figure.
"""
from __future__ import annotations

import re

from ..models import CheckResult, Claim
from .base import CheckContext, split_sentences

NUMBER_WORDS = (
    "two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
    "hundred thousand million billion trillion percent half twice double doubled tripled "
    "quadrupled halved dozen"
).split()

FACTUAL_PATTERNS = [
    r"\d",
    r"\b(?:%s)\b" % "|".join(NUMBER_WORDS),
    r"\b(?:most|least|highest|lowest|largest|smallest|fastest|slowest|biggest|best|worst|record|"
    r"top|majority|minority|every|always|never|only)\b",
    r"\b(?:more|less|fewer|greater|faster|slower|higher|lower|bigger|smaller|longer|shorter|richer|poorer)\s+than\b",
    r"\baccording to\b",
    r"\b(?:invented|discovered|published|founded|introduced|proved|proposed|coined)\b",
]
_FACTUAL = re.compile("|".join(FACTUAL_PATTERNS), re.I)
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of and or to in on at for is are was were be been by with as that this it its from".split())


def is_factual(sentence: str) -> bool:
    s = sentence.strip()
    if s.endswith("?"):
        return False  # a question asserts nothing
    return bool(_FACTUAL.search(s))


def numbers_in(text: str) -> list[tuple[float, int]]:
    """Return (value, decimals) for each number in text."""
    out = []
    for m in _NUM.finditer(text):
        whole = m.group(1).replace(",", "")
        dec = m.group(2) or ""
        out.append((float(f"{whole}.{dec}" if dec else whole), len(dec)))
    return out


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in STOP}


def number_matches(claimed: tuple[float, int], facts: list[tuple[float, int]]) -> bool:
    value, decimals = claimed
    for fv, _ in facts:
        if round(fv, decimals) == value or abs(fv - value) < 1e-9:
            return True
        # allow "about 49" for 49.4 but not 49 for 52: rounding to claimed precision only
    return False


def claim_covers(claim: Claim, sentence: str) -> bool:
    cn = " ".join(_WORD.findall(claim.text.lower()))
    sn = " ".join(_WORD.findall(sentence.lower()))
    if cn and (cn in sn or sn in cn):
        pass
    else:
        st, ct = _tokens(sentence), _tokens(claim.text)
        if not st or len(st & ct) / len(st) < 0.6:
            return False
    sent_nums = {v for v, _ in numbers_in(sentence)}
    claim_nums = {v for v, _ in numbers_in(claim.text)}
    return sent_nums <= claim_nums


def check_claims(ctx: CheckContext) -> CheckResult:
    script, research = ctx.script, ctx.research
    details: list[str] = []
    if script is None or research is None:
        return CheckResult(check="claims_have_sources", passed=False, details=["no script/research to check"])

    for seg in script.segments:
        for claim in seg.claims:
            src = research.source(claim.source_id)
            if src is None:
                details.append(f"[{seg.id}] claim cites unknown source '{claim.source_id}': {claim.text!r}")
                continue
            if not src.is_citable():
                details.append(f"[{seg.id}] source '{src.id}' has no url/reference, so it cannot be cited")
            # grounding: numbers in the claim must exist in that source's facts
            fact_nums = [n for f in research.facts if f.source_id == src.id for n in numbers_in(f.text)]
            for num in numbers_in(claim.text):
                if not number_matches(num, fact_nums):
                    details.append(f"[{seg.id}] number {num[0]:g} in claim {claim.text!r} is not in any fact from '{src.id}'")
        for sentence in split_sentences(seg.narration):
            if is_factual(sentence) and not any(claim_covers(c, sentence) for c in seg.claims):
                details.append(f"[{seg.id}] uncited factual sentence: {sentence!r}")

    return CheckResult(check="claims_have_sources", passed=not details, details=details)
