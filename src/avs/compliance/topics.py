"""Blocks health, finance and legal *advice*.

Talking about these domains is fine (a data story on life expectancy is not
medical advice). What is blocked is telling the viewer what to do about their
health, money or legal situation. A sentence is blocked when it pairs an
advice-intent pattern with a domain term; a topic/title is blocked when it is
itself an advice request ("how to cure...", "best stocks to buy").

When an LLM is configured it is asked as a second opinion. It can add blocks,
never remove them, so the deterministic layer is the floor.
"""
from __future__ import annotations

import re

from ..models import CheckResult
from .base import CheckContext, split_sentences

DOMAINS: dict[str, list[str]] = {
    "health": [
        r"symptoms?", r"diagnos\w*", r"treat(?:ment|ing|s)?", r"cures?", r"medications?", r"medicines?",
        r"drugs?", r"dos(?:e|es|age)", r"\d+\s?mg", r"supplements?", r"vitamins?", r"diet(?:ing)?",
        r"weight loss", r"lose weight", r"diseases?", r"cancer", r"diabetes", r"therapy", r"vaccines?",
        r"prescriptions?", r"pills?", r"antibiotics?", r"blood pressure", r"cholesterol", r"fasting",
        r"detox", r"illness(?:es)?", r"anxiety", r"depression", r"insomnia", r"pregnan\w+",
    ],
    "finance": [
        r"invest\w*", r"stocks?", r"shares", r"crypto\w*", r"bitcoin", r"ethereum", r"portfolio",
        r"trading", r"day trade\w*", r"retirement", r"401k", r"pension", r"loans?", r"mortgages?",
        r"debt", r"credit cards?", r"credit score", r"tax(?:es)?", r"savings", r"etfs?", r"index funds?",
        r"mutual funds?", r"forex", r"options trading", r"passive income", r"get rich", r"returns",
        r"dividends?", r"real estate", r"budget(?:ing)?",
    ],
    "legal": [
        r"lawsuits?", r"\bsue\b", r"suing", r"court", r"contracts?", r"lawyers?", r"attorneys?",
        r"legal(?:ly)?", r"divorce", r"custody", r"visas?", r"immigration", r"tenants?", r"landlords?",
        r"liabilit\w+", r"copyright strike", r"arrest(?:ed)?", r"police", r"evictions?", r"wills?\b",
        r"inheritance", r"loophole", r"settlement",
    ],
}

ADVICE_INTENT = [
    r"\byou (?:should|must|need to|ought to|have to|'d better|had better)\b",
    r"\b(?:i|we) (?:recommend|suggest|advise)\b",
    r"\b(?:my|our) (?:advice|recommendation)\b",
    r"\bmake sure (?:to|you)\b",
    r"\b(?:don't|do not|never|always) (?:take|buy|sell|invest|sign|stop|skip|eat|use|pay|trust|file)\b",
    r"^(?:take|buy|sell|invest|consult|stop taking|start taking|avoid|sue|file|sign|try|eat|put your|move your|switch to)\b",
    r"\bhow to (?:treat|cure|heal|invest|sue|avoid (?:paying )?tax\w*|lose weight|get rich|beat the market|win (?:a|your) (?:case|lawsuit))\b",
    r"\bguaranteed (?:returns?|profits?|cure|results)\b",
    r"\bbefore it(?:'s| is) too late\b",
    r"\b(?:best|top) (?:stocks?|coins?|cryptos?|supplements?|diet|treatment|way to (?:invest|treat|cure))\b",
    r"\bthe right (?:dose|dosage|investment)\b",
    r"\bthis is (?:not )?(?:financial|medical|legal) advice\b",  # disclaimers signal advice content
]
TOPIC_BLOCKLIST = [
    r"\bshould i (?:invest|buy|sell|take|sue)\b",
    r"\b(?:best|top) \d* ?(?:stocks?|coins?|cryptos?|etfs?|supplements?)\b",
    r"\bhow to (?:cure|treat|heal|invest|sue|get rich|lose weight|avoid tax\w*|beat the market)\b",
    r"\b(?:cure|treatment) for\b",
    r"\bfinancial freedom\b",
    r"\blegal loopholes?\b",
]

_ADVICE = [re.compile(p, re.I) for p in ADVICE_INTENT]
_DOMAIN = {d: re.compile(r"\b(?:%s)" % "|".join(pats), re.I) for d, pats in DOMAINS.items()}
_TOPIC = [re.compile(p, re.I) for p in TOPIC_BLOCKLIST]

LLM_PROMPT = """You are a strict content-policy reviewer for an educational YouTube channel.
Does the following script GIVE health, finance or legal ADVICE to the viewer (telling them what to do
about their own health, money or legal situation)? Discussing data or history is NOT advice.
Answer JSON: {{"advice": true|false, "domain": "health|finance|legal|none", "evidence": "quote or empty"}}

TITLE: {title}
SCRIPT:
{text}"""


def scan_text(text: str, domains: list[str]) -> list[str]:
    hits = []
    for sentence in split_sentences(text):
        if not any(p.search(sentence) for p in _ADVICE):
            continue
        for d in domains:
            if _DOMAIN[d].search(sentence):
                hits.append(f"{d} advice: {sentence!r}")
                break
    return hits


def scan_topic(topic: str, domains: list[str]) -> list[str]:
    hits = []
    for p in _TOPIC:
        if p.search(topic):
            hits.append(f"topic is an advice request: {topic!r}")
            break
    return hits


def check_topics(ctx: CheckContext) -> CheckResult:
    domains = ctx.cfg("compliance.blocked_advice_domains", list(DOMAINS))
    details: list[str] = []
    topic = ctx.script.topic if ctx.script else (ctx.research.topic if ctx.research else "")
    title = ctx.script.title if ctx.script else ""
    details += scan_topic(f"{topic} | {title}", domains)
    if ctx.script:
        for seg in ctx.script.segments:
            details += [f"[{seg.id}] {h}" for h in scan_text(seg.narration, domains)]
    if ctx.metadata:
        details += [f"[description] {h}" for h in scan_text(ctx.metadata.description, domains)]

    if ctx.llm is not None and not ctx.llm.is_offline and ctx.script is not None:
        try:
            verdict = ctx.llm.complete_json(LLM_PROMPT.format(title=title, text=ctx.script.full_text()))
            if verdict.get("advice") and verdict.get("domain") in domains:
                details.append(f"LLM reviewer: {verdict['domain']} advice: {verdict.get('evidence', '')!r}")
        except Exception as exc:  # the deterministic layer already ran; it is the floor
            import logging

            logging.getLogger("avs.topics").warning("LLM topic reviewer unavailable: %s", exc)

    return CheckResult(check="no_health_finance_legal_advice", passed=not details, details=details)
