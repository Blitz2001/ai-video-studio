import pytest

from avs.compliance.topics import check_topics, scan_text, scan_topic
from avs.llm import LLM

from .conftest import script, segment

DOMAINS = ["health", "finance", "legal"]


@pytest.mark.parametrize("text", [
    "Life expectancy rose from 48.9 to 68.9 years between 1952 and 2007.",
    "Cancer death rates fell in most countries in the dataset.",
    "Stock prices are a classic example of a random walk.",
    "You should normalise your features before training.",
    "Measure the slope and step the other way.",
])
def test_discussing_a_domain_is_not_advice(text):
    assert scan_text(text, DOMAINS) == []


@pytest.mark.parametrize("text,domain", [
    ("You should take 500mg of ibuprofen for back pain.", "health"),
    ("Stop taking your medication if you feel better.", "health"),
    ("Buy Bitcoin now before it's too late.", "finance"),
    ("I recommend putting your savings into index funds.", "finance"),
    ("You should sue your landlord for that.", "legal"),
    ("Never sign a contract without reading clause 4.", "legal"),
])
def test_advice_is_blocked(text, domain):
    hits = scan_text(text, DOMAINS)
    assert hits and hits[0].startswith(domain)


@pytest.mark.parametrize("topic", ["best stocks to buy in 2026", "how to cure insomnia", "Should I invest in crypto?",
                                   "legal loopholes for landlords"])
def test_advice_topics_are_blocked(topic):
    assert scan_topic(topic, DOMAINS)


def test_check_blocks_script(make_ctx):
    s = script(segment("Here's the trick: you should invest your savings in crypto."), title="Crypto")
    r = check_topics(make_ctx(script=s))
    assert not r.passed and "finance" in r.details[0]


def test_disabled_domain_is_not_blocked(make_ctx, settings):
    settings.raw["compliance"]["blocked_advice_domains"] = ["health"]
    s = script(segment("You should invest your savings in crypto."))
    assert check_topics(make_ctx(script=s)).passed


class FakeReviewer(LLM):
    name = "fake"

    def __init__(self, verdict):
        self.verdict = verdict

    def _complete(self, prompt, system, json_mode):
        import json
        return json.dumps(self.verdict)


def test_llm_reviewer_can_add_a_block(make_ctx):
    s = script(segment("Eating less sugar is the key to living longer, so cut it today."))
    ctx = make_ctx(script=s, llm=FakeReviewer({"advice": True, "domain": "health", "evidence": "cut it today"}))
    r = check_topics(ctx)
    assert not r.passed and "LLM reviewer" in r.details[0]


def test_llm_reviewer_cannot_remove_a_block(make_ctx):
    s = script(segment("You should take 500mg of ibuprofen."))
    ctx = make_ctx(script=s, llm=FakeReviewer({"advice": False, "domain": "none"}))
    assert not check_topics(ctx).passed
