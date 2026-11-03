from avs.compliance.claims import check_claims, is_factual, number_matches, numbers_in

from .conftest import script, segment


def test_fully_cited_script_passes(make_ctx):
    seg = segment(
        "In 1952, average life expectancy was 48.9 years. By 2007 it was 68.9 years. So what changed?",
        [("In 1952, average life expectancy was 48.9 years.", "analysis"), ("By 2007 it was 68.9 years.", "analysis")],
    )
    r = check_claims(make_ctx(script=script(seg)))
    assert r.passed, r.details


def test_uncited_number_fails(make_ctx):
    seg = segment("Life expectancy rose by 20 years.")
    r = check_claims(make_ctx(script=script(seg)))
    assert not r.passed
    assert "uncited factual sentence" in r.details[0]


def test_uncited_superlative_without_numbers_fails(make_ctx):
    seg = segment("Asia improved the fastest of any region.")
    assert not check_claims(make_ctx(script=script(seg))).passed


def test_unknown_source_fails(make_ctx):
    seg = segment("There are 142 countries.", [("There are 142 countries.", "made_up_source")])
    r = check_claims(make_ctx(script=script(seg)))
    assert not r.passed and "unknown source" in r.details[0]


def test_source_without_url_is_not_citable(make_ctx):
    seg = segment("Most countries improved.", [("Most countries improved.", "rumour")])
    r = check_claims(make_ctx(script=script(seg)))
    assert not r.passed and "cannot be cited" in " ".join(r.details)


def test_hallucinated_number_with_real_source_fails(make_ctx):
    # the source is real, but 71.3 is not in any of its facts
    seg = segment("By 2007 it was 71.3 years.", [("By 2007 it was 71.3 years.", "analysis")])
    r = check_claims(make_ctx(script=script(seg)))
    assert not r.passed and "71.3" in " ".join(r.details)


def test_rounding_to_claimed_precision_is_allowed():
    facts = numbers_in("48.9 years")
    assert number_matches((49.0, 0), facts)      # "about 49" for 48.9
    assert not number_matches((52.0, 0), facts)  # but not 52


def test_numbers_must_appear_in_the_covering_claim(make_ctx):
    # claim covers the words but drops the number that is spoken
    seg = segment("In 1952 life expectancy was 48.9 years.", [("life expectancy was low", "analysis")])
    assert not check_claims(make_ctx(script=script(seg))).passed


def test_questions_and_plain_sentences_need_no_claim():
    assert not is_factual("How long will we live in 2050?")
    assert not is_factual("Let's look at the data.")
    assert is_factual("Merge sort was invented by John von Neumann in 1945.")
    assert is_factual("Bubble sort is slower than merge sort.")
