from avs.compliance.similarity import LocalEmbedder, check_similarity, most_similar

from .conftest import script, segment

PAST = ("Gradient descent is how neural networks learn. Picture the loss as a valley and the model as a ball. "
        "At every point we measure the slope and take a step downhill. The learning rate sets how big each step is: "
        "too small and training crawls, too big and the ball overshoots and the loss explodes.")
PARAPHRASE = ("Gradient descent is how neural nets learn. Think of the loss as a valley and the model as a ball. "
              "At each point we measure the slope and step downhill. The learning rate sets how big every step is: "
              "too small and training crawls, too large and the ball overshoots and the loss blows up.")
DIFFERENT = ("Merge sort splits a list in half, sorts each half and merges them back together. Bubble sort swaps "
             "neighbours until nothing is out of order. On large inputs, divide and conquer wins by a wide margin.")


def test_scores_are_ordered_sensibly():
    emb = LocalEmbedder()
    same, _ = most_similar(PAST, [{"run_id": "a", "text": PAST}], emb)
    para, _ = most_similar(PARAPHRASE, [{"run_id": "a", "text": PAST}], emb)
    diff, _ = most_similar(DIFFERENT, [{"run_id": "a", "text": PAST}], emb)
    assert same > 0.99
    assert para > 0.80 > diff
    assert diff < 0.4


def test_paraphrase_of_past_video_is_blocked(make_ctx, store):
    store.add_history("old-run", "Gradient descent explained", PAST)
    r = check_similarity(make_ctx(script=script(segment(PARAPHRASE))))
    assert not r.passed
    assert "old-run" in r.details[1]


def test_new_topic_passes(make_ctx, store):
    store.add_history("old-run", "Gradient descent explained", PAST)
    r = check_similarity(make_ctx(script=script(segment(DIFFERENT))))
    assert r.passed, r.details


def test_pending_runs_count_as_past_but_rejected_and_self_do_not(make_ctx, store):
    pending = store.new_run("concept_explainer", "gd")
    pending.write_json("script.json", script(segment(PAST)))
    pending.update_info(status="awaiting_review")
    assert not check_similarity(make_ctx(script=script(segment(PARAPHRASE)))).passed

    pending.update_info(status="rejected")
    assert check_similarity(make_ctx(script=script(segment(PARAPHRASE)))).passed

    pending.update_info(status="awaiting_review")
    assert check_similarity(make_ctx(script=script(segment(PARAPHRASE)), run=pending)).passed  # itself


def test_no_history_passes(make_ctx):
    assert check_similarity(make_ctx(script=script(segment(PAST)))).passed
