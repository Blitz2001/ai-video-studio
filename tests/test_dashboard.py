"""Drive the real Streamlit dashboard headlessly: approval must come from here,
be bound to the content hash, and be voided by a metadata edit."""
import pytest

from avs.compliance.publishing import check_approval
from avs.agents.publisher import load_check_context

from .conftest import REPO

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app(root, monkeypatch):
    monkeypatch.setenv("AVS_ROOT", str(root))
    return AppTest.from_file(str(REPO / "dashboard" / "app.py"), default_timeout=60)


def _button(at, label):
    return next(b for b in at.button if b.label.startswith(label))


def test_approve_in_dashboard_unlocks_upload_gate(root, monkeypatch, run_with_video, settings, store):
    run = run_with_video()
    at = _app(root, monkeypatch).run()
    assert not at.exception
    _button(at, "✅ Approve").click().run()
    review = run.review()
    assert review.status == "approved" and review.approved_via == "dashboard"
    assert review.content_hash == run.content_hash()
    assert check_approval(load_check_context(run, settings, store)).passed


def test_editing_metadata_voids_approval(root, monkeypatch, run_with_video, settings, store):
    run = run_with_video()
    at = _app(root, monkeypatch).run()
    _button(at, "✅ Approve").click().run()
    assert run.review().status == "approved"
    at.text_input[1].input("A different title").run()   # [0] is reviewer name, [1] is the title field
    _button(at, "Save metadata").click().run()
    assert run.review().status == "pending"
    assert not check_approval(load_check_context(run, settings, store)).passed


def test_reject(root, monkeypatch, run_with_video):
    run = run_with_video()
    at = _app(root, monkeypatch).run()
    _button(at, "⛔ Reject").click().run()
    assert run.review().status == "rejected" and run.info["status"] == "rejected"
