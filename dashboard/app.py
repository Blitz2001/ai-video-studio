"""Review dashboard: the only place a video can be approved.

    avs dashboard        (or: streamlit run dashboard/app.py)

Approving records reviewer, time and a hash of the exact video + thumbnail +
metadata you saw. Editing metadata here voids any earlier approval. The
publisher refuses anything not approved here, anything changed since, and
anything over the daily cap.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from avs.agents.publisher import load_check_context, publish_run
from avs.compliance import run_stage
from avs.compliance.publishing import uploads_today
from avs.config import load_settings
from avs.formats import get_format
from avs.models import Metadata, Review, Script, utcnow
from avs.store import Store

st.set_page_config(page_title="AI Video Studio · Review", page_icon="🎬", layout="wide")

settings = load_settings()
store = Store(settings)
cap = int(settings.get("publish.daily_upload_cap", 1))
STATUS_ICON = {"awaiting_review": "🟡", "approved": "🟢", "published": "✅", "rejected": "⛔", "blocked": "🚫",
               "failed": "💥", "running": "⏳"}

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title(f"🎬 {settings.channel_name}")
st.sidebar.caption(settings.get("channel.niche", ""))
filt = st.sidebar.multiselect("Show", list(STATUS_ICON), default=["awaiting_review", "approved", "blocked", "failed"])
runs = store.list_runs(set(filt) if filt else None)
ctx0 = load_check_context(runs[0], settings, store) if runs else None
st.sidebar.metric("Uploads today", f"{uploads_today(ctx0) if ctx0 else 0} / {cap}",
                  help=f"Daily cap, counted in {settings.get('channel.timezone')}")
reviewer = st.sidebar.text_input("Reviewer name", value=st.session_state.get("reviewer", "me"))
st.session_state["reviewer"] = reviewer

if not runs:
    st.info("No runs yet. Start one with `avs run --format data_story`.")
    st.stop()

labels = [f"{STATUS_ICON.get(r.status, '•')} {r.info.get('title') or r.id}  ·  {r.info.get('format')}" for r in runs]
choice = st.sidebar.radio("Runs", range(len(runs)), format_func=lambda i: labels[i])
run = runs[choice]
info = run.info
fmt = get_format(info["format"])

# ----------------------------------------------------------------------------- header
st.title(info.get("title") or run.id)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", info["status"].replace("_", " "))
c2.metric("Format", info["format"])
c3.metric("LLM", info.get("written_by") or info.get("llm", "?"))
c4.metric("Voice", info.get("voice", "?"))
if info.get("error"):
    st.error(f"{info.get('blocked_at') or 'error'}: {info['error']}")

render = run.read_json("render.json") or {}
meta_raw = run.read_json("metadata.json")
tab_video, tab_script, tab_comp, tab_meta, tab_lic = st.tabs(
    ["Video", "Script & sources", "Compliance", "Metadata", "Licence log"])

with tab_video:
    v1, v2 = st.columns([16, 7])
    if render.get("landscape"):
        v1.video(str(run.path / render["landscape"]))
        v1.caption(f"16:9 · {render.get('duration', 0):.0f}s")
    if render.get("portrait"):
        v2.video(str(run.path / render["portrait"]))
        v2.caption("9:16 short")
    if meta_raw and meta_raw.get("thumbnail"):
        try:
            st.image(str(run.path / meta_raw["thumbnail"]), caption="Thumbnail", width=480)
        except Exception as exc:  # a broken file should not hide the rest of the review
            st.warning(f"Thumbnail could not be shown: {exc}")

with tab_script:
    script = run.load("script.json", Script)
    research = run.read_json("research.json") or {}
    sources = {s["id"]: s for s in research.get("sources", [])}
    if script:
        for seg in script.segments:
            st.markdown(f"**{seg.on_screen_text or seg.id}**  \n{seg.narration}")
            for c in seg.claims:
                s = sources.get(c.source_id, {})
                link = s.get("url") or s.get("reference") or ""
                st.caption(f"↳ “{c.text}” — {s.get('title', c.source_id)} {link}")
    with st.expander("Research facts"):
        st.json(research.get("facts", []))

with tab_comp:
    for stage in ("script_gate", "pre_render", "pre_review", "pre_upload"):
        rep = run.read_json(f"compliance/{stage}.json")
        if not rep:
            continue
        ok = all(r["passed"] or r["severity"] == "warn" for r in rep["results"])
        with st.expander(f"{'✅' if ok else '❌'} {stage}  ·  {rep['created_at']}", expanded=not ok):
            for r in rep["results"]:
                icon = "✅" if r["passed"] else ("⚠️" if r["severity"] == "warn" else "❌")
                st.markdown(f"{icon} `{r['check']}`")
                for d in r["details"][:10]:
                    st.caption(d)
    if st.button("Re-run upload checks now (dry run)"):
        rep = run_stage("pre_upload", load_check_context(run, settings, store), fmt.checks())
        st.code(rep.summary())

with tab_meta:
    if meta_raw:
        meta = Metadata.model_validate(meta_raw)
        with st.form("meta"):
            title = st.text_input("Title", meta.title, max_chars=100)
            desc = st.text_area("Description", meta.description, height=320)
            tags = st.text_input("Tags (comma separated)", ", ".join(meta.tags))
            st.checkbox("Contains realistic synthetic media (set by compliance)", meta.contains_synthetic_media,
                        disabled=True)
            st.caption("Privacy: private (always). Saving edits voids any earlier approval.")
            if st.form_submit_button("Save metadata"):
                meta.title, meta.description = title, desc
                meta.tags = [t.strip() for t in tags.split(",") if t.strip()]
                run.write_json("metadata.json", meta)
                if run.review().status == "approved":
                    run.write_json("review.json", Review(status="pending", note="metadata edited after approval"))
                    run.update_info(status="awaiting_review")
                st.success("Saved. Review and approve again.")
                st.rerun()

with tab_lic:
    rows = run.read_json("licence_log.json") or []
    st.dataframe(rows, width="stretch")

# ----------------------------------------------------------------------------- decision
st.divider()
review = run.review()
st.subheader("Decision")
if review.status != "pending":
    st.write(f"Current: **{review.status}** by {review.reviewer} at {review.decided_at}"
             + (f" — {review.note}" if review.note else ""))
note = st.text_input("Note (optional)")
a, b, c = st.columns(3)
can_review = info["status"] in ("awaiting_review", "approved", "rejected")
if a.button("✅ Approve for private upload", disabled=not can_review, type="primary"):
    run.write_json("review.json", Review(status="approved", reviewer=reviewer, approved_via="dashboard",
                                         decided_at=utcnow(), note=note or None, content_hash=run.content_hash()))
    run.update_info(status="approved")
    st.rerun()
if b.button("⛔ Reject", disabled=not can_review):
    run.write_json("review.json", Review(status="rejected", reviewer=reviewer, approved_via="dashboard",
                                         decided_at=utcnow(), note=note or None))
    run.update_info(status="rejected")
    st.rerun()
if c.button("🚀 Publish now (private)", disabled=info["status"] != "approved"):
    try:
        report, vid = publish_run(run, settings, store)
        if vid:
            st.success(f"Uploaded as PRIVATE: https://studio.youtube.com/video/{vid}/edit")
        else:
            st.error("Blocked by pre_upload compliance")
            st.code(report.summary())
    except Exception as exc:
        st.error(f"Upload failed: {exc}")
