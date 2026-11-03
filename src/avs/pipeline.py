"""The production graph (LangGraph).

    research -> script -> script_gate --fail+LLM--> script (rewrite with feedback, max 2)
                               |--fail--> blocked
                               v
                 assets -> voice -> pre_render --fail--> blocked
                                        v
                                  edit -> package -> pre_review --fail--> blocked
                                                        v
                                                  awaiting_review   (human approves in the dashboard)

Publishing is a separate command (`avs publish`) that re-runs compliance with
the approval and daily-cap gates. Every node persists its output into the run
folder, so graph state stays tiny (ids + status) and each run is auditable.
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from .agents.editor import run_editor
from .agents.packaging import run_packaging
from .agents.voice import run_voice
from .compliance import CheckContext, run_stage
from .config import Settings
from .formats import FormatPlugin, RunContext, get_format
from .llm import get_llm
from .models import Asset, Metadata, RenderOutput, ResearchPack, Script, VoiceTrack
from .store import Run, Store

log = logging.getLogger("avs.pipeline")
MAX_REWRITES = 2


class State(TypedDict, total=False):
    run_id: str
    format: str
    topic: str
    status: str
    error: str
    rewrites: int


class Pipeline:
    def __init__(self, settings: Settings, llm_provider: str | None = None, llm=None):
        self.settings = settings
        self.store = Store(settings)
        self.llm = llm or get_llm(llm_provider or settings.llm_provider, settings.get("llm.fallback_to_offline", True))
        self._feedback: dict[str, list[str]] = {}
        self.graph = self._build()

    NODES = ["research", "script", "script_gate", "assets", "voice", "pre_render", "edit", "package",
             "pre_review", "queue"]

    # ------------------------------------------------------------------ helpers
    def _ctx(self, state: State) -> tuple[Run, FormatPlugin, RunContext]:
        run = self.store.get(state["run_id"])
        plugin = get_format(state["format"])
        ctx = RunContext(settings=self.settings, run=run, llm=self.llm, log=logging.getLogger(f"avs.{plugin.name}"),
                         feedback=self._feedback.get(run.id, []))
        return run, plugin, ctx

    def _check_ctx(self, run: Run) -> CheckContext:
        return CheckContext(
            settings=self.settings, research=run.load("research.json", ResearchPack),
            script=run.load("script.json", Script), assets=run.load_list("assets.json", Asset),
            voice=run.load("voice.json", VoiceTrack), render=run.load("render.json", RenderOutput),
            metadata=run.load("metadata.json", Metadata), run=run, store=self.store, llm=self.llm,
        )

    def _step(self, name: str, state: State, fn) -> State:
        run = self.store.get(state["run_id"])
        t0 = time.time()
        log.info("[%s] %s ...", run.id, name)
        try:
            out = fn() or {}
        except Exception as exc:
            run.update_info(status="failed", error=f"{name}: {exc}", traceback=traceback.format_exc())
            log.error("[%s] %s failed: %s", run.id, name, exc)
            return {**state, "status": "failed", "error": f"{name}: {exc}"}
        steps = run.info.get("steps", []) + [{"step": name, "seconds": round(time.time() - t0, 1)}]
        run.update_info(steps=steps)
        return {**state, **out}

    # ------------------------------------------------------------------ nodes
    def research(self, state: State) -> State:
        def go():
            run, plugin, ctx = self._ctx(state)
            run.write_json("research.json", plugin.research(state["topic"], ctx))
        return self._step("research", state, go)

    def script(self, state: State) -> State:
        def go():
            run, plugin, ctx = self._ctx(state)
            research = run.load("research.json", ResearchPack)
            script = plugin.write_script(research, ctx)
            run.write_json("script.json", script)
            run.update_info(title=script.title, written_by=script.written_by)
        return self._step("script", state, go)

    def _gate(self, stage: str, state: State) -> State:
        def go():
            run, plugin, _ = self._ctx(state)
            report = run_stage(stage, self._check_ctx(run), plugin.checks())
            log.info("\n%s", report.summary())
            if not report.passed:
                self._feedback[run.id] = [d for r in report.failures() for d in r.details]
                run.update_info(status="blocked", blocked_at=stage,
                                error="; ".join(r.check for r in report.failures()))
                return {"status": "blocked"}
            return {"status": "running"}
        return self._step(stage, state, go)

    def script_gate(self, state: State) -> State:
        return self._gate("script_gate", state)

    def pre_render(self, state: State) -> State:
        return self._gate("pre_render", state)

    def pre_review(self, state: State) -> State:
        return self._gate("pre_review", state)

    def assets(self, state: State) -> State:
        def go():
            run, plugin, ctx = self._ctx(state)
            assets = plugin.generate_assets(run.load("script.json", Script), run.load("research.json", ResearchPack), ctx)
            run.write_json("assets.json", [a.model_dump() for a in assets])
        return self._step("assets", state, go)

    def voice(self, state: State) -> State:
        def go():
            run, _, ctx = self._ctx(state)
            run.write_json("voice.json", run_voice(run.load("script.json", Script), ctx))
        return self._step("voice", state, go)

    def edit(self, state: State) -> State:
        def go():
            run, _, ctx = self._ctx(state)
            assets = run.load_list("assets.json", Asset)
            render, extra = run_editor(run.load("script.json", Script), assets, run.load("voice.json", VoiceTrack), ctx)
            run.write_json("render.json", render)
            run.write_json("assets.json", [a.model_dump() for a in assets + extra])
        return self._step("edit", state, go)

    def package(self, state: State) -> State:
        def go():
            run, _, ctx = self._ctx(state)
            assets = run.load_list("assets.json", Asset)
            meta, thumb = run_packaging(run.load("script.json", Script), run.load("research.json", ResearchPack),
                                        run.load("render.json", RenderOutput), assets,
                                        run.load("voice.json", VoiceTrack), ctx)
            run.write_json("metadata.json", meta)
            run.write_json("assets.json", [a.model_dump() for a in assets + [thumb]])
        return self._step("package", state, go)

    def queue(self, state: State) -> State:
        run = self.store.get(state["run_id"])
        run.update_info(status="awaiting_review", error=None)
        log.info("[%s] ready for review in the dashboard", run.id)
        return {**state, "status": "awaiting_review"}

    # ------------------------------------------------------------------ routing
    def _after_gate(self, state: State) -> Literal["ok", "stop"]:
        return "stop" if state.get("status") in ("blocked", "failed") else "ok"

    def _after_script_gate(self, state: State) -> Literal["ok", "rewrite", "stop"]:
        if state.get("status") == "blocked" and not self.llm.is_offline and state.get("rewrites", 0) < MAX_REWRITES:
            return "rewrite"
        return self._after_gate(state)

    def _bump(self, state: State) -> State:
        run = self.store.get(state["run_id"])
        run.update_info(status="running", blocked_at=None)
        return {**state, "status": "running", "rewrites": state.get("rewrites", 0) + 1}

    def _build(self, entry: str = "research"):
        g = StateGraph(State)
        for name in self.NODES:
            g.add_node(name, getattr(self, name))
        g.add_node("rewrite", self._bump)
        g.set_entry_point(entry)
        ok_or_stop = lambda nxt: {"ok": nxt, "stop": END}  # noqa: E731
        g.add_conditional_edges("research", self._after_gate, ok_or_stop("script"))
        g.add_conditional_edges("script", self._after_gate, ok_or_stop("script_gate"))
        g.add_conditional_edges("script_gate", self._after_script_gate,
                                {"ok": "assets", "rewrite": "rewrite", "stop": END})
        g.add_edge("rewrite", "script")
        g.add_conditional_edges("assets", self._after_gate, ok_or_stop("voice"))
        g.add_conditional_edges("voice", self._after_gate, ok_or_stop("pre_render"))
        g.add_conditional_edges("pre_render", self._after_gate, ok_or_stop("edit"))
        g.add_conditional_edges("edit", self._after_gate, ok_or_stop("package"))
        g.add_conditional_edges("package", self._after_gate, ok_or_stop("pre_review"))
        g.add_conditional_edges("pre_review", self._after_gate, ok_or_stop("queue"))
        g.add_edge("queue", END)
        return g.compile()

    # ------------------------------------------------------------------ entry point
    def run(self, fmt: str, topic: str | None = None) -> Run:
        plugin = get_format(fmt)
        topic = topic or plugin.default_topic
        run = self.store.new_run(fmt, topic)
        run.update_info(llm=self.llm.name, voice=self.settings.get("voice.backend"))
        final = self.graph.invoke({"run_id": run.id, "format": fmt, "topic": topic, "status": "running", "rewrites": 0},
                                  {"recursion_limit": 40})
        log.info("[%s] finished with status %s", run.id, final.get("status"))
        return run

    def resume(self, run_id: str, step: str) -> Run:
        """Re-run an existing run from `step` onwards (e.g. after editing the script
        or swapping in your own recorded voice). Approval is reset because the
        content will change."""
        if step not in self.NODES:
            raise ValueError(f"step must be one of {self.NODES}")
        run = self.store.get(run_id)
        info = run.info
        if (run.path / "review.json").exists():
            (run.path / "review.json").unlink()
        run.update_info(status="running", error=None, blocked_at=None)
        final = self._build(step).invoke({"run_id": run.id, "format": info["format"], "topic": info["topic"],
                                          "status": "running", "rewrites": 0}, {"recursion_limit": 40})
        log.info("[%s] resumed from %s, finished with status %s", run.id, step, final.get("status"))
        return run
