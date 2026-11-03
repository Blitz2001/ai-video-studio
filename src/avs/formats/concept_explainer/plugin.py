from __future__ import annotations

import yaml

from ...media.anim import estimate_seconds, render
from ...models import Asset, CheckResult, Fact, ResearchPack, Script, Source
from ..base import FormatPlugin, RunContext
from ..writing import Beat
from . import sims
from .scenes import SCENES

SIM_SOURCE = Source(id="simulation", title="Seeded simulation written for this video", kind="computation",
                    reference="avs/formats/concept_explainer/sims.py (deterministic, seed 7)")


def pick_concept(topic: str, kb: dict) -> str:
    t = topic.lower().strip().replace(" ", "_")
    if t in kb:
        return t
    words = topic.lower()
    scores = {k: sum(1 for kw in v.get("keywords", []) if kw in words) for k, v in kb.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        raise ValueError(f"no concept matches {topic!r}; add it to concepts.yaml")
    return best


class ConceptExplainer(FormatPlugin):
    name = "concept_explainer"
    description = "Animate a CS/ML concept (gradient descent, sorting, RAG) with seeded matplotlib simulations."
    default_topic = "gradient_descent"
    visual_kinds = tuple(SCENES)

    def _kb(self) -> dict:
        return yaml.safe_load((self.folder / "concepts.yaml").read_text())

    # ------------------------------------------------------------------ research
    def research(self, topic: str, ctx: RunContext) -> ResearchPack:
        kb = self._kb()
        key = pick_concept(topic, kb)
        entry = kb[key]
        sources = [Source(**s) for s in entry["sources"]] + [SIM_SOURCE]
        facts = [Fact(**f) for f in entry["facts"]]
        builder = getattr(self, f"_research_{key}")
        sim_facts, storyboard, sim = builder()
        return ResearchPack(format=self.name, topic=key, angle=entry["title"], sources=sources,
                            facts=facts + sim_facts, data={"concept": key, "storyboard": storyboard, "sim": sim})

    def _research_gradient_descent(self):
        runs = sims.gd_summary()
        small, good, big = runs["0.02"], runs["0.08"], runs["0.17"]
        full = sims.descend(0.05, steps=45)
        noisy = sims.descend(0.05, steps=45, noise=1.6)
        facts = [
            Fact(id="sim_good", source_id="simulation",
                 text=f"With a learning rate of 0.08 the loss fell below 0.001 after {good['steps']} steps."),
            Fact(id="sim_small", source_id="simulation",
                 text=f"With a learning rate of 0.02 the loss was still {small['final_loss']:.4f} (about 0.02) after 80 steps."),
            Fact(id="sim_big", source_id="simulation",
                 text=f"With a learning rate of 0.17 the loss grew from {big['start_loss']:.2f} to {big['final_loss']:.1f} "
                      f"(over 8,000): it diverged."),
            Fact(id="sim_limit", source_id="simulation",
                 text="On this surface, loss = x^2 + 6y^2, any learning rate above 1/6 (about 0.167) diverges."),
        ]
        src = "simulation: loss = x² + 6y², start (-4, 1.6)"
        sb = [
            {"id": "hook", "on_screen_text": "Learning = walking downhill", "in_short": True, "facts": ["sim_good"],
             "visual": {"kind": "contour_descent", "params": {"title": "Gradient descent", "source": src,
                                                              "path": good["path"][:good["steps"] + 1]}}},
            {"id": "slope", "on_screen_text": "Follow the slope down", "in_short": True, "facts": ["def"],
             "visual": {"kind": "gradient_1d", "params": {"title": "One parameter, one slope",
                                                          "source": "loss = w², learning rate 0.3", "lr": 0.3, "x0": -2.6, "steps": 5}}},
            {"id": "learning_rate", "on_screen_text": "The learning rate", "in_short": True, "hero": True,
             "facts": ["lr_def", "sim_small", "sim_good", "sim_big", "sim_limit"],
             "visual": {"kind": "lr_compare", "params": {"title": "Same valley, three learning rates", "source": src,
                                                         "runs": [{"lr": 0.02, "label": "0.02  too small", "path": small["path"]},
                                                                  {"lr": 0.08, "label": "0.08  about right", "path": good["path"]},
                                                                  {"lr": 0.17, "label": "0.17  too big", "path": big["path"]}]}}},
            {"id": "stochastic", "on_screen_text": "Why real training looks noisy", "in_short": False, "facts": ["sgd"],
             "visual": {"kind": "sgd_vs_gd", "params": {"title": "Full batch vs mini-batch", "source": src + ", noise seed 7",
                                                        "full": full.round(4).tolist(), "noisy": noisy.round(4).tolist()}}},
            {"id": "history", "on_screen_text": "Recap", "in_short": False, "facts": ["cauchy"],
             "visual": {"kind": "bullet_card", "params": {"title": "Gradient descent in four lines", "lines": [
                 "Measure the slope (the gradient)", "Step the opposite way", "Step size = learning rate",
                 "Repeat until the loss stops falling"]}}},
        ]
        return facts, sb, {"steps_good": good["steps"]}

    def _research_sorting(self):
        s = sims.sorting_summary()
        facts = [
            Fact(id="sim_counts", source_id="simulation",
                 text=f"On the same {s['n']} shuffled bars, bubble sort made {s['bubble_comparisons']} comparisons "
                      f"and merge sort made {s['merge_comparisons']}, about {s['ratio']} times fewer."),
            Fact(id="sim_1000", source_id="simulation",
                 text="For 1,000 items, n squared over 2 is 499,500 comparisons while n log n is about 9,966."),
        ]
        src = f"simulation: {s['n']} bars, seed 7"
        sb = [
            {"id": "hook", "on_screen_text": "Same bars, two algorithms", "in_short": True, "hero": True,
             "facts": ["sim_counts"],
             "visual": {"kind": "sort_race", "params": {"title": "Bubble sort vs merge sort", "source": src, "array": s["array"]}}},
            {"id": "bubble", "on_screen_text": "Bubble sort", "in_short": True, "facts": ["bubble_def", "bubble_big_o"],
             "visual": {"kind": "sort_race", "params": {"title": "Bubble sort: swap neighbours", "source": src, "array": s["array"]}}},
            {"id": "merge", "on_screen_text": "Merge sort", "in_short": True, "facts": ["merge_def", "merge_big_o", "von_neumann"],
             "visual": {"kind": "sort_race", "params": {"title": "Merge sort: split, sort, merge", "source": src, "array": s["array"]}}},
            {"id": "scale", "on_screen_text": "Why it matters at scale", "in_short": False, "facts": ["sim_1000"],
             "visual": {"kind": "big_o", "params": {"title": "Comparisons as n grows", "source": "n²/2 vs n·log₂n"}}},
            {"id": "recap", "on_screen_text": "Recap", "in_short": False, "facts": [],
             "visual": {"kind": "bullet_card", "params": {"title": "Takeaways", "lines": [
                 "Bubble sort: simple, quadratic", "Merge sort: divide and conquer", "Count comparisons, not seconds"]}}},
        ]
        return facts, sb, s

    def _research_rag(self):
        r = sims.rag_summary()
        facts = [Fact(id="sim_retrieval", source_id="simulation",
                      text=f"For the query '{r['query']}', the '{r['top_title']}' document scored {r['top_score']:.2f}; "
                           f"the next best scored {r['second_score']:.2f}.")]
        src = "toy corpus of 5 documents, character n-gram TF-IDF"
        sb = [
            {"id": "hook", "on_screen_text": "Give the model an open book", "in_short": True, "facts": ["rag_def", "rag_paper"],
             "visual": {"kind": "rag_pipeline", "params": {"title": "Retrieval-augmented generation", "source": "pipeline overview",
                                                           "stages": [["Question", "user asks"], ["Embed", "text → vector"],
                                                                      ["Search", "nearest chunks"], ["Prompt", "question + chunks"],
                                                                      ["Generate", "LLM answers"], ["Cite", "show sources"]]}}},
            {"id": "retrieve", "on_screen_text": "Step 1: retrieve", "in_short": True, "hero": True, "facts": ["sim_retrieval"],
             "visual": {"kind": "retrieval_scores", "params": {"title": "Which document matches?", "source": src,
                                                               "query": r["query"], "docs": r["docs"], "scores": r["scores"]}}},
            {"id": "why", "on_screen_text": "Why bother?", "in_short": True, "facts": ["rag_why"],
             "visual": {"kind": "bullet_card", "params": {"title": "Why RAG", "lines": [
                 "Fresh or private knowledge without retraining", "Answers can point to their sources",
                 "Quality depends on retrieval quality"]}}},
        ]
        return facts, sb, r

    # ------------------------------------------------------------------ offline writer
    def fallback_script(self, research: ResearchPack, ctx: RunContext) -> Script:
        key = research.data["concept"]
        return getattr(self, f"_script_{key}")(research, ctx)

    def _segments(self, research, lines: dict[str, list[tuple[str, str | None]]]):
        segs = []
        for beat in research.data["storyboard"]:
            b = Beat(beat)
            for text, src in lines[beat["id"]]:
                b.say(text, src)
            segs.append(b.segment())
        return segs

    def _script_gradient_descent(self, research, ctx) -> Script:
        steps = research.data["sim"]["steps_good"]
        lines = {
            "hook": [("When a neural network learns, it is mostly doing one simple thing: walking downhill.", None),
                     (f"This ball is running gradient descent, and in {steps} steps it reaches the bottom of the valley.", "simulation"),
                     ("Let's see how it knows which way is down.", None)],
            "slope": [("Picture the loss as a curve over one parameter.", None),
                      ("At any point, the slope tells you which way is uphill.", None),
                      ("Gradient descent reduces the loss by stepping in the direction of the negative gradient.", "dlbook"),
                      ("Steep slope, big step. Flat slope, small step.", None)],
            "learning_rate": [("How far each step goes is the learning rate.", None),
                              ("At 0.02, the loss is still about 0.02 after 80 steps.", "simulation"),
                              (f"At 0.08, the loss falls below 0.001 after {steps} steps.", "simulation"),
                              ("At 0.17, each step overshoots, and the loss grows to over 8,000.", "simulation"),
                              ("On this surface, any learning rate above about 0.167 diverges.", "simulation")],
            "stochastic": [("Real models train on huge datasets, so computing the exact gradient on all of it is slow.", None),
                           ("Stochastic gradient descent estimates the gradient from a random subset of the data instead.", "wiki_sgd"),
                           ("The path gets noisy, but each step is far cheaper, and it still heads downhill.", None)],
            "history": [("The idea is old: Augustin-Louis Cauchy first suggested gradient descent in 1847.", "wiki_gd"),
                        ("Measure the slope, step the other way, choose a sensible step size, repeat.", None),
                        (f"That's gradient descent. I'm {ctx.brand}, see you in the next one.", None)],
        }
        return Script(format=self.name, topic=research.topic, title="Gradient Descent, Animated: Why the Learning Rate Matters",
                      thumbnail_text="Walk downhill", summary="An animated, simulation-backed look at gradient descent: "
                      "slopes, learning rates that crawl, converge or explode, and why real training is noisy.",
                      tags=["gradient descent", "machine learning", "deep learning", "learning rate", "optimization",
                            "python", "animation"], segments=self._segments(research, lines))

    def _script_sorting(self, research, ctx) -> Script:
        s = research.data["sim"]
        lines = {
            "hook": [(f"On the same {s['n']} shuffled bars, bubble sort made {s['bubble_comparisons']} comparisons "
                      f"and merge sort made {s['merge_comparisons']}.", "simulation"),
                     ("Same job, very different amounts of work. Here's why.", None)],
            "bubble": [("Bubble sort repeatedly swaps neighbours that are in the wrong order.", "wiki_bubble"),
                       ("Big values bubble to the end, one pass at a time.", None),
                       ("Its worst-case and average cost is O(n^2).", "wiki_bubble")],
            "merge": [("Merge sort is divide and conquer: split the list, sort the halves, merge them.", "wiki_merge"),
                      ("It runs in O(n log n) time in the worst case.", "wiki_merge"),
                      ("It was invented by John von Neumann in 1945.", "wiki_merge")],
            "scale": [("The gap explodes as lists grow.", None),
                      ("For 1,000 items, n squared over 2 is 499,500 comparisons, while n log n is about 9,966.", "simulation")],
            "recap": [("Simple can be slow, and splitting a problem in half is a superpower.", None),
                      (f"I'm {ctx.brand}. See you in the next one.", None)],
        }
        return Script(format=self.name, topic=research.topic, title="Bubble Sort vs Merge Sort, Side by Side",
                      thumbnail_text="Same bars, less work", summary="Two sorting algorithms race on the same data.",
                      tags=["sorting algorithms", "merge sort", "bubble sort", "big o", "computer science"],
                      segments=self._segments(research, lines))

    def _script_rag(self, research, ctx) -> Script:
        r = research.data["sim"]
        lines = {
            "hook": [("A language model only knows what it saw in training. So give it an open book.", None),
                     ("Retrieval-augmented generation lets a model retrieve relevant documents and use them as context "
                      "when it generates an answer.", "wiki_rag"),
                     ("It was introduced by Lewis and colleagues in a 2020 paper.", "lewis2020")],
            "retrieve": [("Step one is retrieval: turn the question and the documents into vectors, and compare them.", None),
                         (f"In our toy example, the '{r['top_title']}' document scored {r['top_score']:.2f}; "
                          f"the next best scored {r['second_score']:.2f}.", "simulation"),
                         ("That top chunk goes into the prompt next to the question.", None)],
            "why": [("Retrieval lets a model use information that was not in its training data, and lets answers cite their sources.", "wiki_rag"),
                    ("But it is only as good as what it retrieves.", None),
                    (f"I'm {ctx.brand}. See you in the next one.", None)],
        }
        return Script(format=self.name, topic=research.topic, title="How RAG Works in 60 Seconds",
                      thumbnail_text="Open book AI", summary="Retrieval-augmented generation, animated.",
                      tags=["rag", "retrieval augmented generation", "llm", "embeddings", "vector search"],
                      segments=self._segments(research, lines))

    # ------------------------------------------------------------------ assets
    def generate_assets(self, script: Script, research: ResearchPack, ctx: RunContext) -> list[Asset]:
        fps = int(ctx.settings.get("render.fps", 30))
        make_short = ctx.settings.get("render.make_short", True)
        beats = {b["id"]: b for b in research.data["storyboard"]}
        out = []
        for seg in script.segments:
            dur = estimate_seconds(seg.narration)
            variants = [("landscape", (1920, 1080))] + ([("portrait", (1080, 1350))] if make_short and seg.in_short else [])
            for aspect, size in variants:
                params = dict(seg.visual.params, _portrait=aspect == "portrait")
                path = ctx.assets_dir / f"{seg.id}{'_portrait' if aspect == 'portrait' else ''}.mp4"
                render(SCENES[seg.visual.kind](params), path, dur, fps=fps, size=size)
                ctx.log.info("rendered %s %s (%s, %.1fs)", seg.id, aspect, seg.visual.kind, dur)
                out.append(Asset(id=f"anim_{seg.id}_{aspect}", segment_id=seg.id, kind="animation",
                                 path=str(path.relative_to(ctx.run.path)), licence="self-generated",
                                 origin=f"matplotlib {seg.visual.kind} scene, seeded simulation", duration=dur,
                                 meta={"aspect": aspect, "hero": beats[seg.id].get("hero", False) and aspect == "landscape"}))
        return out

    # ------------------------------------------------------------------ format checks
    def checks(self):
        def simulation_reproducible(ctx) -> CheckResult:
            """Re-run the simulation and confirm every 'simulation' fact still matches."""
            if not ctx.research:
                return CheckResult(check="concept:simulation_reproducible", passed=False, details=["no research"])
            fresh = getattr(self, f"_research_{ctx.research.data['concept']}")()[0]
            stored = {f.id: f.text for f in ctx.research.facts if f.source_id == "simulation"}
            bad = [f.id for f in fresh if stored.get(f.id) != f.text]
            return CheckResult(check="concept:simulation_reproducible", passed=not bad,
                               details=[f"simulation fact changed on re-run: {b}" for b in bad])
        return [simulation_reproducible]
