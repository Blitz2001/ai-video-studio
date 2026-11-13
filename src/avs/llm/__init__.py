"""One interface for every LLM provider. Agents call `complete_json`; which
vendor answers is a config/.env decision."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger("avs.llm")


class LLMUnavailable(RuntimeError):
    pass


class LLM(ABC):
    name = "base"
    is_offline = False

    @abstractmethod
    def _complete(self, prompt: str, system: str | None, json_mode: bool) -> str: ...

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 retries: int = 4) -> str:
        delay = 4.0
        for attempt in range(retries):
            try:
                return self._complete(prompt, system, json_mode)
            except LLMUnavailable:
                raise
            except Exception as exc:  # rate limits on free tiers are routine
                msg = str(exc).lower()
                transient = any(k in msg for k in ("429", "rate", "quota", "overloaded", "503", "timeout", "unavailable"))
                if attempt == retries - 1 or not transient:
                    raise
                log.warning("%s transient error (%s); retrying in %.0fs", self.name, exc, delay)
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    def complete_json(self, prompt: str, system: str | None = None) -> Any:
        text = self.complete(prompt, system, json_mode=True)
        return parse_json(text)

    def embed(self, texts: list[str]):  # pragma: no cover - provider specific
        return None


def parse_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    return json.loads(text[start:])


class OfflineLLM(LLM):
    """No model at all. Agents see `is_offline` and use their deterministic
    template writers, so the whole pipeline runs with zero keys."""

    name = "offline"
    is_offline = True

    def _complete(self, prompt, system, json_mode):
        raise LLMUnavailable("offline provider has no text generation; use the template writer")


class GeminiLLM(LLM):
    name = "gemini"

    def __init__(self, api_key: str, model: str, embed_model: str):
        from google import genai  # imported lazily: optional dependency

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.embed_model = embed_model

    def _complete(self, prompt, system, json_mode):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.6,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
        return resp.text or ""

    def embed(self, texts):
        import numpy as np

        resp = self.client.models.embed_content(model=self.embed_model, contents=texts)
        return np.array([e.values for e in resp.embeddings], dtype=float)


class ClaudeLLM(LLM):
    name = "claude"

    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _complete(self, prompt, system, json_mode):
        if json_mode:
            prompt += "\n\nRespond with a single JSON value and nothing else."
        msg = self.client.messages.create(
            model=self.model, max_tokens=8000, system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAILLM(LLM):
    name = "openai"

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _complete(self, prompt, system, json_mode):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kw)
        return resp.choices[0].message.content or ""

    def embed(self, texts):
        import numpy as np

        resp = self.client.embeddings.create(model="text-embedding-3-small", input=texts)
        return np.array([d.embedding for d in resp.data], dtype=float)


def get_llm(provider: str, fallback_to_offline: bool = True) -> LLM:
    provider = (provider or "offline").lower()
    try:
        if provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMUnavailable("GEMINI_API_KEY not set")
            return GeminiLLM(key, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                             os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004"))
        if provider == "claude":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMUnavailable("ANTHROPIC_API_KEY not set")
            return ClaudeLLM(key, os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"))
        if provider == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise LLMUnavailable("OPENAI_API_KEY not set")
            return OpenAILLM(key, os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
        if provider == "offline":
            return OfflineLLM()
        raise ValueError(f"unknown LLM provider {provider!r}")
    except (LLMUnavailable, ImportError) as exc:
        if not fallback_to_offline:
            raise
        log.warning("LLM provider %s unavailable (%s); using offline template writer", provider, exc)
        return OfflineLLM()
