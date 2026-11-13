"""Settings: config/studio.yaml for behaviour, .env for secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # optional, but listed as a core dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def project_root() -> Path:
    env = os.environ.get("AVS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *here.parents]:
        if (parent / "config" / "studio.yaml").exists():
            return parent
    return Path.cwd()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Settings:
    root: Path
    raw: dict[str, Any] = field(default_factory=dict)

    # convenience accessors -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def channel_name(self) -> str:
        return self.get("channel.name", "My Channel")

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def state_dir(self) -> Path:
        return self.root / "data" / "state"

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def llm_provider(self) -> str:
        return os.environ.get("AVS_LLM_PROVIDER") or self.get("llm.provider", "offline")


def load_settings(root: Path | None = None, overrides: dict | None = None) -> Settings:
    root = Path(root) if root else project_root()
    if load_dotenv is not None:
        load_dotenv(root / ".env", override=False)
    cfg_path = root / "config" / "studio.yaml"
    raw: dict = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Settings(root=root, raw=raw)
