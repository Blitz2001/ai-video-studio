"""Format discovery. Every sub-folder of this package that exposes `PLUGIN` is a
format, as is every folder listed in AVS_EXTRA_FORMATS (os.pathsep separated,
each pointing at either a single format folder or a folder of formats)."""
from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
import sys
from pathlib import Path

from .base import FormatPlugin, RunContext

__all__ = ["FormatPlugin", "RunContext", "discover", "get_format"]


def _load_from_folder(folder: Path) -> FormatPlugin | None:
    init = folder / "__init__.py"
    if not init.exists():
        return None
    mod_name = f"avs_ext_formats.{folder.name}"
    spec = importlib.util.spec_from_file_location(mod_name, init, submodule_search_locations=[str(folder)])
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return getattr(module, "PLUGIN", None)


def discover() -> dict[str, FormatPlugin]:
    found: dict[str, FormatPlugin] = {}
    pkg_dir = Path(__file__).parent
    for info in pkgutil.iter_modules([str(pkg_dir)]):
        if not info.ispkg:
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        plugin = getattr(module, "PLUGIN", None)
        if isinstance(plugin, FormatPlugin):
            found[plugin.name] = plugin
    for entry in filter(None, os.environ.get("AVS_EXTRA_FORMATS", "").split(os.pathsep)):
        root = Path(entry)
        candidates = [root] if (root / "__init__.py").exists() else [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
        for folder in candidates:
            plugin = _load_from_folder(folder)
            if isinstance(plugin, FormatPlugin):
                found[plugin.name] = plugin
    return found


def get_format(name: str) -> FormatPlugin:
    formats = discover()
    if name not in formats:
        raise KeyError(f"unknown format {name!r}; available: {', '.join(sorted(formats))}")
    return formats[name]
