"""Dataset loaders. Each returns a tidy DataFrame with columns
entity, year, value [, group, weight, x] plus the path it was read from."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "ai-video-studio/0.1 (+data_story format)"}

# OWID entities that are aggregates, not countries
OWID_AGGREGATE_HINTS = ("(", "World", "income", "Europe", "Asia", "Africa", "America", "Oceania", "Union")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cache_dir(root: Path) -> Path:
    d = root / "data" / "datasets" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_bundled(spec: dict, root: Path) -> tuple[pd.DataFrame, Path]:
    path = root / "data" / "datasets" / spec["file"]
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "entity": df[spec.get("entity", "entity")],
        "year": df[spec.get("year", "year")].astype(int),
        "value": df[spec["metric"]].astype(float),
    })
    for col, key in (("group", "group"), ("weight", "weight"), ("x", "x_metric")):
        if spec.get(key):
            out[col] = df[spec[key]]
    return out, path


def load_owid(spec: dict, root: Path) -> tuple[pd.DataFrame, Path]:
    cache = _cache_dir(root) / f"owid-{spec['slug']}.csv"
    url = f"https://ourworldindata.org/grapher/{spec['slug']}.csv?v=1&csvType=full&useColumnShortNames=true"
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        cache.write_bytes(r.content)
    except Exception:
        if not cache.exists():
            raise
    df = pd.read_csv(cache)
    value_col = [c for c in df.columns if c not in ("Entity", "Code", "Year")][0]
    df = df[df["Code"].notna() & ~df["Code"].astype(str).str.startswith("OWID")]
    return pd.DataFrame({"entity": df["Entity"], "year": df["Year"].astype(int),
                         "value": df[value_col].astype(float)}).dropna(), cache


def load_worldbank(spec: dict, root: Path) -> tuple[pd.DataFrame, Path]:
    cache = _cache_dir(root) / f"wb-{spec['indicator']}.json"
    url = (f"https://api.worldbank.org/v2/country/all/indicator/{spec['indicator']}"
           f"?format=json&per_page=20000&date=1960:2030")
    try:
        r = requests.get(url, headers=UA, timeout=30)
        r.raise_for_status()
        cache.write_text(r.text)
    except Exception:
        if not cache.exists():
            raise
    rows = json.loads(cache.read_text())[1]
    df = pd.DataFrame([{"entity": x["country"]["value"], "iso3": x["countryiso3code"], "year": int(x["date"]),
                        "value": x["value"]} for x in rows]).dropna(subset=["value"])
    df = df[df["iso3"].str.len() == 3]  # drop aggregates without ISO3 codes
    return df[["entity", "year", "value"]], cache


def load_kaggle(spec: dict, root: Path) -> tuple[pd.DataFrame, Path]:
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise RuntimeError("Kaggle loader needs KAGGLE_USERNAME and KAGGLE_KEY in .env")
    dest = _cache_dir(root) / f"kaggle-{spec['ref'].replace('/', '__')}"
    if not (dest / spec["file"]).exists():
        subprocess.run(["kaggle", "datasets", "download", "-d", spec["ref"], "-f", spec["file"], "-p", str(dest), "--unzip"],
                       check=True)
    bundled = dict(spec, file=str((dest / spec["file"]).relative_to(root / "data" / "datasets")))
    return load_bundled(bundled, root)


LOADERS = {"bundled": load_bundled, "owid": load_owid, "worldbank": load_worldbank, "kaggle": load_kaggle}


def load_dataset(spec: dict, root: Path) -> tuple[pd.DataFrame, Path]:
    return LOADERS[spec["loader"]](spec, root)
