"""
Training dataset and embedding metadata alignment utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set

import pandas as pd

from .id_mapping import normalize_poi_id


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _extract_poi_ids(obj: Any, out: Set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lk = str(key).lower()
            if lk in {"poi_id", "target_next_poi", "next_poi"} and value is not None:
                out.add(normalize_poi_id(value))
            if lk in {"route_poi_ids", "state_prefix", "pois", "route"} and isinstance(value, list):
                for item in value:
                    if isinstance(item, (str, int, float)):
                        out.add(normalize_poi_id(item))
                    elif isinstance(item, dict) and item.get("poi_id") is not None:
                        out.add(normalize_poi_id(item["poi_id"]))
            _extract_poi_ids(value, out)
    elif isinstance(obj, list):
        for item in obj:
            _extract_poi_ids(item, out)


def collect_training_poi_ids(dataset_dir: str | Path = "outputs/datasets") -> Set[str]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return set()

    poi_ids: Set[str] = set()
    for file_path in dataset_path.glob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()

        if suffix == ".jsonl":
            for record in _iter_jsonl(file_path):
                _extract_poi_ids(record, poi_ids)
        elif suffix == ".csv":
            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue
            for col in df.columns:
                col_l = str(col).lower()
                if "poi" in col_l and "id" in col_l:
                    vals = df[col].dropna().astype(str)
                    poi_ids.update(normalize_poi_id(v) for v in vals.tolist())

    return poi_ids


def summarize_dataset_embedding_alignment(
    dataset_dir: str | Path = "outputs/datasets",
    embedding_meta_csv: str | Path = "outputs/emb/poi_meta.csv",
) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir)
    meta_path = Path(embedding_meta_csv)

    report: Dict[str, Any] = {
        "checked": False,
        "dataset_dir": str(dataset_path),
        "embedding_meta_csv": str(meta_path),
        "dataset_unique_poi_ids": 0,
        "embedding_unique_poi_ids": 0,
        "missing_in_embeddings": 0,
        "missing_examples": [],
    }

    if not dataset_path.exists() or not meta_path.exists():
        return report

    training_ids = collect_training_poi_ids(dataset_path)
    if not training_ids:
        report["checked"] = True
        return report

    try:
        meta_df = pd.read_csv(meta_path, usecols=["poi_id"])
    except Exception:
        return report

    embedding_ids = set(meta_df["poi_id"].astype(str).map(normalize_poi_id))
    missing = sorted(training_ids - embedding_ids)

    report.update({
        "checked": True,
        "dataset_unique_poi_ids": len(training_ids),
        "embedding_unique_poi_ids": len(embedding_ids),
        "missing_in_embeddings": len(missing),
        "missing_examples": missing[:10],
    })
    return report

