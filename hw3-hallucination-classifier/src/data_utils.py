"""Dataset loading, sampling, and metric helpers."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_classes(data_dir: str | Path) -> list[dict]:
    path = Path(data_dir) / "classes.json"
    with path.open("r", encoding="utf-8") as f:
        classes = json.load(f)
    return sorted(classes, key=lambda item: int(item["id"]))


def load_split(data_dir: str | Path, split: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{split}.csv"
    df = pd.read_csv(path)
    if "id" not in df.columns or "paper_id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"{path} must contain id, paper_id, and text columns")
    return df


def pdf_path_for(data_dir: str | Path, split: str, paper_id: str) -> Path:
    return Path(data_dir) / "paper_evidence" / split / f"{paper_id}.pdf"


def stratified_limit(df: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic stratified subset when labels exist."""
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    if "label" not in df.columns:
        return df.sample(n=max_rows, random_state=seed).sort_values("id").reset_index(drop=True)

    groups = list(df.groupby("label", group_keys=False))
    base = max(1, max_rows // max(1, len(groups)))
    parts = []
    remaining = max_rows
    for _, group in groups:
        take = min(len(group), base, remaining)
        if take > 0:
            parts.append(group.sample(n=take, random_state=seed))
            remaining -= take
    if remaining > 0:
        used = pd.concat(parts).index if parts else []
        rest = df.drop(index=used)
        if len(rest) > 0:
            parts.append(rest.sample(n=min(remaining, len(rest)), random_state=seed))
    return pd.concat(parts).sort_values("id").reset_index(drop=True)


def oversample_records(records: list[dict], factors: dict[int, int], seed: int = 42) -> list[dict]:
    expanded: list[dict] = []
    for record in records:
        label = int(record["label"])
        expanded.extend([record] * int(factors.get(label, 1)))
    rng = random.Random(seed)
    rng.shuffle(expanded)
    return expanded


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_label_id(text: str, default: int = 0) -> int:
    """Parse a generated label robustly without accepting invalid ids."""
    if text is None:
        return default
    match = re.search(r"\b([0-4])\b", str(text))
    if match:
        return int(match.group(1))
    lowered = str(text).lower()
    name_to_id = {
        "attribution": 0,
        "entity": 1,
        "number": 2,
        "overgeneralization": 3,
        "over-generalization": 3,
        "temporal": 4,
    }
    for key, value in name_to_id.items():
        if key in lowered:
            return value
    return default


def classification_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    labels = [0, 1, 2, 3, 4]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "report": classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
