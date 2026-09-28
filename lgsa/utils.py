"""Seeding, device selection, metrics and result I/O."""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

__all__ = [
    "set_seed",
    "resolve_device",
    "evaluate",
    "aggregate",
    "write_csv",
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def evaluate(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Target-side accuracy, micro-F1 and macro-F1."""
    prediction = logits.argmax(dim=1).cpu().numpy()
    truth = labels.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "micro_f1": float(f1_score(truth, prediction, average="micro")),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def aggregate(values: Sequence[float]) -> Dict[str, float]:
    """Mean and (population) standard deviation of a list of scores."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(array.mean()), "std": float(array.std()), "n": int(array.size)}


def write_csv(path: str, rows: Iterable[Dict], fieldnames: List[str] | None = None) -> str:
    """Write a list of dicts to CSV (UTF-8 BOM so Excel opens it correctly)."""
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = fieldnames or list(rows[0].keys())
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
