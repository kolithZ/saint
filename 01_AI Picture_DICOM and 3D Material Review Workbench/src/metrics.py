"""Metrics implemented locally so the project does not depend on scikit-learn."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def binary_auroc(labels: Iterable[int], scores: Iterable[float]) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    y = np.asarray(list(labels), dtype=np.int64)
    s = np.asarray(list(scores), dtype=np.float64)
    if len(y) != len(s) or not len(y):
        raise ValueError("labels and scores must be non-empty and equally sized")
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires both NORMAL and PNEUMONIA cases")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def binary_metrics(
    labels: Iterable[int], scores: Iterable[float], threshold: float = 0.5
) -> dict[str, float | int | dict[str, int]]:
    """Report AUROC plus the fixed-threshold measures used by this project."""
    y = np.asarray(list(labels), dtype=np.int64)
    s = np.asarray(list(scores), dtype=np.float64)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    prediction = (s >= threshold).astype(np.int64)
    tp = int(np.sum((prediction == 1) & (y == 1)))
    fn = int(np.sum((prediction == 0) & (y == 1)))
    fp = int(np.sum((prediction == 1) & (y == 0)))
    tn = int(np.sum((prediction == 0) & (y == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(y)),
        "auroc": binary_auroc(y, s),
        "sensitivity": sensitivity,
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
