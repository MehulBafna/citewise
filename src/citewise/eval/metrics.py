"""Accuracy and macro-F1 (the standard PubMedQA metrics), with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

LABELS = ("yes", "no", "maybe")
Metric = Callable[[Sequence[str], Sequence[str]], float]


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    return float(np.mean([t == p for t, p in zip(y_true, y_pred, strict=True)]))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    scores = []
    for label in LABELS:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, int]]:
    table = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred, strict=True):
        table[t][p] += 1
    return table


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Metric,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval of the metric over resampled items."""
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    t, p = np.asarray(y_true), np.asarray(y_pred)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        values.append(metric(list(t[idx]), list(p[idx])))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_difference(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: Metric,
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """metric(a) - metric(b) on the same items, with a paired bootstrap 95% interval."""
    n = len(y_true)
    if n == 0:
        return {"difference": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    t, a, b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    diffs = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        diffs.append(metric(list(t[idx]), list(a[idx])) - metric(list(t[idx]), list(b[idx])))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "difference": metric(list(t), list(a)) - metric(list(t), list(b)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }
