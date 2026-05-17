# -*- coding: utf-8 -*-
"""DeLong test for paired ROC AUC comparisons."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def _fast_delong(predictions_sorted: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    n_models, n_examples = predictions_sorted.shape
    n_neg = n_examples - n_pos
    pos = predictions_sorted[:, :n_pos]
    neg = predictions_sorted[:, n_pos:]
    tx = np.empty((n_models, n_pos), dtype=float)
    ty = np.empty((n_models, n_neg), dtype=float)
    tz = np.empty((n_models, n_examples), dtype=float)
    for r in range(n_models):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(predictions_sorted[r])
    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    covariance = np.atleast_2d(np.cov(v01) / n_pos + np.cov(v10) / n_neg)
    return aucs, covariance


def delong_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict[str, float]:
    """Compare two correlated AUCs from the same cases using DeLong's test."""
    y_true = np.asarray(y_true, dtype=int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if len({len(y_true), len(p1), len(p2)}) != 1:
        raise ValueError("y_true, p1, and p2 must have the same length.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("DeLong test requires both positive and negative cases.")
    if np.sum(y_true == 1) < 2 or np.sum(y_true == 0) < 2:
        raise ValueError("DeLong test requires at least two positive and two negative cases.")
    order = np.argsort(-y_true)
    n_pos = int(np.sum(y_true == 1))
    aucs, covariance = _fast_delong(np.vstack([p1, p2])[:, order], n_pos)
    contrast = np.array([[1.0, -1.0]])
    variance = float((contrast @ covariance @ contrast.T).item())
    z = 0.0 if variance <= 0 else float((aucs[0] - aucs[1]) / np.sqrt(variance))
    p_value = 1.0 if variance <= 0 else float(2.0 * stats.norm.sf(abs(z)))
    return {
        "auc1": float(roc_auc_score(y_true, p1)),
        "auc2": float(roc_auc_score(y_true, p2)),
        "diff": float(aucs[0] - aucs[1]),
        "z_stat": z,
        "p_value": p_value,
    }


def delong_pairwise(y_true: np.ndarray, probs_dict: dict[str, np.ndarray]) -> dict[tuple[str, str], dict[str, float]]:
    names = list(probs_dict)
    return {
        (name1, name2): delong_test(y_true, probs_dict[name1], probs_dict[name2])
        for i, name1 in enumerate(names)
        for name2 in names[i + 1 :]
    }


def format_delong_table(pairwise_results: dict[tuple[str, str], dict[str, float]]) -> str:
    lines = ["Pairwise DeLong Test Results", "-" * 50]
    for (name1, name2), res in pairwise_results.items():
        lines.append(
            f"{name1} vs {name2}: AUC={res['auc1']:.4f} vs {res['auc2']:.4f}, "
            f"diff={res['diff']:.4f}, p={res['p_value']:.6f}"
        )
    return "\n".join(lines)
