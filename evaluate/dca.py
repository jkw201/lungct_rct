# -*- coding: utf-8 -*-
"""Decision Curve Analysis with percentile bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import COLORS, SEED


def _threshold_weights(thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=float)
    clipped = np.clip(thresholds, 1e-6, 1 - 1e-6)
    return clipped / (1.0 - clipped)


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(int)
    thresholds = np.clip(np.asarray(thresholds, dtype=float), 1e-6, 1 - 1e-6)
    weights = _threshold_weights(thresholds)
    n = len(y_true)
    nb = np.zeros_like(thresholds, dtype=float)
    for i, threshold in enumerate(thresholds):
        pred_pos = (y_prob >= threshold).astype(int)
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        nb[i] = (tp / n) - (fp / n) * weights[i]
    return nb


def treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prev = np.mean(y_true.astype(int))
    return prev - (1.0 - prev) * _threshold_weights(thresholds)


def treat_none_net_benefit(thresholds: np.ndarray) -> np.ndarray:
    return np.zeros_like(thresholds, dtype=float)


def plot_dca(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    threshold_range: tuple = (0.01, 0.80, 0.01),
    n_bootstraps: int = 1000,
    output_path: str | None = None,
    show_ci: bool = True,
    figsize: tuple = (7, 6),
    seed: int = SEED,
) -> pd.DataFrame:
    thresholds = np.arange(*threshold_range)
    thresholds = np.clip(thresholds, 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=int)
    rng = np.random.RandomState(seed)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    nb_all = treat_all_net_benefit(y_true, thresholds)
    ax.axhline(y=0, color="gray", lw=1, alpha=0.5, linestyle="-", label="Treat None")
    ax.plot(thresholds, nb_all, "k--", lw=1, alpha=0.5, label="Treat All")

    all_nb = {}
    for name, y_prob in probs_dict.items():
        y_prob = np.asarray(y_prob, dtype=float)
        nb_base = net_benefit(y_true, y_prob, thresholds)
        all_nb[name] = nb_base
        color = COLORS.get(name, {"main": "gray", "light": "lightgray"})
        ax.plot(thresholds, nb_base, color=color["main"], lw=2.5, label=name)

        if show_ci and n_bootstraps > 0:
            nb_boot = []
            for _ in range(n_bootstraps):
                idx = rng.randint(0, len(y_prob), len(y_prob))
                nb_boot.append(net_benefit(y_true[idx], y_prob[idx], thresholds))
            arr = np.asarray(nb_boot)
            lower = np.percentile(arr, 2.5, axis=0)
            upper = np.percentile(arr, 97.5, axis=0)
            ax.fill_between(thresholds, lower, upper, color=color["light"], alpha=0.5)

    ax.set_xlim([thresholds.min(), thresholds.max()])
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    records = []
    for idx, threshold in enumerate(thresholds):
        row = {"threshold": threshold, "Treat_None": 0.0, "Treat_All": nb_all[idx]}
        for name, nb_values in all_nb.items():
            row[name] = nb_values[idx]
        records.append(row)
    return pd.DataFrame(records)
