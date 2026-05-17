# -*- coding: utf-8 -*-
"""Calibration curve plotting with percentile bootstrap intervals."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from config import COLORS, SEED


def plot_calibration(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    n_bins: int = 10,
    n_bootstraps: int = 1000,
    output_path: str | None = None,
    show_ci: bool = True,
    figsize: tuple = (7, 6),
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    """Plot calibration curves using percentile bootstrap CI bands."""
    y_true = np.asarray(y_true, dtype=int)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rng = np.random.RandomState(seed)
    metrics = {}

    for name, y_prob in probs_dict.items():
        y_prob = np.asarray(y_prob, dtype=float)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        brier = float(brier_score_loss(y_true, y_prob))
        ece = float(np.mean(np.abs(prob_true - prob_pred)))
        metrics[name] = {"brier": brier, "ece": ece}

        color = COLORS.get(name, {"main": "gray", "light": "lightgray"})
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            markersize=6,
            label=f"{name} (Brier={brier:.3f}, ECE={ece:.3f})",
            color=color["main"],
            lw=2,
        )

        if show_ci and n_bootstraps > 0:
            calib_curves = []
            for _ in range(n_bootstraps):
                idx = rng.randint(0, len(y_prob), len(y_prob))
                if len(np.unique(y_true[idx])) < 2:
                    continue
                try:
                    pt, pp = calibration_curve(
                        y_true[idx], y_prob[idx], n_bins=n_bins, strategy="uniform"
                    )
                except ValueError:
                    continue
                if len(pp) > 1:
                    calib_curves.append(np.interp(prob_pred, pp, pt))
            if calib_curves:
                arr = np.asarray(calib_curves)
                lower = np.percentile(arr, 2.5, axis=0)
                upper = np.percentile(arr, 97.5, axis=0)
                ax.fill_between(prob_pred, lower, upper, color=color["light"], alpha=0.5)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return metrics
