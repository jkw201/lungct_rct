# -*- coding: utf-8 -*-
"""ROC/PR metrics and plotting utilities."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

from config import COLORS, SEED


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func,
    *,
    n_bootstraps: int = 1000,
    seed: int = SEED,
) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_func(y_true[idx], y_prob[idx]))
    if not scores:
        return float("nan"), float("nan")
    return tuple(np.percentile(scores, [2.5, 97.5]))


def compute_metrics_with_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "Model",
    n_bootstraps: int = 1000,
    threshold: float = 0.5,
    seed: int = SEED,
) -> dict[str, object]:
    """Compute metrics using a prespecified threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc_value = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    auc_low, auc_high = bootstrap_ci(
        y_true, y_prob, roc_auc_score, n_bootstraps=n_bootstraps, seed=seed
    )
    return {
        "Model": name,
        "Threshold": threshold,
        "AUC": auc_value,
        "AUC_CI_Low": auc_low,
        "AUC_CI_High": auc_high,
        "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "PPV": precision_score(y_true, y_pred, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) else 0.0,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def plot_roc_with_ci(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    n_bootstraps: int = 1000,
    output_path: str | None = None,
    figsize: tuple = (7, 6),
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rng = np.random.RandomState(seed)
    mean_fpr = np.linspace(0, 1, 100)
    results = {}
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC plotting requires both positive and negative cases.")
    for name, y_prob in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        boot_tprs, boot_aucs = [], []
        for _ in range(n_bootstraps):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            bfpr, btpr, _ = roc_curve(y_true[idx], y_prob[idx])
            interp_tpr = np.interp(mean_fpr, bfpr, btpr)
            interp_tpr[0] = 0.0
            boot_tprs.append(interp_tpr)
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
        if boot_aucs:
            ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
        else:
            ci_low, ci_high = float("nan"), float("nan")
        results[name] = {"auc": auc_value, "ci_low": ci_low, "ci_high": ci_high}
        color = COLORS.get(name, {"main": "gray", "light": "lightgray"})
        ax.plot(fpr, tpr, color=color["main"], lw=2.5, label=f"{name}: AUC={auc_value:.3f}")
        if boot_tprs:
            arr = np.asarray(boot_tprs)
            ax.fill_between(mean_fpr, np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0), color=color["light"], alpha=0.45)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Reference")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return results


def plot_pr_with_ci(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    n_bootstraps: int = 1000,
    output_path: str | None = None,
    figsize: tuple = (7, 6),
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    rng = np.random.RandomState(seed)
    mean_recall = np.linspace(0, 1, 100)
    results = {}
    if len(np.unique(y_true)) < 2:
        raise ValueError("PR plotting requires both positive and negative cases.")
    for name, y_prob in probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)
        boot_precisions, boot_aps = [], []
        for _ in range(n_bootstraps):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            p, r, _ = precision_recall_curve(y_true[idx], y_prob[idx])
            boot_precisions.append(np.interp(mean_recall, r[::-1], p[::-1]))
            boot_aps.append(average_precision_score(y_true[idx], y_prob[idx]))
        if boot_aps:
            ci_low, ci_high = np.percentile(boot_aps, [2.5, 97.5])
        else:
            ci_low, ci_high = float("nan"), float("nan")
        results[name] = {"ap": ap_value, "ci_low": ci_low, "ci_high": ci_high}
        color = COLORS.get(name, {"main": "gray", "light": "lightgray"})
        ax.plot(recall, precision, color=color["main"], lw=2.5, label=f"{name}: AP={ap_value:.3f}")
        if boot_precisions:
            arr = np.asarray(boot_precisions)
            ax.fill_between(mean_recall, np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0), color=color["light"], alpha=0.45)
    ax.axhline(np.mean(y_true), color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return results
