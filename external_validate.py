# -*- coding: utf-8 -*-
"""External validation using a saved training artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from config import SEED
from data_loader import load_feature_table
from train_radiomics import predict_with_artifact


def _bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not scores:
        return float("nan"), float("nan")
    return tuple(np.percentile(scores, [2.5, 97.5]))


def compute_external_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    seed: int = SEED,
    n_boot: int = 1000,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc_low, auc_high = _bootstrap_auc_ci(y_true, y_prob, seed, n_boot)
    return {
        "N": int(len(y_true)),
        "Positive": int(np.sum(y_true == 1)),
        "Negative": int(np.sum(y_true == 0)),
        "Threshold": float(threshold),
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "AUC_CI_Low": float(auc_low),
        "AUC_CI_High": float(auc_high),
        "AveragePrecision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation from a saved artifact")
    parser.add_argument("--artifact", required=True, help="radiomics_final_artifact.joblib")
    parser.add_argument("--data", required=True, help="External feature table")
    parser.add_argument("--output", default="external_results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()

    artifact = joblib.load(args.artifact)
    df = load_feature_table(args.data)
    feature_names = artifact["feature_names"]
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"External table is missing training features: {missing[:10]}")

    X = df[feature_names].to_numpy(dtype=float)
    y_prob = predict_with_artifact(artifact, X)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame({"ID": df["ID"], "Prob": y_prob})
    if "cat" in df.columns:
        pred["Label"] = df["cat"].astype(int).to_numpy()
    pred.to_csv(output / "external_predictions.csv", index=False)

    if "cat" in df.columns:
        metrics = compute_external_metrics(
            df["cat"].to_numpy(dtype=int), y_prob, threshold=args.threshold, seed=args.seed, n_boot=args.n_boot
        )
        pd.DataFrame([metrics]).to_csv(output / "external_metrics.csv", index=False)
    print(f"Saved external validation outputs to: {output}")


if __name__ == "__main__":
    main()
