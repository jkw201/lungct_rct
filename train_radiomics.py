# -*- coding: utf-8 -*-
"""Leakage-aware radiomics model training and nested cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import SEED
from data_loader import load_and_split
from feature_selection import select_features_lasso
from models import get_ml_models


DEFAULT_K_LIST = [5, 8, 10, 12, 18]
INNER_CV_SEED_OFFSET = 100
OUTER_REFIT_SEED_OFFSET = 1000


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC": _safe_auc(y_true, y_prob),
        "AveragePrecision": _safe_ap(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _prepare_xy(
    X_train_raw: np.ndarray,
    X_valid_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(X_train_raw)
    x_valid_imp = imputer.transform(X_valid_raw)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_imp)
    x_valid = scaler.transform(x_valid_imp)
    return x_train, x_valid, imputer, scaler


def _fit_probabilistic_model(model_template, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
    """Fit a classifier, adding fold-specific class balancing where required."""
    model = clone(model_template)
    if model_name == "XGBoost":
        n_pos = max(int(np.sum(y_train == 1)), 1)
        n_neg = max(int(np.sum(y_train == 0)), 1)
        model.set_params(scale_pos_weight=n_neg / n_pos)
    model.fit(X_train, y_train)
    return model


def _fit_pipeline(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    model_name: str,
    k: int,
    seed: int,
) -> dict[str, object]:
    models = get_ml_models(seed)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(models)}")
    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(X_train_raw)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imp)
    selected_names, selected_idx = select_features_lasso(
        x_scaled, y_train, feature_names, cv=5, random_state=seed, method="lasso", top_k=k
    )
    model = _fit_probabilistic_model(
        models[model_name], model_name, x_scaled[:, selected_idx], y_train
    )
    return {
        "model_name": model_name,
        "k": k,
        "seed": seed,
        "feature_names": feature_names,
        "selected_feature_names": selected_names,
        "selected_idx": selected_idx,
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
    }


def predict_with_artifact(artifact: dict[str, object], X: np.ndarray) -> np.ndarray:
    """Apply a saved artifact to new data without fitting anything."""
    x_imp = artifact["imputer"].transform(X)
    x_scaled = artifact["scaler"].transform(x_imp)
    return artifact["model"].predict_proba(x_scaled[:, artifact["selected_idx"]])[:, 1]


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    feature_names: list[str],
    *,
    n_splits: int = 10,
    k_list: list[int] | None = None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive CV grid. Do not use the same grid to select and claim final performance."""
    if k_list is None:
        k_list = DEFAULT_K_LIST
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = get_ml_models(seed)
    prediction_rows = []
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        x_train, x_valid, _, _ = _prepare_xy(X[train_idx], X[valid_idx])
        y_train, y_valid = y[train_idx], y[valid_idx]
        for k in k_list:
            selected_names, selected_idx = select_features_lasso(
                x_train, y_train, feature_names, cv=5, random_state=seed, method="lasso", top_k=k
            )
            for model_name, model_template in models.items():
                model = _fit_probabilistic_model(
                    model_template, model_name, x_train[:, selected_idx], y_train
                )
                y_prob = model.predict_proba(x_valid[:, selected_idx])[:, 1]
                fold_rows.append(
                    {
                        "Model": model_name,
                        "K": k,
                        "Fold": fold,
                        "SelectedFeatures": ";".join(selected_names),
                        **_binary_metrics(y_valid, y_prob),
                    }
                )
                for patient_id, label, prob in zip(ids[valid_idx], y_valid, y_prob):
                    prediction_rows.append(
                        {
                            "ID": patient_id,
                            "Label": int(label),
                            "Prob": float(prob),
                            "Fold": fold,
                            "Model": model_name,
                            "K": k,
                        }
                    )
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def summarize_cv(fold_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, k), g in pred_df.groupby(["Model", "K"]):
        y_true = g["Label"].to_numpy(dtype=int)
        y_prob = g["Prob"].to_numpy(dtype=float)
        fold_subset = fold_df[(fold_df["Model"] == model) & (fold_df["K"] == k)]
        rows.append(
            {
                "Model": model,
                "K": k,
                "OOF_AUC": _safe_auc(y_true, y_prob),
                "OOF_AveragePrecision": _safe_ap(y_true, y_prob),
                "MeanFoldAUC": float(fold_subset["AUC"].mean()),
                "StdFoldAUC": float(fold_subset["AUC"].std(ddof=1)),
                "MeanFoldSensitivity": float(fold_subset["Sensitivity"].mean()),
                "MeanFoldSpecificity": float(fold_subset["Specificity"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["OOF_AUC", "OOF_AveragePrecision", "MeanFoldAUC", "K", "Model"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )


def run_nested_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    feature_names: list[str],
    *,
    outer_splits: int = 5,
    inner_splits: int = 5,
    k_list: list[int] | None = None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nested CV for unbiased internal performance after model/K selection.

    The inner loop selects Model and K on the outer-training data. The selected
    pipeline is then refit on the full outer-training fold and evaluated once
    on the untouched outer-test fold.
    """
    if k_list is None:
        k_list = DEFAULT_K_LIST
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_rows = []
    pred_rows = []
    selection_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        inner_fold_df, inner_pred_df = run_cross_validation(
            X[train_idx],
            y[train_idx],
            ids[train_idx],
            feature_names,
            n_splits=inner_splits,
            k_list=k_list,
            seed=seed + INNER_CV_SEED_OFFSET + outer_fold,
        )
        inner_summary = summarize_cv(inner_fold_df, inner_pred_df)
        chosen = inner_summary.iloc[0]
        model_name = str(chosen["Model"])
        k = int(chosen["K"])
        selection_rows.append(
            {
                "OuterFold": outer_fold,
                "SelectedModel": model_name,
                "SelectedK": k,
                "InnerOOF_AUC": float(chosen["OOF_AUC"]),
                "InnerOOF_AveragePrecision": float(chosen["OOF_AveragePrecision"]),
            }
        )
        artifact = _fit_pipeline(
            X[train_idx],
            y[train_idx],
            feature_names,
            model_name=model_name,
            k=k,
            seed=seed + OUTER_REFIT_SEED_OFFSET + outer_fold,
        )
        y_prob = predict_with_artifact(artifact, X[test_idx])
        metrics = _binary_metrics(y[test_idx], y_prob)
        outer_rows.append({"OuterFold": outer_fold, "Model": model_name, "K": k, **metrics})
        for patient_id, label, prob in zip(ids[test_idx], y[test_idx], y_prob):
            pred_rows.append(
                {
                    "ID": patient_id,
                    "Label": int(label),
                    "Prob": float(prob),
                    "OuterFold": outer_fold,
                    "SelectedModel": model_name,
                    "SelectedK": k,
                }
            )
    return pd.DataFrame(outer_rows), pd.DataFrame(pred_rows), pd.DataFrame(selection_rows)


def fit_final_artifact(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    model_name: str,
    k: int,
    seed: int = SEED,
) -> dict[str, object]:
    return _fit_pipeline(X, y, feature_names, model_name=model_name, k=k, seed=seed)


def _write_artifact_summary(artifact: dict[str, object], path: Path) -> None:
    """Write a JSON summary. The complete fitted objects are stored in joblib."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "note": "Summary only; complete fitted imputer/scaler/model are in the .joblib artifact.",
                "model_name": artifact["model_name"],
                "k": artifact["k"],
                "seed": artifact["seed"],
                "feature_names": artifact["feature_names"],
                "selected_feature_names": artifact["selected_feature_names"],
                "imputer": "SimpleImputer(strategy='median')",
                "scaler": "StandardScaler()",
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-aware radiomics training")
    parser.add_argument("--data", required=True, help="Input labelled CSV/XLSX table")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--nested_outer_splits", type=int, default=5)
    parser.add_argument("--nested_inner_splits", type=int, default=5)
    parser.add_argument("--skip_nested", action="store_true")
    parser.add_argument("--k_list", type=int, nargs="+", default=DEFAULT_K_LIST)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--final_model", default="CatBoost")
    parser.add_argument("--final_k", type=int, default=10)
    parser.add_argument("--no_final_artifact", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    X, y, ids, feature_names, _ = load_and_split(args.data)

    fold_df, pred_df = run_cross_validation(
        X, y, ids, feature_names, n_splits=args.n_splits, k_list=args.k_list, seed=args.seed
    )
    summary_df = summarize_cv(fold_df, pred_df)
    fold_df.to_csv(output / "cv_fold_metrics_descriptive.csv", index=False)
    pred_df.to_csv(output / "cv_oof_predictions_descriptive_long.csv", index=False)
    summary_df.to_csv(output / "cv_summary_descriptive.csv", index=False)

    if not args.skip_nested:
        outer_df, nested_pred_df, selection_df = run_nested_cross_validation(
            X,
            y,
            ids,
            feature_names,
            outer_splits=args.nested_outer_splits,
            inner_splits=args.nested_inner_splits,
            k_list=args.k_list,
            seed=args.seed,
        )
        outer_df.to_csv(output / "nested_outer_fold_metrics.csv", index=False)
        nested_pred_df.to_csv(output / "nested_oof_predictions.csv", index=False)
        selection_df.to_csv(output / "nested_inner_selections.csv", index=False)

    if not args.no_final_artifact:
        artifact = fit_final_artifact(
            X, y, feature_names, model_name=args.final_model, k=args.final_k, seed=args.seed
        )
        joblib.dump(artifact, output / "radiomics_final_artifact.joblib")
        _write_artifact_summary(artifact, output / "radiomics_final_artifact_summary.json")
    print(f"Saved outputs to: {output}")


if __name__ == "__main__":
    main()
