# -*- coding: utf-8 -*-
"""Feature selection and fold-level preprocessing helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import SEED


def select_features_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    cv: int = 5,
    random_state: int = SEED,
    method: str = "lasso",
    top_k: Optional[int] = None,
) -> tuple[list[str], list[int]]:
    """Select features from a preprocessed training matrix only."""
    if method == "lasso":
        selector = LassoCV(cv=cv, random_state=random_state, n_jobs=1, max_iter=10000)
        selector.fit(X_train, y_train)
        coefs = np.abs(selector.coef_)
    elif method == "l1_lr":
        selector = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            random_state=random_state,
            class_weight="balanced",
        )
        selector.fit(X_train, y_train)
        coefs = np.abs(selector.coef_[0])
    else:
        raise ValueError(f"Unknown method: {method}")

    rank_indices = np.argsort(coefs)[::-1]
    if top_k is not None:
        selected_idx = sorted(rank_indices[:top_k].tolist())
    else:
        selected_idx = sorted([i for i, coef in enumerate(coefs) if coef > 1e-6])
    if not selected_idx:
        selected_idx = [int(rank_indices[0])]
    return [feature_names[i] for i in selected_idx], selected_idx


def preprocess_fold(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    """Median-impute and standardize with parameters fitted on train only."""
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc = scaler.transform(X_val_imp)
    return X_train_sc, X_val_sc, imputer, scaler


def get_lasso_ranking_global(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[str]:
    """Global ranking for visualization only; not for performance estimation."""
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    lasso = LassoCV(cv=10, random_state=SEED, n_jobs=1, max_iter=10000)
    lasso.fit(X_sc, y)
    rank_idx = np.argsort(np.abs(lasso.coef_))[::-1]
    return [feature_names[i] for i in rank_idx]
