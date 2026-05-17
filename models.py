# -*- coding: utf-8 -*-
"""Classifier definitions used in the radiomics benchmark."""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import SEED


def get_ml_models(random_state: int = SEED) -> dict[str, object]:
    """
    Return probabilistic classifiers with fixed random states.

    XGBoost's class-balance parameter depends on the class counts in each
    training fold and is therefore set by ``train_radiomics._fit_probabilistic_model``.
    """
    return {
        "Logistic Regression": LogisticRegression(
            penalty="l2", max_iter=5000, class_weight="balanced", random_state=random_state
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=random_state, verbosity=0, n_jobs=1
        ),
        "LightGBM": LGBMClassifier(
            random_state=random_state, class_weight="balanced", verbose=-1, n_jobs=1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_state=random_state,
            thread_count=1,
        ),
    }


def get_default_model(random_state: int = SEED) -> CatBoostClassifier:
    """Return the prespecified default model for final training."""
    return get_ml_models(random_state)["CatBoost"]
