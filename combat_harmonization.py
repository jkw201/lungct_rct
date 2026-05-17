# -*- coding: utf-8 -*-
"""Leakage-aware batch harmonization utilities for sensitivity analyses.

This module supports covariates by first modeling their training-set feature
effects, harmonizing residuals by batch, and adding the covariate contribution
back. Outcome labels such as ``cat`` can preserve disease-associated signal in
an explanatory harmonization analysis, but they must not be used to transform
validation or external data for predictive performance estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


OUTCOME_COVARIATE_NAMES = {"cat", "Cat", "label", "Label", "target", "Target", "outcome"}


@dataclass
class LocationScaleHarmonizer:
    """Training-set-only location/scale batch harmonizer with optional covariates."""

    batch_col: str
    feature_cols: list[str]
    covar_cols: list[str] | None = None
    eps: float = 1e-8
    allow_outcome_covariates: bool = False
    covariate_columns_: list[str] = field(default_factory=list, init=False)
    numeric_covar_medians_: dict[str, float] = field(default_factory=dict, init=False)

    def _check_columns(self, df: pd.DataFrame) -> None:
        required = [self.batch_col, *self.feature_cols, *(self.covar_cols or [])]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for harmonization: {missing}")

    def _warn_if_outcome_covariates(self) -> None:
        covars = set(self.covar_cols or [])
        if covars & OUTCOME_COVARIATE_NAMES and not self.allow_outcome_covariates:
            raise ValueError(
                "Outcome covariates such as 'cat' may leak labels during validation/external "
                "prediction. Set allow_outcome_covariates=True only for labelled, explanatory "
                "sensitivity analyses, not for predictive performance estimation."
            )

    def _make_covariate_design(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        if not self.covar_cols:
            return np.ones((len(df), 1), dtype=float)
        covar_frames = []
        for col in self.covar_cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                if fit:
                    median = float(series.median()) if series.notna().any() else 0.0
                    self.numeric_covar_medians_[col] = median
                median = self.numeric_covar_medians_.get(col, 0.0)
                covar_frames.append(
                    pd.DataFrame({col: series.astype(float).fillna(median)}, index=df.index)
                )
            else:
                clean = series.astype("string").fillna("__missing__")
                dummies = pd.get_dummies(clean, prefix=col, drop_first=True, dtype=float)
                covar_frames.append(dummies)
        covars = pd.concat(covar_frames, axis=1) if covar_frames else pd.DataFrame(index=df.index)
        if fit:
            self.covariate_columns_ = list(covars.columns)
        else:
            covars = covars.reindex(columns=self.covariate_columns_, fill_value=0.0)
        return np.column_stack([np.ones(len(df), dtype=float), covars.to_numpy(dtype=float)])

    def fit(self, df: pd.DataFrame) -> "LocationScaleHarmonizer":
        """Fit covariate and batch parameters using training data only."""
        self._check_columns(df)
        self._warn_if_outcome_covariates()
        x = df[self.feature_cols].to_numpy(dtype=float)
        design = self._make_covariate_design(df, fit=True)
        self.beta_ = np.linalg.lstsq(design, x, rcond=None)[0]
        residuals = x - design @ self.beta_

        self.pooled_mean_ = np.nanmean(residuals, axis=0)
        self.pooled_std_ = np.nanstd(residuals, axis=0, ddof=1)
        self.pooled_std_ = np.where(self.pooled_std_ < self.eps, 1.0, self.pooled_std_)
        self.batch_params_ = {}
        batch_values = df[self.batch_col].to_numpy()
        for batch in np.unique(batch_values):
            bx = residuals[batch_values == batch]
            mean = np.nanmean(bx, axis=0)
            std = np.nanstd(bx, axis=0, ddof=1)
            std = np.where(std < self.eps, 1.0, std)
            self.batch_params_[batch] = (mean, std)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data with fitted training-set parameters only."""
        if not hasattr(self, "pooled_mean_"):
            raise RuntimeError("Harmonizer must be fitted before transform().")
        self._check_columns(df)
        x = df[self.feature_cols].to_numpy(dtype=float)
        design = self._make_covariate_design(df, fit=False)
        covariate_effect = design @ self.beta_
        residuals = x - covariate_effect
        batch_values = df[self.batch_col].to_numpy()
        transformed_residuals = residuals.copy()
        for batch in np.unique(batch_values):
            idx = batch_values == batch
            if batch in self.batch_params_:
                mean, std = self.batch_params_[batch]
                transformed_residuals[idx] = (
                    ((residuals[idx] - mean) / std) * self.pooled_std_ + self.pooled_mean_
                )
        out = df.copy()
        out[self.feature_cols] = transformed_residuals + covariate_effect
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def fit_harmonizer(
    train_df: pd.DataFrame,
    *,
    batch_col: str,
    feature_cols: list[str],
    covar_cols: list[str] | None = None,
    allow_outcome_covariates: bool = False,
) -> LocationScaleHarmonizer:
    """Fit harmonization parameters on a training set only."""
    return LocationScaleHarmonizer(
        batch_col=batch_col,
        feature_cols=feature_cols,
        covar_cols=covar_cols,
        allow_outcome_covariates=allow_outcome_covariates,
    ).fit(train_df)


def harmonize_train_valid(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    batch_col: str,
    feature_cols: list[str],
    covar_cols: list[str] | None = None,
    allow_outcome_covariates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, LocationScaleHarmonizer]:
    """Fit on train and transform train/validation without refitting on validation."""
    harmonizer = fit_harmonizer(
        train_df,
        batch_col=batch_col,
        feature_cols=feature_cols,
        covar_cols=covar_cols,
        allow_outcome_covariates=allow_outcome_covariates,
    )
    return harmonizer.transform(train_df), harmonizer.transform(valid_df), harmonizer
