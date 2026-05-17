# -*- coding: utf-8 -*-
"""Data loading utilities for tabular radiomics features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LABEL_CANDIDATES = ("cat", "Cat", "label", "Label", "target", "Target")
ID_CANDIDATES = ("ID", "id", "PatientID", "patient_id", "case_id")


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    colset = set(columns)
    for candidate in candidates:
        if candidate in colset:
            return candidate
    return None


def load_feature_table(
    path: str | Path,
    *,
    id_col: str | None = None,
    label_col: str | None = None,
    drop_hu_raw: bool = True,
) -> pd.DataFrame:
    """Load a feature table and standardize identifier and label columns."""
    df = _read_table(path)
    if df.empty:
        raise ValueError(f"Input table is empty: {path}")

    if id_col is None:
        id_col = _first_existing(df.columns, ID_CANDIDATES)
    if id_col is not None and id_col != "ID":
        df = df.rename(columns={id_col: "ID"})

    if label_col is None:
        label_col = _first_existing(df.columns, LABEL_CANDIDATES)
    if label_col is not None and label_col != "cat":
        df = df.rename(columns={label_col: "cat"})

    if "ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ID"})

    if drop_hu_raw:
        raw_cols = [c for c in df.columns if "HU_Raw" in str(c)]
        df = df.drop(columns=raw_cols)

    df["ID"] = df["ID"].astype(str)
    return df


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: Iterable[str] = (),
    require_numeric: bool = True,
) -> list[str]:
    """Infer candidate feature columns without using labels or identifiers."""
    base_exclude = {"ID", "cat", "Center", "center", "Batch", "batch", "Site", "site"}
    excluded = base_exclude | set(exclude_cols)
    cols = [c for c in df.columns if c not in excluded and not str(c).startswith("DL_")]
    if require_numeric:
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric feature columns were found.")
    return cols


def load_and_split(
    path: str | Path,
    *,
    feature_cols: list[str] | None = None,
    id_col: str | None = None,
    label_col: str | None = None,
    drop_hu_raw: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Load a labelled feature table as X, y, ids, feature_names, df."""
    df = load_feature_table(
        path, id_col=id_col, label_col=label_col, drop_hu_raw=drop_hu_raw
    )
    if "cat" not in df.columns:
        raise ValueError("No label column found. Expected one of: " + ", ".join(LABEL_CANDIDATES))

    if feature_cols is None:
        feature_cols = infer_feature_columns(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}")

    ids = df["ID"].to_numpy(dtype=str)
    y = df["cat"].to_numpy(dtype=int)
    X = df[feature_cols].to_numpy(dtype=float)
    return X, y, ids, feature_cols, df
