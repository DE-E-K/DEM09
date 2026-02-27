"""data_preprocessing.py
Mirrors the notebook's final preprocessing pipeline exactly.
Run from project root: python -m src.main --run-all
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import (
    DATA_PROCESSED_DIR,
    ESSENTIAL_COLS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RAW_DATA_PATH,
    TARGET_COL,
    TARGET_ORIGINAL,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_columns: List[str]
    categorical_columns: List[str]
    numerical_columns: List[str]
    processed_path: Path


# Step 1 — column name normalisation (identical to notebook `to_snake`)
# ---------------------------------------------------------------------------

def to_snake(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\(\)/&-]+", " ", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake(c) for c in df.columns]
    drop_cols = [c for c in df.columns if c.startswith("unnamed") or c in {"index", "id"}]
    return df.drop(columns=drop_cols, errors="ignore")


# Step 2a — stopovers: convert string labels to integers
# 'Direct' → 0,  '1 Stop' → 1,  '2 Stops' → 2,  numeric strings → int
# ---------------------------------------------------------------------------

def clean_stopovers(df: pd.DataFrame) -> pd.DataFrame:
    col = "stopovers"
    if col not in df.columns:
        return df
    df = df.copy()

    def _parse(val):
        if pd.isna(val):
            return 0
        s = str(val).strip().lower()
        if s in {"direct", "non-stop", "nonstop", "non stop", "0"}:
            return 0
        # extract the first digit found (e.g. '1 stop' → 1)
        import re as _re
        m = _re.search(r"\d+", s)
        return int(m.group()) if m else 0

    df[col] = df[col].apply(_parse).astype(int)
    return df


# Step 2b — type coercion
# ---------------------------------------------------------------------------

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in df.columns if "date" in c or "time" in c]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# Step 3 — reconstruct total_fare_bdt where missing
# ---------------------------------------------------------------------------

def build_total_fare(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_ORIGINAL not in df.columns:
        df[TARGET_ORIGINAL] = np.nan

    if {"base_fare_bdt", "tax_surcharge_bdt"}.issubset(df.columns):
        reconstructed = df["base_fare_bdt"] + df["tax_surcharge_bdt"]
        df[TARGET_ORIGINAL] = df[TARGET_ORIGINAL].fillna(reconstructed)

    return df


# Step 4 — missing value imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        mode_val = df[col].mode(dropna=True)
        fill = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        df[col] = df[col].fillna(fill)

    return df


# Step 5 — remove negative fares
# ---------------------------------------------------------------------------

def remove_negative_fares(df: pd.DataFrame) -> pd.DataFrame:
    fare_cols = [c for c in ["base_fare_bdt", "tax_surcharge_bdt", TARGET_ORIGINAL] if c in df.columns]
    if fare_cols:
        before = len(df)
        df = df[(df[fare_cols] >= 0).all(axis=1)]
        logger.info("Removed %d negative-fare rows", before - len(df))
    return df


# Step 6 — IQR-based outlier capping (winsorization)
# ---------------------------------------------------------------------------

def cap_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# Step 7 — drop duplicates
# ---------------------------------------------------------------------------

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Removed %d duplicate rows", before - len(df))
    return df


# Step 8 — feature engineering from departure_date_time
# ---------------------------------------------------------------------------

def _departure_period(hour) -> str:
    if pd.isna(hour):
        return "Unknown"
    h = int(hour)
    if 5 <= h < 12:
        return "Morning"
    if 12 <= h < 17:
        return "Afternoon"
    if 17 <= h < 21:
        return "Evening"
    return "Night"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    dep_col = next((c for c in df.columns if "departure_date_time" in c or "departure_datetime" in c), None)
    if dep_col:
        dep = df[dep_col]
        df["month"] = dep.dt.month
        df["day"] = dep.dt.day
        df["weekday"] = dep.dt.day_name()
        df["departure_period"] = dep.dt.hour.apply(_departure_period)

    # log-transform the target (np.log1p to handle zeros)
    if TARGET_ORIGINAL in df.columns:
        df[TARGET_COL] = np.log1p(df[TARGET_ORIGINAL])

    return df


# Step 9 — select essential columns
# ---------------------------------------------------------------------------

def select_essential(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in ESSENTIAL_COLS if c in df.columns]
    missing = [c for c in ESSENTIAL_COLS if c not in df.columns]
    if missing:
        logger.warning("Essential columns not found and will be skipped: %s", missing)
    return df[available].copy()


# Public interface
# ---------------------------------------------------------------------------

def load_and_preprocess(
    path: str | None = None,
    save_processed: bool = True,
) -> DatasetBundle:
    data_path = path or str(RAW_DATA_PATH)
    logger.info("Loading raw data from %s", data_path)
    df = pd.read_csv(data_path)

    df = clean_columns(df)
    df = clean_stopovers(df)   # must run BEFORE coerce_types (still a string column)
    df = coerce_types(df)
    df = build_total_fare(df)
    df = impute_missing(df)
    df = remove_negative_fares(df)
    df = cap_outliers_iqr(df)
    df = drop_duplicates(df)
    df = engineer_features(df)
    df = select_essential(df)

    processed_path: Path
    if save_processed:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_path = DATA_PROCESSED_DIR / f"flight_fares_processed_{ts}.csv"
        df.to_csv(processed_path, index=False)
        logger.info("Saved processed data to %s (%d rows)", processed_path.name, len(df))
    else:
        processed_path = DATA_PROCESSED_DIR / "flight_fares_processed_latest.csv"

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    return DatasetBundle(
        df=df,
        feature_columns=feature_cols,
        categorical_columns=CATEGORICAL_FEATURES,
        numerical_columns=NUMERIC_FEATURES,
        processed_path=processed_path,
    )


def train_test_ready(
    path: str | None = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    bundle = load_and_preprocess(path=path, save_processed=True)
    X = bundle.df[bundle.feature_columns].copy()
    y = bundle.df[TARGET_COL].copy()
    return X, y, bundle.categorical_columns, bundle.numerical_columns
