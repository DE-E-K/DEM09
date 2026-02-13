from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_ALIASES,
    CITY_NORMALIZATION_MAP,
    DATE_ALIASES,
    NUMERIC_ALIASES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SEASON_MAP,
    TARGET_COL,
)


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    feature_columns: List[str]
    categorical_columns: List[str]
    numerical_columns: List[str]


def normalize_column_name(name: str) -> str:
    cleaned = str(name).strip().lower().replace("&", "and")
    cleaned = cleaned.replace("/", "_").replace("-", "_").replace(" ", "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=normalized)
    drop_cols = [c for c in df.columns if c.startswith("unnamed") or c in {"index", "id"}]
    return df.drop(columns=drop_cols, errors="ignore")


def _find_first_existing_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    alias_set = {normalize_column_name(a) for a in aliases}
    for col in df.columns:
        if normalize_column_name(col) in alias_set:
            return col
    return None


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}

    for canonical, aliases in NUMERIC_ALIASES.items():
        found = _find_first_existing_column(df, aliases)
        if found:
            rename_map[found] = canonical

    for canonical, aliases in CATEGORICAL_ALIASES.items():
        found = _find_first_existing_column(df, aliases)
        if found:
            rename_map[found] = canonical

    date_col = _find_first_existing_column(df, DATE_ALIASES)
    if date_col:
        rename_map[date_col] = "date"

    return df.rename(columns=rename_map)


def coerce_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_city_names(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["source", "destination"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace(CITY_NORMALIZATION_MAP)
                .str.title()
            )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in ["base_fare", "tax_surcharge", "total_fare"] if c in df.columns]
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = [c for c in ["airline", "source", "destination"] if c in df.columns]
    for col in categorical_cols:
        mode_value = df[col].mode(dropna=True)
        fallback = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        df[col] = df[col].fillna(fallback).replace("", fallback)

    return df


def fix_invalid_fares(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["base_fare", "tax_surcharge", "total_fare"]:
        if col in df.columns:
            df = df[df[col].isna() | (df[col] >= 0)]
    return df


def build_total_fare_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "total_fare" not in df.columns and {"base_fare", "tax_surcharge"}.issubset(df.columns):
        df["total_fare"] = df["base_fare"].fillna(0) + df["tax_surcharge"].fillna(0)

    if "total_fare" in df.columns and {"base_fare", "tax_surcharge"}.issubset(df.columns):
        recomputed = df["base_fare"].fillna(0) + df["tax_surcharge"].fillna(0)
        missing_total = df["total_fare"].isna()
        df.loc[missing_total, "total_fare"] = recomputed[missing_total]

    return df


def engineer_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.day_name()
    df["season"] = df["month"].map(SEASON_MAP)
    return df


def prepare_features_target(df: pd.DataFrame) -> DatasetBundle:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not available after preprocessing.")

    excluded = {TARGET_COL, "date"}
    feature_columns = [c for c in df.columns if c not in excluded]

    categorical_columns = [
        c for c in feature_columns
        if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]) or c in {"weekday", "season"}
    ]
    numerical_columns = [c for c in feature_columns if c not in categorical_columns]

    return DatasetBundle(
        df=df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )


def load_and_preprocess(path: str | None = None, save_processed: bool = True) -> DatasetBundle:
    data_path = path or str(RAW_DATA_PATH)
    df = pd.read_csv(data_path)
    df = clean_columns(df)
    df = canonicalize_columns(df)
    df = coerce_numeric(df, ["base_fare", "tax_surcharge", "total_fare"])
    df = normalize_city_names(df)
    df = build_total_fare_if_needed(df)
    df = handle_missing_values(df)
    df = fix_invalid_fares(df)
    df = engineer_date_features(df)

    bundle = prepare_features_target(df)

    if save_processed:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        bundle.df.to_csv(PROCESSED_DATA_PATH, index=False)

    return bundle


def train_test_ready(path: str | None = None) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    bundle = load_and_preprocess(path=path, save_processed=True)
    X = bundle.df[bundle.feature_columns].copy()
    y = bundle.df[TARGET_COL].copy()
    return X, y, bundle.categorical_columns, bundle.numerical_columns
