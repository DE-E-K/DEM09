"""train.py
Baseline model training with two separate sklearn Pipelines:
  - preprocessor_linear  (StandardScaler  + OHE) for linear models
  - preprocessor_tree    (median imputer  + OHE) for tree-based models

Target column is total_fare_log (np.log1p).  Inverse: np.expm1.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import (
    CATEGORICAL_FEATURES,
    MAX_TRAIN_ROWS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_TEST_SPLIT,
)

logger = logging.getLogger(__name__)

ModelMap = Dict[str, Pipeline]

# Preprocessing factories (mirrors notebook cell 47)
# ---------------------------------------------------------------------------

def _categorical_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_preprocessor_linear(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """For linear models: applies StandardScaler on numeric columns."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", _categorical_pipe(), categorical_features),
    ], remainder="drop")


def build_preprocessor_tree(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """For tree models: no scaling, median imputation only."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", _categorical_pipe(), categorical_features),
    ], remainder="drop")


# Model registry with preprocessor routing
# ---------------------------------------------------------------------------

def get_models(
    numeric_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Pipeline]:
    """Return a dict of {name: fitted Pipeline} ready for .fit()."""
    prep_linear = build_preprocessor_linear(numeric_features, categorical_features)
    prep_tree = build_preprocessor_tree(numeric_features, categorical_features)

    return {
        "LinearRegression": Pipeline([
            ("prep", prep_linear), ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("prep", prep_linear), ("model", Ridge()),
        ]),
        "Lasso": Pipeline([
            ("prep", prep_linear), ("model", Lasso(max_iter=5000)),
        ]),
        "HuberRegressor": Pipeline([
            ("prep", prep_linear), ("model", HuberRegressor()),
        ]),
        "DecisionTree": Pipeline([
            ("prep", prep_tree), ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("prep", prep_tree), ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("prep", prep_tree), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]),
    }


# Metrics (mirrors notebook evaluate())
# ---------------------------------------------------------------------------

def evaluate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    try:
        rmse = float(mean_squared_error(y_true_arr, y_pred_arr, squared=False))
    except TypeError:
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    eps = 1e-8
    denom = np.where(np.abs(y_true_arr) < eps, eps, np.abs(y_true_arr))
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / denom)))

    return {
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "mape": mape,
        "rmse": rmse,
        "max_error": float(max_error(y_true_arr, y_pred_arr)),
    }


# Baseline training
# ---------------------------------------------------------------------------

def train_baseline_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: List[str] | None = None,
    numerical_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, ModelMap, Dict[str, np.ndarray]]:
    cat_cols = categorical_cols or CATEGORICAL_FEATURES
    num_cols = numerical_cols or NUMERIC_FEATURES

    pipelines = get_models(num_cols, cat_cols)
    records = []
    trained: ModelMap = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, pipe in pipelines.items():
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = evaluate_regression_metrics(y_test, y_pred)
            records.append({"model": name, "stage": "baseline", **metrics})
            trained[name] = pipe
            predictions[name] = y_pred
            logger.info("Baseline %-20s  R2=%.6f  RMSE=%.6f", name, metrics["r2"], metrics["rmse"])
        except Exception as exc:
            logger.warning("Baseline %s failed: %s", name, exc)

    if not records:
        raise RuntimeError(
            "All baseline models failed to train. "
            "Check that preprocessing produced the expected numeric/categorical columns."
        )
    results = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
    return results, trained, predictions


# Helper used by main.py to prepare split data from a pre-processed bundle
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
    max_rows: int | None = MAX_TRAIN_ROWS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(float).copy()

    # drop rows with missing target
    valid = y.notna()
    X, y = X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)

    if max_rows is not None and len(X) > max_rows:
        idx = X.sample(n=max_rows, random_state=RANDOM_STATE).index
        X, y = X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)

    return train_test_split(X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)
