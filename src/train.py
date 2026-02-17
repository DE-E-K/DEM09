from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import MAX_TRAIN_ROWS, RANDOM_STATE, TARGET_COL, TRAIN_TEST_SPLIT
from data_preprocessing import train_test_ready


ModelMap = Dict[str, object]


def build_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    transformers = []

    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            )
        )

    if numerical_cols:
        transformers.append(
            (
                "numerical",
                StandardScaler(),
                numerical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_models() -> ModelMap:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=10000),
        "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def evaluate_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
    )


def train_baseline_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, np.ndarray]]:
    models = get_models()
    model_pipelines: Dict[str, Pipeline] = {}
    predictions: Dict[str, np.ndarray] = {}

    records = []
    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate_regression_metrics(y_test, y_pred)
        records.append(
            {
                "model": model_name,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )
        model_pipelines[model_name] = pipe
        predictions[model_name] = y_pred

    result_df = pd.DataFrame(records).sort_values(by="rmse")
    return result_df, model_pipelines, predictions


def load_split_data(path: str | None = None, max_rows: int | None = MAX_TRAIN_ROWS):
    X, y, categorical_cols, numerical_cols = train_test_ready(path)

    if max_rows is not None and len(X) > max_rows:
        sampled_index = X.sample(n=max_rows, random_state=RANDOM_STATE).index
        X = X.loc[sampled_index].copy()
        y = y.loc[sampled_index].copy()

    X_train, X_test, y_train, y_test = split_data(X, y)
    return X_train, X_test, y_train, y_test, categorical_cols, numerical_cols
