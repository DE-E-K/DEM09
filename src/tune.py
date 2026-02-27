"""tune.py
Hyperparameter tuning that mirrors the notebook's search_spaces (cell 51):
  - Ridge / Lasso        → GridSearchCV
  - RandomForest         → RandomizedSearchCV  (n_iter=12)
  - GradientBoosting     → RandomizedSearchCV  (n_iter=12)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from .train import (
    build_preprocessor_linear,
    build_preprocessor_tree,
    evaluate_regression_metrics,
    get_models,
)

logger = logging.getLogger(__name__)

# Search spaces — mirrors notebook cell 51
# ---------------------------------------------------------------------------

_SEARCH_SPACES = {
    "Ridge": {
        "type": "grid",
        "params": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0]},
    },
    "Lasso": {
        "type": "grid",
        "params": {"model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0]},
    },
    "RandomForest": {
        "type": "random",
        "n_iter": 12,
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
    },
    "GradientBoosting": {
        "type": "random",
        "n_iter": 12,
        "params": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
        },
    },
}


# Tuning orchestration
# ---------------------------------------------------------------------------

def tune_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    categorical_cols: List[str] | None = None,
    numerical_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    cat_cols = categorical_cols or CATEGORICAL_FEATURES
    num_cols = numerical_cols or NUMERIC_FEATURES

    # Reference base pipelines from train.get_models
    base_pipelines = get_models(num_cols, cat_cols)

    records = []
    best_models: Dict[str, object] = {}

    for name, cfg in _SEARCH_SPACES.items():
        if name not in base_pipelines:
            logger.warning("No base pipeline found for %s — skipping tuning", name)
            continue

        try:
            pipe = clone(base_pipelines[name])

            if cfg["type"] == "grid":
                search = GridSearchCV(
                    estimator=pipe,
                    param_grid=cfg["params"],
                    cv=CV_FOLDS,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                )
            else:
                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=cfg["params"],
                    n_iter=cfg.get("n_iter", 10),
                    cv=CV_FOLDS,
                    scoring="neg_root_mean_squared_error",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )

            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            pred = best_pipe.predict(X_test)
            metrics = evaluate_regression_metrics(y_test, pred)

            records.append({
                "model": f"{name}_Tuned",
                "stage": "tuned",
                "cv_best_rmse": float(-search.best_score_),
                "best_params": str(search.best_params_),
                **metrics,
            })
            best_models[f"{name}_Tuned"] = best_pipe
            logger.info(
                "Tuned  %-25s  R2=%.4f  RMSE=%.4f  CV_RMSE=%.4f",
                f"{name}_Tuned", metrics["r2"], metrics["rmse"], -search.best_score_,
            )
        except Exception as exc:
            logger.warning("Tuning %s failed: %s", name, exc)

    results = pd.DataFrame(records).sort_values("rmse").reset_index(drop=True)
    return results, best_models
