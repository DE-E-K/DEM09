from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from config import CV_FOLDS, RANDOM_STATE
from train import build_preprocessor, evaluate_regression_metrics


def _search_spaces() -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    return {
        "Ridge": (
            Ridge(random_state=RANDOM_STATE),
            {"model__alpha": [0.1, 1.0, 10.0, 50.0]},
        ),
        "Lasso": (
            Lasso(random_state=RANDOM_STATE),
            {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        "DecisionTree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_split": [2, 5, 10],
            },
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 8, 15],
                "model__min_samples_split": [2, 5],
            },
        ),
    }


def tune_models(
    X_train,
    X_test,
    y_train,
    y_test,
    categorical_cols,
    numerical_cols,
):
    records = []
    best_models: Dict[str, Pipeline] = {}

    for model_name, (base_model, param_grid) in _search_spaces().items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(categorical_cols, numerical_cols)),
                ("model", base_model),
            ]
        )
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=CV_FOLDS,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        pred = best.predict(X_test)
        metrics = evaluate_regression_metrics(y_test, pred)

        records.append(
            {
                "model": model_name,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "cv_best_rmse": float(-search.best_score_),
                "best_params": search.best_params_,
            }
        )
        best_models[model_name] = best

    results = pd.DataFrame(records).sort_values("rmse")
    return results, best_models
