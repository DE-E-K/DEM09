from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, MAX_TRAIN_ROWS, MODEL_COMPARISON_PATH, RAW_DATA_PATH
from src.data_preprocessing import load_and_preprocess
from src.eda import run_eda
from src.interpret import interpret_model, plot_actual_vs_predicted, plot_residuals
from src.train import load_split_data, train_baseline_models
from src.tune import tune_models


def run_pipeline(data_path: str | None = None):
    bundle = load_and_preprocess(path=data_path or str(RAW_DATA_PATH), save_processed=True)

    run_eda(bundle.df)

    X_train, X_test, y_train, y_test, categorical_cols, numerical_cols = load_split_data(
        data_path,
        max_rows=MAX_TRAIN_ROWS,
    )

    baseline_results, baseline_models, baseline_predictions = train_baseline_models(
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_cols,
        numerical_cols,
    )

    tuned_results, tuned_models = tune_models(
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_cols,
        numerical_cols,
    )

    comparison = pd.concat(
        [
            baseline_results.assign(stage="baseline"),
            tuned_results.assign(stage="tuned"),
        ],
        ignore_index=True,
    ).sort_values("rmse")

    MODEL_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    best_row = comparison.iloc[0]
    best_model_name = best_row["model"]
    best_stage = best_row["stage"]

    if best_stage == "baseline":
        best_model = baseline_models[best_model_name]
        best_pred = baseline_predictions[best_model_name]
    else:
        best_model = tuned_models[best_model_name]
        best_pred = best_model.predict(X_test)

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)

    plot_actual_vs_predicted(y_test, best_pred)
    plot_residuals(y_test, best_pred)
    interpret_model(best_model_name, best_model)

    return {
        "comparison": comparison,
        "best_model": best_model_name,
        "best_stage": best_stage,
        "model_path": str(BEST_MODEL_PATH),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flight Fare Prediction Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Optional path to raw dataset CSV")
    parser.add_argument("--run-all", action="store_true", help="Run full preprocessing, EDA, training, tuning")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_all:
        result = run_pipeline(data_path=args.data_path)
        print("Pipeline complete")
        print(f"Best model: {result['best_model']} ({result['best_stage']})")
        print(f"Saved: {result['model_path']}")
    else:
        print("Use --run-all to execute the full pipeline.")


if __name__ == "__main__":
    main()
