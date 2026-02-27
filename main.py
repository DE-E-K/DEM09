"""main.py  (project root)
End-to-end pipeline orchestrator.
Run from the project root:
    python main.py --run-all
    python main.py --run-all --data-path data/raw/Flight_Price_Dataset_of_Bangladesh.csv

Preprocessing runs ONCE; the resulting DataFrame is passed to every downstream step.
Artifacts are written with a single run timestamp.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import joblib
import pandas as pd

from src.config import (
    LOGS_DIR,
    MODELS_DIR,
    RAW_DATA_PATH,
    REPORTS_DIR,
    MAX_TRAIN_ROWS,
)
from src.data_preprocessing import load_and_preprocess
from src.eda import run_eda
from src.interpret import interpret_model, plot_actual_vs_predicted, plot_residuals
from src.train import split_features_target, train_baseline_models
from src.tune import tune_models

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_run_ts_init = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_log_file = LOGS_DIR / f"run_{_run_ts_init}.log"

_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)          # capture everything in the file
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter(_LOG_FORMAT))

logging.basicConfig(level=logging.DEBUG, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)
logger.info("Log file → %s", _log_file)


def run_pipeline(data_path: str | None = None) -> dict:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Pipeline started — run_id=%s", run_ts)

    # Step 1: Preprocessing (runs ONCE)
    # ------------------------------------------------------------------
    bundle = load_and_preprocess(path=data_path or str(RAW_DATA_PATH), save_processed=True)
    logger.info("Preprocessed %d rows, %d features", len(bundle.df), len(bundle.feature_columns))

    # Step 2: EDA (non-blocking — skip on error)
    # ------------------------------------------------------------------
    try:
        run_eda(bundle.df)
    except Exception as exc:
        logger.warning("EDA skipped due to error: %s", exc)
    
    # Step 3: Train/test split (uses already-preprocessed df)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_features_target(bundle.df, max_rows=MAX_TRAIN_ROWS)
    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # Step 4: Baseline training
    # ------------------------------------------------------------------
    baseline_results, baseline_models, baseline_predictions = train_baseline_models(
        X_train, X_test, y_train, y_test,
        categorical_cols=bundle.categorical_columns,
        numerical_cols=bundle.numerical_columns,
    )

    # Step 5: Hyperparameter tuning
    # ------------------------------------------------------------------
    tuned_results, tuned_models = tune_models(
        X_train, X_test, y_train, y_test,
        categorical_cols=bundle.categorical_columns,
        numerical_cols=bundle.numerical_columns,
    )

    # Step 6: Compare and select best model
    # ------------------------------------------------------------------
    comparison = pd.concat(
        [baseline_results, tuned_results],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse").reset_index(drop=True)

    comparison["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = REPORTS_DIR / f"model_comparison_{run_ts}.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info("Model comparison saved → %s", comparison_path.name)

    best_row = comparison.iloc[0]
    best_model_name: str = best_row["model"]
    all_trained = {**baseline_models, **tuned_models}

    if best_model_name not in all_trained:
        logger.error("Best model '%s' not found in trained models dict!", best_model_name)
        raise RuntimeError(f"Best model '{best_model_name}' is missing from trained models.")

    best_model = all_trained[best_model_name]
    best_pred = best_model.predict(X_test)
    logger.info(
        "Best model: %s  R2=%.4f  RMSE=%.4f",
        best_model_name, best_row["r2"], best_row["rmse"],
    )

    # Step 7: Save best model artifact
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / f"best_model_{run_ts}.joblib"
    joblib.dump(best_model, model_save_path)
    logger.info("Best model saved → %s", model_save_path.name)

    # Step 8: Diagnostics and interpretation (non-blocking)
    # ------------------------------------------------------------------
    try:
        plot_actual_vs_predicted(y_test, best_pred)
        plot_residuals(y_test, best_pred)
    except Exception as exc:
        logger.warning("Diagnostic plots failed: %s", exc)

    try:
        interpret_model(best_model_name, best_model)
    except Exception as exc:
        logger.warning("Model interpretation failed: %s", exc)

    return {
        "comparison": comparison,
        "best_model": best_model_name,
        "model_path": str(model_save_path),
        "comparison_path": str(comparison_path),
        "run_id": run_ts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flight Fare Prediction Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw dataset CSV")
    parser.add_argument("--run-all", action="store_true", help="Run full pipeline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_all:
        result = run_pipeline(data_path=args.data_path)
        logger.info("Pipeline complete — best model: %s", result["best_model"])
        logger.info("Artifacts: model=%s  comparison=%s", result["model_path"], result["comparison_path"])
    else:
        logger.info("Use --run-all to execute the full pipeline.")


if __name__ == "__main__":
    main()
