"""flight_retraining_dag.py
Weekly automated retraining pipeline for the Flight Fare Prediction model.

Schedule  : monthly (1st day, midnight)
Trigger   : Airflow scheduler
Entry point: python main.py --run-all   (root-level orchestrator)
Project root is read from the FLIGHT_FARE_PROJECT_ROOT env var;
falls back to /opt/airflow/dags/repo if not set.

Task flow:
    validate_data → retrain_model → validate_model → notify_success
                        ↓ (on failure)
                    notify_failure
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# Project root — override via env var for different deployment environments
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.environ.get("FLIGHT_FARE_PROJECT_ROOT", "/opt/airflow/dags/repo")
RAW_DATA_FILE = os.path.join(PROJECT_ROOT, "data", "raw", "Flight_Price_Dataset_of_Bangladesh.csv")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR      = os.path.join(PROJECT_ROOT, "logs")

# Default args
# ---------------------------------------------------------------------------
default_args = {
    "owner": "flight-fare-ml",
    "depends_on_past": False,
    "email": [os.environ.get("ALERT_EMAIL", "")],
    "email_on_failure": bool(os.environ.get("ALERT_EMAIL")),
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=2),   # abort if a task hangs
}

# Task callables
# ---------------------------------------------------------------------------

def validate_data(**context) -> None:
    """Raise if the raw dataset is missing or empty."""
    path = Path(RAW_DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_FILE}\n"
            "Mount the data volume or place the CSV before running."
        )
    size_mb = path.stat().st_size / 1_048_576
    if size_mb < 0.1:
        raise ValueError(f"Raw dataset looks empty ({size_mb:.2f} MB): {RAW_DATA_FILE}")
    log.info("Raw dataset OK — %.2f MB at %s", size_mb, RAW_DATA_FILE)


def validate_model(**context) -> None:
    """
    After retraining, confirm:
      1. A new .joblib artifact exists in models/
      2. The model loads without error
      3. A smoke-test prediction returns a positive float
    """
    import glob, joblib, numpy as np, pandas as pd

    sys.path.insert(0, PROJECT_ROOT)

    artifacts = sorted(glob.glob(os.path.join(MODELS_DIR, "best_model_*.joblib")))
    if not artifacts:
        raise RuntimeError(f"No model artifact found in {MODELS_DIR} after retraining.")

    latest = artifacts[-1]
    log.info("Validating model artifact: %s", latest)

    model = joblib.load(latest)

    # Smoke-test: one row with plausible feature values
    from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    sample = pd.DataFrame([{
        "airline": "Biman Bangladesh Airlines",
        "source_name": "Hazrat Shahjalal International Airport",
        "destination_name": "Shah Amanat International Airport",
        "class": "Economy",
        "seasonality": "Regular",
        "weekday": "Wednesday",
        "departure_period": "Morning",
        "booking_source": "Online Website",
        "duration_hrs": 1.2,
        "stopovers": 0,
        "days_before_departure": 14,
        "month": 6,
    }])

    log_pred = model.predict(sample[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    fare_bdt = float(np.expm1(log_pred[0]))

    if fare_bdt <= 0:
        raise ValueError(f"Smoke-test prediction returned non-positive fare: {fare_bdt}")

    log.info(
        "Model validation passed — smoke-test fare = %.2f BDT (artifact: %s)",
        fare_bdt, os.path.basename(latest),
    )
    # Push to XCom so downstream tasks can reference it
    context["ti"].xcom_push(key="smoke_test_fare_bdt", value=round(fare_bdt, 2))
    context["ti"].xcom_push(key="model_artifact",      value=os.path.basename(latest))


def notify_failure(context) -> None:
    """Called by on_failure_callback — logs a structured failure summary."""
    task_instance = context.get("task_instance")
    dag_run       = context.get("dag_run")
    exception     = context.get("exception")
    log.error(
        "DAG FAILURE | dag=%s | run_id=%s | task=%s | error=%s",
        dag_run.dag_id if dag_run else "unknown",
        dag_run.run_id if dag_run else "unknown",
        task_instance.task_id if task_instance else "unknown",
        exception,
    )


# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="flight_fare_retraining",
    default_args=default_args,
    description=(
        "Monthly retraining pipeline: validate data → retrain all models → "
        "validate best model → notify."
    ),
    schedule_interval="@monthly",     # 1st of every month at midnight
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,          # prevent overlapping retraining runs
    tags=["machine-learning", "flight-fare", "monthly"],
    on_failure_callback=notify_failure,
    doc_md="""
## Flight Fare Retraining DAG

Runs every **month** (1st day, midnight). Full pipeline:

| Step | Task | Description |
|---|---|---|
| 1 | `validate_data` | Confirm raw CSV exists and is non-empty |
| 2 | `retrain_model` | `python main.py --run-all` — preprocessing, baseline training, tuning, artifact save |
| 3 | `validate_model` | Load new artifact, run smoke-test prediction, assert fare > 0 BDT |
| 4 | `notify_success` | Log completion summary with model name and smoke-test fare |

**Override project root:** set `FLIGHT_FARE_PROJECT_ROOT` env var in your Airflow deployment.
""",
) as dag:

    # Task 1 — Data validation
    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        doc_md="Check raw dataset exists and is non-empty before triggering retraining.",
    )

    # Task 2 — Full pipeline retraining (root entry point)
    retrain_task = BashOperator(
        task_id="retrain_model",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"mkdir -p {LOGS_DIR} && "
            f"python main.py --run-all "
            f"2>&1 | tee {LOGS_DIR}/airflow_retrain_$(date +%Y%m%d_%H%M%S).log"
        ),
        doc_md=(
            "Runs the full pipeline via `python main.py --run-all`. "
            "Output is tee'd to `logs/airflow_retrain_<timestamp>.log`."
        ),
    )

    # Task 3 — Model artifact validation
    validate_model_task = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
        doc_md=(
            "Loads newest model artifact and runs a smoke-test prediction. "
            "Pushes `smoke_test_fare_bdt` and `model_artifact` to XCom."
        ),
    )

    # Task 4 — Success notification
    notify_success_task = BashOperator(
        task_id="notify_success",
        bash_command=(
            "echo \"Retraining complete — "
            "model={{ ti.xcom_pull(task_ids='validate_model', key='model_artifact') }} | "
            "smoke-test fare={{ ti.xcom_pull(task_ids='validate_model', key='smoke_test_fare_bdt') }} BDT\""
        ),
        doc_md="Logs a completion summary including the model filename and smoke-test fare.",
    )

    # Task dependency chain
    # ---------------------------------------------------------------------------
    validate_data_task >> retrain_task >> validate_model_task >> notify_success_task

