import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_DATA_FILENAME = "Flight_Price_Dataset_of_Bangladesh.csv"
RAW_DATA_PATH = DATA_RAW_DIR / RAW_DATA_FILENAME

# Dynamic artifact paths — always resolve to the newest timestamped file so
# a re-run never silently loads a stale artifact.
# Override with env vars for container / CI deployments.
# ---------------------------------------------------------------------------
def _newest(directory: Path, pattern: str, fallback: str) -> Path:
    env_override = os.environ.get(fallback.upper().replace("-", "_"))
    if env_override:
        return Path(env_override)
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else directory / fallback


PROCESSED_DATA_PATH = _newest(
    DATA_PROCESSED_DIR,
    "flight_fares_processed_*.csv",
    "flight_fares_processed_latest.csv",
)
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
BEST_MODEL_PATH = _newest(
    MODELS_DIR,
    "best_model_*.joblib",
    "best_model_latest.joblib",
)

# Target columns
# ---------------------------------------------------------------------------
TARGET_COL = "total_fare_log"       # log1p-transformed target used during training
TARGET_ORIGINAL = "total_fare_bdt"  # raw BDT column in the source data

# Canonical feature set (mirrors notebook essential_cols)
# These are the ONLY columns passed to model.predict().
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "airline",
    "source_name",
    "destination_name",
    "class",
    "seasonality",
    "weekday",
    "departure_period",
    "booking_source",   # booking channel affects pricing (online vs agent vs airline direct)
]

NUMERIC_FEATURES = [
    "duration_hrs",
    "stopovers",
    "days_before_departure",
    "month",
]

ESSENTIAL_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COL]

TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3
MAX_TRAIN_ROWS = None  # None = use the full dataset

REGRESSION_METRICS = ["r2", "mae", "mape", "rmse", "max_error"]
