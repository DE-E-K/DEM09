from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_FILENAME = "Flight_Price_Dataset_of_Bangladesh.csv"
RAW_DATA_PATH = DATA_RAW_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "flight_fares_processed.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TARGET_COL = "total_fare"
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

NUMERIC_ALIASES = {
    "base_fare": ["base_fare", "base fare", "baseprice", "base_price", "fare"],
    "tax_surcharge": ["tax_surcharge", "tax & surcharge", "tax", "taxes", "surcharge"],
    "total_fare": ["total_fare", "total fare", "total", "price", "ticket_price", "ticket price"],
}

CATEGORICAL_ALIASES = {
    "airline": ["airline", "carrier"],
    "source": ["source", "origin", "from", "departure_city"],
    "destination": ["destination", "dest", "to", "arrival_city"],
}

DATE_ALIASES = ["date", "journey_date", "travel_date", "departure_date"]

CITY_NORMALIZATION_MAP = {
    "dacca": "dhaka",
    "ctg": "chittagong",
}

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn",
}

REGRESSION_METRICS = ["r2", "mae", "rmse"]
