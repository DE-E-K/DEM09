"""api.py
Flask REST API for flight fare prediction.

Endpoints
---------
GET  /health        — liveness check
POST /predict       — predict fare from JSON payload

Production usage:
    gunicorn -w 4 api:app

Development:
    python api.py
"""
from __future__ import annotations

import logging
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from src.config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    LOGS_DIR,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
)

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
_api_log_file = LOGS_DIR / "api.log"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_api_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
logger.info("API log file → %s", _api_log_file)

app = Flask(__name__)

# Model features (must match training exactly)
# ---------------------------------------------------------------------------
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # 11 columns

REQUIRED_FIELDS = [
    "airline", "source_name", "destination_name",
    "class", "seasonality", "departure_period",
    "weekday", "duration_hrs", "stopovers",
    "days_before_departure", "month",
    "booking_source"  # optional — defaults to most common value if absent
]

# Global resources — initialised to safe defaults before attempted load
# ---------------------------------------------------------------------------
model = None
_default_seasonality = "Regular"
_default_booking_source = "Online Website"

try:
    model = joblib.load(BEST_MODEL_PATH)
    _ref_df = pd.read_csv(PROCESSED_DATA_PATH)
    if "seasonality" in _ref_df.columns:
        _default_seasonality = str(_ref_df["seasonality"].mode().iloc[0])
    if "booking_source" in _ref_df.columns:
        _default_booking_source = str(_ref_df["booking_source"].mode().iloc[0])
    logger.info("Model loaded from %s", BEST_MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load model: %s — /predict will return 503", exc)


# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run the training pipeline first."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided."}), 400

    records = [data] if isinstance(data, dict) else data
    input_df = pd.DataFrame(records)

    # -- Required field validation ------------------------------------
    missing = [f for f in REQUIRED_FIELDS if f not in input_df.columns]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # -- Default optional fields -------------------------------------
    if "booking_source" not in input_df.columns:
        input_df["booking_source"] = _default_booking_source

    # -- Coerce numeric columns ---------------------------------------
    for col in ["duration_hrs", "stopovers", "days_before_departure", "month"]:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

    # -- Select only the model features (enforce column order) --------
    input_df = input_df[MODEL_FEATURES]

    try:
        log_pred = model.predict(input_df)
        # Correct inverse of np.log1p used during training
        fares_bdt = np.expm1(log_pred).tolist()
        return jsonify({"prediction_bdt": fares_bdt})
    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 400


# Entry point (development only — use gunicorn in production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, port=5000)

