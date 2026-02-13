from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best_model.joblib"

REQUIRED_FIELDS = {
    "airline",
    "source",
    "destination",
    "base_fare",
    "tax_surcharge",
    "month",
    "day",
    "weekday",
    "season",
}


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)


def validate_payload(payload: Dict) -> Dict:
    missing = sorted(REQUIRED_FIELDS - set(payload.keys()))
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    return payload


def predict(payload: Dict) -> float:
    validated = validate_payload(payload)
    model = load_model()

    row = pd.DataFrame([validated])
    pred = model.predict(row)
    return float(pred[0])
