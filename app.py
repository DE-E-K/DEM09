import numpy as np
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

from src.config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROCESSED_DATA_PATH,
)

# Resource loading — uses dynamic BEST_MODEL_PATH from config
# ---------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load(BEST_MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return model, df


try:
    model, df = load_resources()
except FileNotFoundError:
    st.error("Model or processed data not found. Run `python -m src.main --run-all` first.")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES   # 11 columns


def get_options(col: str) -> list:
    if col not in df.columns:
        return []
    vals = df[col].dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        return sorted(str(v) for v in vals)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Flight Fare Prediction")
st.write("Enter flight details to estimate the total fare (BDT).")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox("Airline", get_options("airline"))
        source = st.selectbox("Source City", get_options("source_name"))
        destination = st.selectbox("Destination City", get_options("destination_name"))
        flight_class = st.selectbox("Class", get_options("class"))
        booking_source = st.selectbox(
            "Booking Source",
            get_options("booking_source") or ["Online Website", "Travel Agent", "Airline Direct", "Mobile App"],
        )
        stopovers = st.number_input(
            "Stopovers", min_value=0, max_value=5,
            value=int(df["stopovers"].median()) if "stopovers" in df.columns else 0,
            step=1,
        )

    with col2:
        journey_date = st.date_input("Date of Journey", datetime.today())
        departure_period = st.selectbox("Departure Period", ["Morning", "Afternoon", "Evening", "Night"])
        seasonality = st.selectbox("Seasonality", get_options("seasonality") or ["Regular", "Eid", "Winter Holidays", "Hajj"])
        duration_hrs = st.number_input("Flight Duration (hours)", min_value=0.5, max_value=30.0, value=2.0, step=0.5)
        days_before = st.number_input(
            "Days Booked in Advance", min_value=0, max_value=365, value=14, step=1,
        )

    submitted = st.form_submit_button("Predict Fare")


if submitted:
    journey_dt = pd.Timestamp(journey_date)

    # Build input with exactly the MODEL_FEATURES columns
    input_dict = {
        "airline": airline,
        "source_name": source,
        "destination_name": destination,
        "class": flight_class,
        "seasonality": seasonality,
        "weekday": journey_dt.day_name(),
        "departure_period": departure_period,
        "booking_source": booking_source,
        "duration_hrs": float(duration_hrs),
        "stopovers": int(stopovers),
        "days_before_departure": int(days_before),
        "month": journey_dt.month,
    }

    input_df = pd.DataFrame([input_dict])[MODEL_FEATURES]   # enforce column order

    try:
        log_pred = model.predict(input_df)
        # Correct inverse of np.log1p used during training
        fare_bdt = float(np.expm1(log_pred[0]))
        st.success(f"**Estimated Fare: BDT {fare_bdt:,.2f}**")

        with st.expander("Show Input Details"):
            st.dataframe(input_df.T.rename(columns={0: "value"}))

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

