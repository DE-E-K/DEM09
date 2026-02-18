import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np
from src.config import BEST_MODEL_PATH, PROCESSED_DATA_PATH

# Load the model and data
@st.cache_resource
def load_resources():
    model = joblib.load(BEST_MODEL_PATH)
    # Load processed data to get unique values via value_counts for better ordering
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return model, df

try:
    model, df = load_resources()
except FileNotFoundError:
    st.error("Model or processed data not found. Please run the pipeline first.")
    st.stop()

st.title("Flight Fare Prediction App")
st.write("Enter flight details to predict the total fare.")

# Helper to get unique sorted options
def get_options(col):
    return sorted(df[col].dropna().unique().tolist())

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        airline = st.selectbox("Airline", get_options('airline'))
        # Use source_name directly
        source = st.selectbox("Source", get_options('source_name'))
        stopovers = st.selectbox("Stopovers", get_options('stopovers'))
        flight_class = st.selectbox("Class", get_options('class'))
        
    with col2:
        # Use destination_name directly
        destination = st.selectbox("Destination", get_options('destination_name'))
        date = st.date_input("Date of Journey", datetime.today())
        
        departure_period = st.selectbox("Departure Period", ["Morning", "Afternoon", "Evening", "Night"])
        seasonality = st.selectbox("Seasonality (Optional)", ["Regular", "Winter Holidays", "Eid", "Hajj"])
        
        # Optional: Allow user to specify duration or imply it
        duration_hrs = st.number_input("Duration (Hours)", min_value=0.5, max_value=30.0, value=2.0, step=0.5)

    submitted = st.form_submit_button("Predict Fare")

if submitted:
    # 1. Basic User Inputs
    input_dict = {
        'airline': airline,
        'source_name': source, # Mapped to source_name
        'destination_name': destination, # Mapped to destination_name
        'stopovers': stopovers,
        'class': flight_class,
        'departure_period': departure_period,
        'seasonality': seasonality,
        'duration_hrs': duration_hrs,
        'date': pd.to_datetime(date)
    }
    
    input_df = pd.DataFrame([input_dict])

    # 2. Date Features
    input_df['month'] = input_df['date'].dt.month
    input_df['day'] = input_df['date'].dt.day
    input_df['weekday'] = input_df['date'].dt.day_name()
    input_df['days_before_departure'] = (input_df['date'] - pd.Timestamp.now()).dt.days.clip(lower=0) 
    # For simplicity, assuming booking is today. 
    # Or we could ask 'Days in advance' instead of booking date relative to now. 
    # Let's derive it or default to a reasonable median if negative.
    input_df['days_before_departure'] = input_df['days_before_departure'].apply(lambda x: x if x >= 0 else 7) 

    # Season mapping
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Autumn", 11: "Autumn",
    }
    input_df['season'] = input_df['month'].map(season_map)
    
    try:
        # Predict (Log Transformed)
        log_prediction = model.predict(input_df)
        
        # Inverse Transform (Exp)
        prediction = np.exp(log_prediction)
        
        st.success(f"Predicted Fare: BDT {prediction[0]:,.2f}")
        
        with st.expander("Show Details"):
            st.json(input_df.iloc[0].astype(str).to_dict())
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
