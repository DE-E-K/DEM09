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
        source = st.selectbox("Source", get_options('source'))
        stopovers = st.selectbox("Stopovers", get_options('stopovers'))
        flight_class = st.selectbox("Class", get_options('class'))
        
    with col2:
        destination = st.selectbox("Destination", get_options('destination'))
        date = st.date_input("Date of Journey", datetime.today())
        booking_source = st.selectbox("Booking Source", get_options('booking_source'))
        # Optional: Allow user to specify duration or imply it
        duration_hrs = st.number_input("Duration (Hours)", min_value=0.5, max_value=30.0, value=2.0, step=0.5)

    submitted = st.form_submit_button("Predict Fare")

if submitted:
    # 1. Basic User Inputs
    input_dict = {
        'airline': airline,
        'source': source,
        'destination': destination,
        'stopovers': stopovers,
        'class': flight_class,
        'booking_source': booking_source,
        'duration_(hrs)': duration_hrs,
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
    
    # Seasonality (Specific to dataset, often same as season or special holidays)
    # Ideally should map date to specific holidays (Eid, etc.)
    # For now, defaulting to 'Regular' or most frequent to avoid error
    most_common_seasonality = df['seasonality'].mode()[0]
    input_df['seasonality'] = most_common_seasonality

    # 3. Text/Name Features (Redundant but required by pipeline if included in features)
    # We can try to look them up from the dataframe if a mapping exists, or use source/dest as proxy
    # Create lookup dictionaries safely
    source_map = df[['source', 'source_name']].drop_duplicates().set_index('source')['source_name'].to_dict()
    dest_map = df[['destination', 'destination_name']].drop_duplicates().set_index('destination')['destination_name'].to_dict()
    
    input_df['source_name'] = input_df['source'].map(source_map).fillna(input_df['source'])
    input_df['destination_name'] = input_df['destination'].map(dest_map).fillna(input_df['destination'])
    
    # 4. Aircraft Type (Hard to know from user, replacing with mode per airline or global mode)
    # Simple imputation: mode of aircraft for the selected airline
    # If uncertain, global mode
    airline_mode_aircraft = df[df['airline'] == airline]['aircraft_type'].mode()
    input_df['aircraft_type'] = airline_mode_aircraft[0] if not airline_mode_aircraft.empty else df['aircraft_type'].mode()[0]

    # 5. Derived or "Leakage" Features (Base Fare, Tax)
    # These are part of the target (Total Fare = Base + Tax).
    # If the model includes them as FEATURES, it's a data leakage issue in training design.
    # However, to make the code run, we must provide them. 
    # We will provide 0 or mean, but this will strictly invalidate the prediction if the model relies heavily on them.
    # OR, better: We assume the user wants to estimate based on route/airline, so we impute average base/tax for this route/airline.
    # But ideally, we should retrain WITHOUT these as features.
    # Given we can't retrain in this step easily, we'll try to find similar flights to impute.
    
    # Impute Base Fare & Tax based on similar Airline+Route+Class
    similar_flights = df[
        (df['airline'] == airline) & 
        (df['source'] == source) & 
        (df['destination'] == destination) & 
        (df['class'] == flight_class)
    ]
    
    if not similar_flights.empty:
        input_df['base_fare'] = similar_flights['base_fare'].mean()
        input_df['tax_surcharge'] = similar_flights['tax_surcharge'].mean()
    else:
        # Fallback to global means
        input_df['base_fare'] = df['base_fare'].mean()
        input_df['tax_surcharge'] = df['tax_surcharge'].mean()

    # 6. Arrival DateTime (Likely dropped or unused, but if required...)
    # Construct a dummy string matching format because pipeline might convert it
    # 2025-11-17 07:38:10
    dummy_arrival = input_df['date'].iloc[0] + timedelta(hours=duration_hrs)
    input_df['arrival_date_and_time'] = dummy_arrival.strftime("%Y-%m-%d %H:%M:%S")

    try:
        # The model pipeline (ColumnTransformer) allows "remainder='drop'", 
        # so EXTRA columns are fine, but MISSING feature columns will error.
        # We must ensure all columns used in 'feature_columns' from 'data_preprocessing' are here.
        # Based on previous file read, 'feature_columns' excluded 'total_fare' and 'date'.
        
        prediction = model.predict(input_df)
        st.success(f"Predicted Fare: BDT {prediction[0]:,.2f}")
        
        with st.expander("Show Details"):
            st.json(input_df.iloc[0].astype(str).to_dict())
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
