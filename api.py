from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from src.config import BEST_MODEL_PATH, PROCESSED_DATA_PATH

app = Flask(__name__)

# Load model and data resources
# Load model and data resources
model = None
df = None

def load_resources():
    global model, df, most_common_seasonality
    try:
        model = joblib.load(BEST_MODEL_PATH)
        df = pd.read_csv(PROCESSED_DATA_PATH)
        
        most_common_seasonality = df['seasonality'].mode()[0]
        
        print("Model and resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        model = None
        global_mode_aircraft = df['aircraft_type'].mode()[0]
        most_common_seasonality = df['seasonality'].mode()[0]
        mean_base_fare = df['base_fare'].mean()
        mean_tax = df['tax_surcharge'].mean()
        
        print("Model and resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        model = None

load_resources()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        if not data:
             return jsonify({"error": "No input data provided"}), 400

        # Expecting a list of dicts or a single dict
        if isinstance(data, dict):
            input_data = [data]
        else:
            input_data = data
            
        input_df = pd.DataFrame(input_data)
        
        # Required fields check
        required_fields = ['airline', 'source_name', 'destination_name', 'date', 'stopovers', 'class']
        missing = [field for field in required_fields if field not in input_df.columns]
        if missing:
             return jsonify({"error": f"Missing required fields: {missing}"}), 400

        # 1. Date Features
        input_df['date'] = pd.to_datetime(input_df['date'], dayfirst=False)
        input_df['month'] = input_df['date'].dt.month
        input_df['day'] = input_df['date'].dt.day
        input_df['weekday'] = input_df['date'].dt.day_name()
        
        # Days before departure (default to 7 if not provided or negative)
        if 'days_before_departure' not in input_df.columns:
             input_df['days_before_departure'] = (input_df['date'] - pd.Timestamp.now()).dt.days
             input_df['days_before_departure'] = input_df['days_before_departure'].apply(lambda x: x if x >= 0 else 7)

        # Season mapping
        season_map = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Summer", 4: "Summer", 5: "Summer",
            6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
            10: "Autumn", 11: "Autumn",
        }
        input_df['season'] = input_df['month'].map(season_map)
        
        # 2. Impute Defaults for Missing Optional Fields
        if 'booking_source' not in input_df.columns:
            input_df['booking_source'] = "Online Website" # Default
            
        if 'duration_hrs' not in input_df.columns:
             input_df['duration_hrs'] = 2.0 # Default

        input_df['seasonality'] = most_common_seasonality
        
        # 3. Lookups - Removed as we expect source_name/destination_name directly or they are already present
        # input_df['source_name'] = input_df['source'].map(source_map).fillna(input_df['source'])
        # input_df['destination_name'] = input_df['destination'].map(dest_map).fillna(input_df['destination'])
        
        
        # 4. Impute Aircraft Type - Removed
        
        # 5. Impute Base Fare & Tax - Removed

        # 6. Arrival DateTime Dummy
        # We need this column to exist? Check preprocessor. 
        # If preprocessor drops it, fine. If it uses it (e.g. to extract time), we need it.
        # Assuming it might be used or passed through.
        input_df['arrival_date_and_time'] = input_df.apply(
            lambda x: (x['date'] + timedelta(hours=x['duration_hrs'])).strftime("%Y-%m-%d %H:%M:%S"), axis=1
        )
        
        # Predict (Log Transformed)
        log_prediction = model.predict(input_df)
        
        # Inverse Transform (Exp)
        prediction = np.exp(log_prediction)
        
        return jsonify({"prediction": prediction.tolist()})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
