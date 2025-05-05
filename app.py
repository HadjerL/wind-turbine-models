from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import datetime
import joblib 
from statsmodels.tsa.seasonal import MSTL

app = Flask(__name__)
CORS(app) 

MODEL_PATH = "lstm_multilabel_anomalie_normal_model.h5"
SCALER_PATH = "scaler.pkl"
model = tf.keras.models.load_model(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)
# scaler = StandardScaler()


FORECAST_MODEL_PATH = "MSTL_LSTM_48_model.h5"
forecast_model = tf.keras.models.load_model(
    FORECAST_MODEL_PATH, 
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
)

FORECAST_WINDOW = 48  # Look-back window
FORECAST_HORIZON = 24  # Prediction horizon
MINIMUM_ROWS = 576 / 2 # Minimum rows for 24 hours of data at 1h intervals

FEATURE_COLUMNS = [
    "Status_Type", "Rotor_Speed", "Rotational_Speed",
    "Gearbox_Oil_Inlet_Temperature", "RMS_Current_Phase_1_HV_Grid",
    "RMS_Current_Phase_2_HV_Grid", "RMS_Current_Phase_3_HV_Grid",
    "RMS_Voltage_Phase_1_HV_Grid", "RMS_Voltage_Phase_2_HV_Grid",
    "RMS_Voltage_Phase_3_HV_Grid", "Min_Pitch_Angle",
    "Rotor_Bearing_Temperature", "Outside_Temperature",
    "Wind_Speed", "Power_Output", "Wind_Direction_Sin", 
    "Wind_Direction_Cos", "Month"  # Missing feature extracted from Timestamp
]

CLASS_LABELS = [
    'Communication', 'Electrical system', 'Gearbox',
    'Hydraulic system', 'Pitch system', 'Yaw system', 'other'
]

def add_mstl_components(data, periods=[12, 24]):
    """Add MSTL decomposition components to data"""
    series = data['Power_Output'].copy()
    mstl = MSTL(series, periods=periods)
    result = mstl.fit()

    data['trend'] = result.trend
    seasonal_components = result.seasonal

    for i, period in enumerate(periods):
        col_name = f'seasonal_{period}'
        data[col_name] = seasonal_components.iloc[:, i] if isinstance(seasonal_components, pd.DataFrame) else seasonal_components[:, i]

    data['residual'] = result.resid
    return data

def prepare_forecast_data(data):
    """Prepare data for forecasting model"""
    # Ensure Timestamp is the index
    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"])
        data = data.set_index("Timestamp")
    
    # Resample to hourly and select Power_Output
    data = data.resample('h').mean()
    data = data[['Power_Output']]
    
    # Add MSTL components
    data = add_mstl_components(data)
    return data

def create_forecast_sequence(data, look_back=FORECAST_WINDOW):
    """Create input sequence for forecasting"""
    if len(data) < look_back:
        raise ValueError(f"Need at least {look_back} hours of historical data")
    
    # Get the most recent window
    sequence = data.iloc[-look_back:].values
    return np.array([sequence])  # Add batch dimension


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input data must be a non-empty array"}), 400
        
        # Validate row count for anomaly detection
        if len(data) < MINIMUM_ROWS:
            return jsonify({
                "error": f"Insufficient data. Requires at least {MINIMUM_ROWS} rows. Received {len(data)}."
            }), 400
        
        df = pd.DataFrame(data)
        scaler = joblib.load(SCALER_PATH)
        # Ensure Timestamp is present
        if "Timestamp" not in df.columns:
            return jsonify({"error": "Missing Timestamp"}), 400

        # Extract "Month" from Timestamp
        df["Month"] = pd.to_datetime(df["Timestamp"]).dt.month

        # Ensure all required features exist
        if not all(col in df.columns for col in FEATURE_COLUMNS):
            return jsonify({"error": "Missing required features"}), 400

        # Select only the relevant feature columns
        input_data = df[FEATURE_COLUMNS].copy()


        # Scale the numerical features
        numerical_columns = [col for col in FEATURE_COLUMNS if col not in ["Status_Type, Asset_ID", "Timestamp"]]  # Exclude categorical columns

        # Fit and transform the numerical columns only (scaling)
        input_data[numerical_columns] = scaler.fit_transform(input_data[numerical_columns])

        # Convert to the correct shape for the model
        input_data = input_data.values.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=1)  # Reshape to (batch_size, 1, features)

        # Make predictions
        predictions = model.predict(input_data)

        # Prepare results
        results = []
        for i, pred in enumerate(predictions):
            predicted_classes = [CLASS_LABELS[j] for j, value in enumerate(pred) if value > 0.99]  # Adjust threshold
            results.append({
                "Predicted_Classes": predicted_classes if predicted_classes else ["Normal"]
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_forecast', methods=['POST'])
def predict_forecast():
    """Endpoint for power output forecasting with proper scaling workflow"""
    try:
        # Get and validate input data
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input data must be a non-empty array"}), 400
        
        if len(data) < MINIMUM_ROWS:
            return jsonify({
                "error": f"Insufficient data. Requires at least {MINIMUM_ROWS} rows. Received {len(data)}."
            }), 400
        # scaler = joblib.load(SCALER_PATH)
        scaler = StandardScaler()
        df = pd.DataFrame(data)
        
        # Initial preparation (without MSTL yet)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df = df.set_index("Timestamp")
        df = df[['Power_Output']]
        df = df.resample('h').mean()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df = df.asfreq('h')
        df = df.interpolate(method='linear')
        
        # Create dummy DataFrame for scaling
        dummy_df = pd.DataFrame(columns=FEATURE_COLUMNS)
        dummy_df['Power_Output'] = df['Power_Output']
        for col in [c for c in FEATURE_COLUMNS if c not in ['Status_Type', 'Month']]:
            if col not in dummy_df.columns:
                dummy_df[col] = 0
        
        # Scale the data (only Power_Output will have real values)
        scaled_data = scaler.fit_transform(dummy_df)
        df['Power_Output'] = scaled_data[:, FEATURE_COLUMNS.index('Power_Output')]
        
        # NOW add MSTL components to the SCALED data
        prepared_data = add_mstl_components(df)
        
        # Create input sequence
        input_sequence = create_forecast_sequence(prepared_data)
        
        # Make prediction
        forecast_scaled = forecast_model.predict(input_sequence)
        
        # Inverse transform predictions
        inverse_dummy = np.zeros((len(forecast_scaled[0]), len(FEATURE_COLUMNS)))
        power_output_idx = FEATURE_COLUMNS.index('Power_Output')
        inverse_dummy[:, power_output_idx] = forecast_scaled[0]
        forecast_power = scaler.inverse_transform(inverse_dummy)[:, power_output_idx]
        
        # Generate timestamps
        last_timestamp = prepared_data.index[-1]
        forecast_timestamps = [last_timestamp + datetime.timedelta(hours=i+1) for i in range(FORECAST_HORIZON)]
        
        return jsonify([{
            "timestamp": ts.isoformat(),
            "power_output": float(value)
        } for ts, value in zip(forecast_timestamps, forecast_power)])
        
    except Exception as e:
        return jsonify({"error": f"{str(e)} data: {len(data)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
