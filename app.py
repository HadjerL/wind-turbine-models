from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import datetime
import joblib 
from statsmodels.tsa.seasonal import MSTL
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from itertools import combinations

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

        df = df[['Power_Output']].interpolate(method='linear')
        
        # Resample and fill any new missing values
        df = df.resample('h').mean()
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
        return jsonify({"error": f"{str(e)} data: {len(df)}"}), 500


MODEL_FILES = {
    "lstm": "LSTM_18_best_model.h5",
    "cnn": "CNN_18_best_model.h5",
    "rnn": "rnn_18_best_model.h5",
}

def load_all_models():
    models = {}
    for key, path in MODEL_FILES.items():
        models[key] = tf.keras.models.load_model(path)
    return models

# Load models (you can preload or load on-demand)
models = load_all_models()

def prepare_evaluation_data(y_test, model, X_test, thresholds):
    y_test_np = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test
    y_pred_probs = model.predict(X_test)
    y_pred_np = (y_pred_probs > np.array(thresholds)).astype(int)
    # Add "normal" class where all zeros
    normal_class_pred = np.all(y_pred_np == 0, axis=1).reshape(-1,1)
    normal_class_test = np.all(y_test_np == 0, axis=1).reshape(-1,1)
    y_pred_np = np.hstack([y_pred_np, normal_class_pred])
    y_test_np = np.hstack([y_test_np, normal_class_test])
    return y_test_np, y_pred_np

def global_evaluation(y_test_np, y_pred_np, class_columns):
    accuracy = accuracy_score(y_test_np, y_pred_np)
    report = classification_report(y_test_np, y_pred_np, target_names=class_columns, digits=4, output_dict=True)
    return accuracy, report

def multi_label_evaluation(y_test_np, y_pred_np, class_columns):
    accuracy, report = global_evaluation(y_test_np, y_pred_np, class_columns)
    return {"accuracy": accuracy, "classification_report": report}

def class_pair_evaluation(y_test_np, y_pred_np, class_columns):
    pairs = list(combinations(range(len(class_columns)), 2))
    metrics = []

    two_active_mask = np.sum(y_test_np, axis=1) == 2
    if np.any(two_active_mask):
        accuracy = accuracy_score(y_test_np[two_active_mask], y_pred_np[two_active_mask])
    else:
        accuracy = None

    for i, j in pairs:
        pair_name = f"{class_columns[i]} & {class_columns[j]}"
        mask = (y_test_np[:, i] == 1) & (y_test_np[:, j] == 1)
        support = np.sum(mask)

        if support > 0:
            y_true_pair = y_test_np[mask][:, [i, j]]
            y_pred_pair = y_pred_np[mask][:, [i, j]]

            precision = precision_score(y_true_pair, y_pred_pair, average='samples', zero_division=0)
            recall = recall_score(y_true_pair, y_pred_pair, average='samples', zero_division=0)
            f1 = f1_score(y_true_pair, y_pred_pair, average='samples', zero_division=0)

            metrics.append({"pair": pair_name, "precision": precision, "recall": recall, "f1_score": f1, "support": support})

    return {"accuracy_two_active": accuracy, "pair_metrics": metrics}

def evaluate_single_class(y_test_np, y_pred_np, class_columns):
    normal_idx = class_columns.index("normal")
    mask = (y_test_np.sum(axis=1) == 1) & (y_test_np[:, normal_idx] == 0)

    y_filtered = y_test_np[mask]
    if len(y_filtered) == 0:
        return {"message": "No samples with exactly one active class found."}

    y_pred_filtered = y_pred_np[mask]

    accuracy, report = global_evaluation(y_filtered, y_pred_filtered, class_columns)
    return {"accuracy": accuracy, "classification_report": report}

def evaluate_normal_vs_abnormal(y_test_np, y_pred_np, class_columns):
    normal_idx = class_columns.index("normal")
    y_test_binary = (y_test_np[:, normal_idx] == 1).astype(int)
    y_pred_binary = (y_pred_np[:, normal_idx] == 1).astype(int)

    accuracy, report = global_evaluation(y_test_binary, y_pred_binary, ["Abnormal", "Normal"])
    return {"accuracy": accuracy, "classification_report": report}

def evaluate_model(model, X_test, y_test, class_columns, thresholds):
    y_test_np, y_pred_np = prepare_evaluation_data(y_test, model, X_test, thresholds)
    class_columns_with_normal = class_columns + ["normal"]

    return {
        "multi_label_evaluation": multi_label_evaluation(y_test_np, y_pred_np, class_columns_with_normal),
        "class_pair_evaluation": class_pair_evaluation(y_test_np, y_pred_np, class_columns_with_normal),
        "evaluate_single_class": evaluate_single_class(y_test_np, y_pred_np, class_columns_with_normal),
        "evaluate_normal_vs_abnormal": evaluate_normal_vs_abnormal(y_test_np, y_pred_np, class_columns_with_normal)
    }


tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()


@app.route('/tune_models', methods=['POST'])
def admin_tune_models():
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input data must be a non-empty array"}), 400

        df = pd.DataFrame(data)

        if "Timestamp" not in df.columns:
            return jsonify({"error": "Missing Timestamp"}), 400

        if "Month" not in df.columns:
            df["Month"] = pd.to_datetime(df["Timestamp"]).dt.month

        missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing required feature columns: {missing_features}"}), 400

        missing_labels = [col for col in CLASS_LABELS if col not in df.columns]
        if missing_labels:
            return jsonify({"error": f"Missing fault label columns: {missing_labels}"}), 400

        if "Asset_ID" in df.columns:
            df = df.drop("Asset_ID", axis=1)

        df = df.drop("Timestamp", axis=1)

        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = df[CLASS_LABELS].values.astype(int)
        X = np.expand_dims(X, axis=1)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        epochs = 1
        batch_size = 32

        evaluation_results = {}

        for model_name, model in models.items():
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

            model.save(f'{model_name}_tuned.h5')

            thresholds = [0.5] * len(CLASS_LABELS)
            evaluation_results[model_name] = evaluate_model(model, X_test, y_test, CLASS_LABELS, thresholds)

        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        cleaned_evaluation_results = convert_numpy_types(evaluation_results)

        return jsonify({"message": "Models tuned successfully", "evaluation": cleaned_evaluation_results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
