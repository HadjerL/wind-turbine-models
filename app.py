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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import glob
from tensorflow import keras
import os
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import (
    Conv1D, Flatten, Dense, LSTM, SimpleRNN, Dropout, GlobalAveragePooling1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
import keras_tuner as kt

app = Flask(__name__)
CORS(app) 


def get_model_dicts():
    """Get dictionaries of all available models with directory name as prefix"""
    classification_models = {}
    forecasting_models = {}
    
    # Scan classification models
    for model_dir in os.listdir(CLASSIFICATION_MODEL_DIR):
        model_path = os.path.join(CLASSIFICATION_MODEL_DIR, model_dir)
        if os.path.isdir(model_path):
            for file in os.listdir(model_path):
                if file.endswith('.h5'):
                    # Combine directory name with version
                    model_name = f"{model_dir}_{file.replace('.h5', '')}"
                    classification_models[model_name] = os.path.join(model_path, file)
    
    # Scan forecasting models
    for model_dir in os.listdir(FORECAST_MODEL_DIR):
        model_path = os.path.join(FORECAST_MODEL_DIR, model_dir)
        if os.path.isdir(model_path):
            for file in os.listdir(model_path):
                if file.endswith('.h5'):
                    model_name = f"{model_dir}_{file.replace('.h5', '')}"
                    forecasting_models[model_name] = os.path.join(model_path, file)
    
    return classification_models, forecasting_models

def get_current_model_path(directory):
    """Find and return the path of the model file with '(current)' in its name"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if '(current)' in file.lower() and file.endswith('.h5'):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No current model found in {directory}")


#-------------------------------------------------------------------------------------------------------------------
#---------------------Load Models and Constants---------------------------------------------------

CLASSIFICATION_MODEL_DIR = "models/classification"
MODEL_PATH = get_current_model_path(CLASSIFICATION_MODEL_DIR)
model = tf.keras.models.load_model(MODEL_PATH)

# Load forecasting model
FORECAST_MODEL_DIR = "models/forecasting"
FORECAST_MODEL_PATH = get_current_model_path(FORECAST_MODEL_DIR)
forecast_model = tf.keras.models.load_model(
    FORECAST_MODEL_PATH, 
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
)

SCALER_PATH = "scaler.pkl"
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

# Generate the dictionaries dynamically
MODEL_FILES, FORECAST_MODEL_FILES = get_model_dicts()

TUNER_DIR = "tuner"
#-------------------------------------------------------------------------------------------------------------------
#---------------------Data preparation---------------------------------------------------
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




#-------------------------------------------------------------------------------------------------------------------
#---------------------Classification Functions---------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        MODEL_PATH = get_current_model_path(CLASSIFICATION_MODEL_DIR)
        model = tf.keras.models.load_model(MODEL_PATH)

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
            predicted_classes = [CLASS_LABELS[j] for j, value in enumerate(pred) if value > 0.5]  # Adjust threshold
            results.append({
                "Predicted_Classes": predicted_classes if predicted_classes else ["Normal"]
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500





#-------------------------------------------------------------------------------------------------------------------
#---------------------Forecasting Functions---------------------------------------------------
@app.route('/predict_forecast', methods=['POST'])
def predict_forecast():
    """Endpoint for power output forecasting with proper scaling workflow"""
    try:
        FORECAST_MODEL_PATH = get_current_model_path(FORECAST_MODEL_DIR)
        forecast_model = tf.keras.models.load_model(
            FORECAST_MODEL_PATH, 
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
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




#-------------------------------------------------------------------------------------------------------------------
#---------------------Classification Evaluation Functions---------------------------------------------------

# MODEL_FILES = {
#     "lstm": "models/classification/lstm.h5",
#     "cnn": "models/classification/cnn(current).h5",
#     "rnn": "models/classification/rnn.h5",
# }

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


#---------------------------------------------Classification Evaluation Endpoint---------------------------------------------------
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
        new_versions = {}  # Will store only the newly created versions
        
        # First find the latest version of each model type
        latest_versions = {}
        for model_type in os.listdir(CLASSIFICATION_MODEL_DIR):
            model_dir = os.path.join(CLASSIFICATION_MODEL_DIR, model_type)
            if os.path.isdir(model_dir):
                versions = []
                for f in os.listdir(model_dir):
                    if f.startswith('v') and f.endswith('.h5'):
                        try:
                            version_num = int(f[1:-3])
                            versions.append((version_num, f))
                        except ValueError:
                            continue
                if versions:
                    latest_version = max(versions, key=lambda x: x[0])
                    latest_versions[model_type] = latest_version[1]  # 'v2.h5'

        for model_type, version_file in latest_versions.items():
            try:
                model_path = os.path.join(CLASSIFICATION_MODEL_DIR, model_type, version_file)
                model = tf.keras.models.load_model(model_path)
                
                # Extract version info
                version_num = int(version_file[1:-3])
                new_version_num = version_num + 1
                new_version_name = f"v{new_version_num}.h5"
                
                # Clone the model to avoid modifying the original
                model_clone = tf.keras.models.clone_model(model)
                model_clone.set_weights(model.get_weights())
                
                # Compile and train
                model_clone.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                history = model_clone.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
                
                # Save new version
                model_dir = os.path.join(CLASSIFICATION_MODEL_DIR, model_type)
                os.makedirs(model_dir, exist_ok=True)
                new_model_path = os.path.join(model_dir, new_version_name)
                model_clone.save(new_model_path)
                
                # Evaluate the model
                thresholds = [0.5] * len(CLASS_LABELS)
                response_key = f"{model_type}_v{new_version_num}"
                new_versions[response_key] = {
                    **evaluate_model(model_clone, X_test, y_test, CLASS_LABELS, thresholds),
                    "original_version": f"{model_type}_{version_file.replace('.h5', '')}",
                    "saved_path": new_model_path
                }
                
                
            except Exception as e:
                evaluation_results[f"{model_type}_error"] = {"error": str(e)}
                continue

        evaluation_results.update(new_versions)

        # Evaluate the current model without tuning
        try:
            current_model_path = get_current_model_path(CLASSIFICATION_MODEL_DIR)
            if current_model_path:
                model_type = os.path.basename(os.path.dirname(current_model_path))
                model_name = os.path.basename(current_model_path).replace('.h5', '')
                
                current_model = tf.keras.models.load_model(current_model_path)
                thresholds = [0.5] * len(CLASS_LABELS)
                evaluation_results[f"{model_type}_current"] = {
                    **evaluate_model(current_model, X_test, y_test, CLASS_LABELS, thresholds),
                    "note": "Original current model (not fine-tuned)"
                }
        except Exception as e:
            evaluation_results["current_model"] = {"error": str(e)}

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

        return jsonify({
            "message": "Latest models tuned successfully",
            "evaluation": cleaned_evaluation_results,
            "new_versions": list(new_versions.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500





#-------------------------------------------------------------------------------------------------------------------
#---------------------Forecasting Evaluation Functions---------------------------------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    """Compute Mean Absolute Percentage Error (MAPE)."""
    mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_forecast(y_actual, y_pred):
    """Evaluate forecast performance across ALL horizon steps combined."""
    y_true_all = y_actual.ravel()
    y_pred_all = y_pred.ravel()

    metrics = {
        'mse': mean_squared_error(y_true_all, y_pred_all),
        'rmse': np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
        'mae': mean_absolute_error(y_true_all, y_pred_all),
        'r2': r2_score(y_true_all, y_pred_all),
        'mape': mean_absolute_percentage_error(y_true_all, y_pred_all),
        'mean_test': np.mean(y_true_all)
    }
    return metrics

def evaluate_forecast_by_step(y_actual, y_pred):
    """Evaluate forecast performance for each horizon step."""
    horizon_steps = y_actual.shape[1]
    metrics = []

    for step in range(horizon_steps):
        y_true_step = y_actual[:, step]
        y_pred_step = y_pred[:, step]

        metrics.append({
            'horizon': step + 1,
            'mse': mean_squared_error(y_true_step, y_pred_step),
            'rmse': np.sqrt(mean_squared_error(y_true_step, y_pred_step)),
            'mae': mean_absolute_error(y_true_step, y_pred_step),
            'r2': r2_score(y_true_step, y_pred_step),
            'mape': mean_absolute_percentage_error(y_true_step, y_pred_step)
        })

    return metrics

# FORECAST_MODEL_FILES = {
#     "lstm": "models/forecasting/lstm(current).h5",
#     "cnn": "models/forecasting/cnn.h5", 
# }

def load_all_forecast_models():
    models = {}
    for key, path in FORECAST_MODEL_FILES.items():
        models[key] = tf.keras.models.load_model(
            path, 
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
    return models

# Load forecast models
forecast_models = load_all_forecast_models()

def create_dataset(data, look_back=48, forecast_horizon=24, target_col="Power_Output"):
    """Create sequences for forecasting"""
    X, y = [], []
    feature_cols = data.columns
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data.iloc[i:(i+look_back)][feature_cols].values)
        y.append(data.iloc[(i+look_back):(i+look_back+forecast_horizon)][target_col].values)
    return np.array(X), np.array(y)


@app.route('/tune_forecast', methods=['POST'])
def tune_forecast():
    """Endpoint for tuning forecasting models with version management"""
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input data must be a non-empty array"}), 400

        df = pd.DataFrame(data)
        
        # Validate input
        if "Timestamp" not in df.columns:
            return jsonify({"error": "Missing Timestamp"}), 400
            
        if "Power_Output" not in df.columns:
            return jsonify({"error": "Missing Power_Output column"}), 400

        # Prepare data
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp")
        df = df[['Power_Output']].interpolate(method='linear')
        df = df.resample('h').mean()
        df = df.interpolate(method='linear')
        
        # Add MSTL components (no scaling yet)
        prepared_data = add_mstl_components(df)
        
        # Create a list of only the features we'll use for forecasting
        forecast_features = ['Power_Output', 'trend', 'residual', 'seasonal_12', 'seasonal_24']
        
        # Scale only these specific features
        scaler = StandardScaler()
        prepared_data[forecast_features] = scaler.fit_transform(prepared_data[forecast_features])
        
        # Split data
        train_size = int(len(prepared_data) * 0.8)
        train, test = prepared_data.iloc[:train_size], prepared_data.iloc[train_size:]
        
        # Create sequences using only the forecast features
        X_train, y_train = create_dataset(train[forecast_features], FORECAST_WINDOW, FORECAST_HORIZON)
        X_test, y_test = create_dataset(test[forecast_features], FORECAST_WINDOW, FORECAST_HORIZON)
        
        # Get timestamps
        test_timestamps = test.index[FORECAST_WINDOW:-FORECAST_HORIZON]
        
        epochs = 1
        batch_size = 32
        
        evaluation_results = {}
        new_versions = {}  # Will store only the newly created versions
        
        # First find the latest version of each model type
        latest_versions = {}
        for model_type in os.listdir(FORECAST_MODEL_DIR):
            model_dir = os.path.join(FORECAST_MODEL_DIR, model_type)
            if os.path.isdir(model_dir):
                versions = []
                for f in os.listdir(model_dir):
                    if f.startswith('v') and f.endswith('.h5'):
                        try:
                            version_num = int(f[1:-3])
                            versions.append((version_num, f))
                        except ValueError:
                            continue
                if versions:
                    latest_version = max(versions, key=lambda x: x[0])
                    latest_versions[model_type] = latest_version[1]  # 'v2.h5'

        # Fine-tune each model with version management
        for model_type, version_file in latest_versions.items():
            try:
                model_path = os.path.join(FORECAST_MODEL_DIR, model_type, version_file)
                model = tf.keras.models.load_model(
                    model_path,
                    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
                )

                # Extract version info
                version_num = int(version_file[1:-3])
                new_version_num = version_num + 1
                new_version_name = f"v{new_version_num}.h5"

                # Clone the model to avoid modifying the original
                model_clone = tf.keras.models.clone_model(model)
                model_clone.set_weights(model.get_weights())
                
                # Compile and train
                model_clone.compile(optimizer='adam', loss='mse', metrics=['mae'])
                history = model_clone.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    verbose=1
                )
                
                # Save new version
                model_dir = os.path.join(FORECAST_MODEL_DIR, model_type)
                os.makedirs(model_dir, exist_ok=True)
                new_model_path = os.path.join(model_dir, new_version_name)
                model_clone.save(new_model_path)
                
                # Evaluate
                y_pred = model_clone.predict(X_test)
                
                # Inverse transform all at once for proper scaling
                def inverse_transform_predictions(predictions):
                    # Create array with same shape as original features
                    dummy = np.zeros((predictions.shape[0] * predictions.shape[1], len(forecast_features)))
                    # Only fill the Power_Output column
                    dummy[:, 0] = predictions.ravel()
                    # Inverse transform
                    inv = scaler.inverse_transform(dummy)
                    # Reshape back to original prediction shape
                    return inv[:, 0].reshape(predictions.shape)
                
                y_test_inv = inverse_transform_predictions(y_test)
                y_pred_inv = inverse_transform_predictions(y_pred)
                
                # Calculate metrics
                overall_metrics = evaluate_forecast(y_test_inv, y_pred_inv)
                step_metrics = evaluate_forecast_by_step(y_test_inv, y_pred_inv)
                
                # Prepare forecast visualization data
                visualization_data = []
                for i in range(min(len(X_test), 5)):  # Limit to 5 examples
                    input_start = test_timestamps[i]
                    input_timestamps = [input_start + pd.Timedelta(hours=j) for j in range(FORECAST_WINDOW)]
                    
                    true_start = input_start + pd.Timedelta(hours=FORECAST_WINDOW)
                    true_timestamps = [true_start + pd.Timedelta(hours=j) for j in range(FORECAST_HORIZON)]
                    
                    # Get the power output values (index 0 in our features)
                    input_values = scaler.inverse_transform(X_test[i])[:, 0].tolist()
                    
                    visualization_data.append({
                        "input_sequence": {
                            "timestamps": [ts.isoformat() for ts in input_timestamps],
                            "values": input_values
                        },
                        "true_values": {
                            "timestamps": [ts.isoformat() for ts in true_timestamps],
                            "values": y_test_inv[i].tolist()
                        },
                        "predicted_values": {
                            "timestamps": [ts.isoformat() for ts in true_timestamps],
                            "values": y_pred_inv[i].tolist()
                        }
                    })
                
                response_key = f"{model_type}_v{new_version_num}"
                new_versions[response_key] = {
                    "overall_metrics": overall_metrics,
                    "step_metrics": step_metrics,
                    "visualization_data": visualization_data,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "original_version": f"{model_type}_{version_file.replace('.h5', '')}",
                    "saved_path": new_model_path
                }
                
            except Exception as e:
                evaluation_results[f"{model_type}_error"] = {"error": str(e)}
                continue
        
        # Only include newly created versions in results
        evaluation_results.update(new_versions)
        
        # Evaluate the current LSTM model without tuning
        try:
            current_model_path = get_current_model_path(FORECAST_MODEL_DIR)
            if current_model_path:
                model_type = os.path.basename(os.path.dirname(current_model_path))
                model_name = os.path.basename(current_model_path).replace('.h5', '')
                
                current_model = tf.keras.models.load_model(
                    current_model_path,
                    custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
                )
                
                y_pred_current = current_model.predict(X_test)
                y_pred_current_inv = inverse_transform_predictions(y_pred_current)
            
            # Calculate metrics for current model
            overall_metrics_current = evaluate_forecast(y_test_inv, y_pred_current_inv)
            step_metrics_current = evaluate_forecast_by_step(y_test_inv, y_pred_current_inv)
            
            # Prepare visualization data for current model
            visualization_data_current = []
            for i in range(min(len(X_test), 5)):
                input_start = test_timestamps[i]
                input_timestamps = [input_start + pd.Timedelta(hours=j) for j in range(FORECAST_WINDOW)]
                
                true_start = input_start + pd.Timedelta(hours=FORECAST_WINDOW)
                true_timestamps = [true_start + pd.Timedelta(hours=j) for j in range(FORECAST_HORIZON)]
                
                input_values = scaler.inverse_transform(X_test[i])[:, 0].tolist()
                
                visualization_data_current.append({
                    "input_sequence": {
                        "timestamps": [ts.isoformat() for ts in input_timestamps],
                        "values": input_values
                    },
                    "true_values": {
                        "timestamps": [ts.isoformat() for ts in true_timestamps],
                        "values": y_test_inv[i].tolist()
                    },
                    "predicted_values": {
                        "timestamps": [ts.isoformat() for ts in true_timestamps],
                        "values": y_pred_current_inv[i].tolist()
                    }
                })
            
            evaluation_results[f"{model_type}_current"] = {
                "overall_metrics": evaluate_forecast(y_test_inv, y_pred_current_inv),
                "step_metrics": evaluate_forecast_by_step(y_test_inv, y_pred_current_inv),
                "visualization_data": visualization_data_current,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "note": "Original current model (not fine-tuned)"
            }
            
        except Exception as e:
            evaluation_results["current_model_error"] = {"error": str(e)}

        return jsonify({
            "message": "Latest models tuned successfully",
            "results": evaluation_results,
            "forecast_horizon": FORECAST_HORIZON,
            "new_versions": list(new_versions.keys())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


#--------------------------------------------------------------------------------------------------------------------
#---------------------Evalution---------------------------------------------------------------


@app.route('/evaluate_classification_model/<model_name>/<architecture>/<version>', methods=['POST'])
def evaluate_classification_model(model_name, architecture, version):
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input data must be a non-empty array"}), 400

        df = pd.DataFrame(data)

        # Validate input
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

        # Load the specific model version
        model_path = os.path.join(CLASSIFICATION_MODEL_DIR, architecture, model_name, f"v{version}.h5")
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model {model_path} {architecture} {model_name} version {version} not found"}), 404

        model = tf.keras.models.load_model(model_path)
        
        # Evaluate the model
        thresholds = [0.5] * len(CLASS_LABELS)
        model_key = f"{model_name}_{architecture}_v{version}"
        
        response_data = {
            "evaluation": evaluate_model(model, X_test, y_test, CLASS_LABELS, thresholds),
            "message": "Model evaluated successfully"
        }

        # Convert numpy types to native Python types
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

        cleaned_response = convert_numpy_types(response_data)

        return jsonify(cleaned_response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/evaluate_forecasting_model/<architecture>/<model_name>/<version>', methods=['POST'])
def evaluate_forecasting_model(architecture, model_name, version):
    try:
        data = request.json
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input data must be a non-empty array"}), 400

        df = pd.DataFrame(data)
        
        # Validate input
        if "Timestamp" not in df.columns:
            return jsonify({"error": "Missing Timestamp"}), 400
            
        if "Power_Output" not in df.columns:
            return jsonify({"error": "Missing Power_Output column"}), 400

        # Prepare data
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp")
        df = df[['Power_Output']].interpolate(method='linear')
        df = df.resample('h').mean()
        df = df.interpolate(method='linear')
        
        # Add MSTL components (no scaling yet)
        prepared_data = add_mstl_components(df)
        
        # Create a list of only the features we'll use for forecasting
        forecast_features = ['Power_Output', 'trend', 'residual', 'seasonal_12', 'seasonal_24']
        
        # Scale only these specific features
        scaler = StandardScaler()
        prepared_data[forecast_features] = scaler.fit_transform(prepared_data[forecast_features])
        
        # Split data
        train_size = int(len(prepared_data) * 0.8)
        train, test = prepared_data.iloc[:train_size], prepared_data.iloc[train_size:]
        
        # Create sequences using only the forecast features
        X_test, y_test = create_dataset(test[forecast_features], FORECAST_WINDOW, FORECAST_HORIZON)
        
        # Get timestamps
        test_timestamps = test.index[FORECAST_WINDOW:-FORECAST_HORIZON]
        
        # Load the specific model version
        model_path = os.path.join(FORECAST_MODEL_DIR, model_name, architecture, f"v{version}.h5")
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model {model_name} version {version} not found"}), 404

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Inverse transform all at once for proper scaling
        def inverse_transform_predictions(predictions):
            # Create array with same shape as original features
            dummy = np.zeros((predictions.shape[0] * predictions.shape[1], len(forecast_features)))
            # Only fill the Power_Output column
            dummy[:, 0] = predictions.ravel()
            # Inverse transform
            inv = scaler.inverse_transform(dummy)
            # Reshape back to original prediction shape
            return inv[:, 0].reshape(predictions.shape)
        
        y_test_inv = inverse_transform_predictions(y_test)
        y_pred_inv = inverse_transform_predictions(y_pred)
        
        # Calculate metrics
        overall_metrics = evaluate_forecast(y_test_inv, y_pred_inv)
        step_metrics = evaluate_forecast_by_step(y_test_inv, y_pred_inv)
        
        # Prepare forecast visualization data
        visualization_data = []
        for i in range(min(len(X_test), 5)):  # Limit to 5 examples
            input_start = test_timestamps[i]
            input_timestamps = [input_start + pd.Timedelta(hours=j) for j in range(FORECAST_WINDOW)]
            
            true_start = input_start + pd.Timedelta(hours=FORECAST_WINDOW)
            true_timestamps = [true_start + pd.Timedelta(hours=j) for j in range(FORECAST_HORIZON)]
            
            # Get the power output values (index 0 in our features)
            input_values = scaler.inverse_transform(X_test[i])[:, 0].tolist()
            
            visualization_data.append({
                "input_sequence": {
                    "timestamps": [ts.isoformat() for ts in input_timestamps],
                    "values": input_values
                },
                "true_values": {
                    "timestamps": [ts.isoformat() for ts in true_timestamps],
                    "values": y_test_inv[i].tolist()
                },
                "predicted_values": {
                    "timestamps": [ts.isoformat() for ts in true_timestamps],
                    "values": y_pred_inv[i].tolist()
                }
            })
        
        evaluation_results = {
                "overall_metrics": overall_metrics,
                "step_metrics": step_metrics,
                "visualization_data": visualization_data,
                "test_samples": len(X_test),
                "original_version": f"{model_name}_v{version}",
            }

        return jsonify({
            "message": "Model evaluated successfully",
            "results": evaluation_results,
            "forecast_horizon": FORECAST_HORIZON
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#--------------------------------------------------------------------------------------------------------------------
#---------------------Get available classification models---------------------------------------------------

import os
import shutil

def get_model_info(directory):
    """Get all models in directory with architecture and version info"""
    model_data = {
        "model_type": "classification" if "classification" in directory else "forecasting",
        "architectures": []
    }
    
    # Get all model types (cnn, lstm, rnn)
    for model_type in os.listdir(directory):
        model_type_dir = os.path.join(directory, model_type)
        if not os.path.isdir(model_type_dir):
            continue
            
        # Get all architectures for this model type
        for archi in os.listdir(model_type_dir):
            archi_dir = os.path.join(model_type_dir, archi)
            if not os.path.isdir(archi_dir):
                continue
                
            archi_data = {
                "model_type": model_type,
                "architecture": archi,
                "versions": [],
                "current_version": None
            }
            
            # Get all versions in this architecture
            for version_file in os.listdir(archi_dir):
                if version_file.endswith('.h5'):
                    version_name = version_file.replace('.h5', '')
                    is_current = "(current)" in version_name.lower()
                    version_name = version_name.replace('(current)', '').strip()
                    
                    version_path = os.path.join(archi_dir, version_file)
                    creation_timestamp = os.path.getctime(version_path)
                    creation_date = datetime.datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M')
                    
                    version_data = {
                        "name": version_name,
                        "path": version_path,
                        "is_current": is_current,
                        "creation_date": creation_date,
                        "is_tuned": "_tuned" in version_file.lower()
                    }
                    
                    archi_data["versions"].append(version_data)
                    
                    if is_current:
                        archi_data["current_version"] = version_name
            
            # Sort versions by creation date (newest first)
            archi_data["versions"].sort(key=lambda x: x["creation_date"], reverse=True)
            model_data["architectures"].append(archi_data)
    
    return model_data


@app.route('/available_classification_models', methods=['GET'])
def get_available_classification_models():
    try:
        model_info = get_model_info(CLASSIFICATION_MODEL_DIR)
        return jsonify({
            "model_type": model_info["model_type"],
            "architectures": model_info["architectures"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/available_forecasting_models', methods=['GET'])
def get_available_forecasting_models():
    try:
        model_info = get_model_info(FORECAST_MODEL_DIR)
        return jsonify({
            "model_type": model_info["model_type"],
            "architectures": model_info["architectures"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_active_model/<model_type>/<architecture>/<model_name>/<version>', methods=['POST'])
def set_active_model(model_type, architecture, model_name, version):
    try:
        # Validate model type
        valid_types = ["classification", "forecasting"]
        if model_type not in valid_types:
            return jsonify({"error": f"Invalid model type. Must be one of: {valid_types}"}), 400
        
        # Set base directory
        base_dir = os.path.join("models", model_type)
        architecture_dir = os.path.join(base_dir, architecture)
        model_dir = os.path.join(architecture_dir, model_name)
        
        # Verify model directory exists
        if not os.path.exists(model_dir):
            return jsonify({"error": f"Model directory {architecture}/{model_name} not found"}), 404
        
        # Find the requested model file
        target_file = None
        for file in os.listdir(model_dir):
            file_version = file.split('(')[0].replace('.h5', '')  # Extract version from filename
            if version == file_version and file.endswith('.h5'):
                target_file = os.path.join(model_dir, file)
                break
        
        if not target_file:
            return jsonify({"error": f"Version {version} not found in {architecture}/{model_name}"}), 404
        
        # Remove (current) tag from ALL models in this model_type (entire classification/forecasting folder)
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if '(current)' in file and file.endswith('.h5'):
                    current_path = os.path.join(root, file)
                    new_name = file.replace('(current)', '')
                    os.rename(current_path, os.path.join(root, new_name))
        
        # Mark new model as current
        file_base = os.path.basename(target_file)
        if '(current)' not in file_base:
            new_name = file_base.replace('.h5', '(current).h5')
            new_path = os.path.join(model_dir, new_name)
            os.rename(target_file, new_path)
        else:
            new_path = target_file
        
        return jsonify({
            "message": f"{architecture}/{model_name} {version} set as active",
            "current_model": new_path,
            "model_type": model_type,
            "architecture": architecture
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to set active model: {str(e)}"}), 500



#--------------------------------------------------------------------------------------------------------------------
#---------------------Delete model endpoint---------------------------------------------------


@app.route('/delete_model/<model_type>/<model_name>/<architecture>/<version>', methods=['DELETE'])
def delete_model(model_type, architecture, model_name, version):
    try:
        # Validate model type
        valid_types = ["classification", "forecasting"]
        if model_type not in valid_types:
            return jsonify({
                "error": f"Invalid model type. Must be one of: {valid_types}",
                "status": 400
            }), 400
        
        # Set base directory with architecture
        base_dir = os.path.join("models", model_type)
        model_dir = os.path.join(base_dir, model_name)
        architecture_dir = os.path.join(model_dir, architecture)
        
        # Verify model directory exists
        if not os.path.exists(architecture_dir):
            return jsonify({
                "error": f"Model directory '{architecture}/{model_name}' not found",
                "status": 404
            }), 404
        
        # Find the exact version match (handles cases where version is subset of another)
        target_file = None
        for file in os.listdir(architecture_dir):
            # Extract clean version (handles v1, v1(current), etc.)
            file_version = file.split('(')[0].replace('.h5', '')
            if version == file_version and file.endswith('.h5'):
                target_file = os.path.join(architecture_dir, file)
                break
        
        if not target_file:
            return jsonify({
                "error": f"Version '{version}' not found in model '{model_name}'",
                "available_versions": [f.split('(')[0].replace('.h5', '') 
                                    for f in os.listdir(architecture_dir) if f.endswith('.h5')],
                "status": 404
            }), 404
        
        # Prevent deleting current model
        if '(current)' in os.path.basename(target_file):
            return jsonify({
                "error": "Cannot delete the active model. Set another model as active first.",
                "status": 400
            }), 400
        
        # Additional safety check - don't delete if this is the only model
        model_files = [f for f in os.listdir(architecture_dir) if f.endswith('.h5')]
        if len(model_files) <= 1:
            return jsonify({
                "error": "Cannot delete the only remaining model version",
                "status": 400
            }), 400
        
        # Delete the file
        os.remove(target_file)
        
        # If directory is empty, remove it (with additional checks)
        try:
            if not os.listdir(architecture_dir):
                os.rmdir(architecture_dir)
                return jsonify({
                    "message": f"Model '{model_name}' version '{version}' deleted and directory removed",
                    "deleted_path": target_file,
                    "status": 200
                })
        except OSError as e:
            # Directory not empty or other error
            pass
        
        return jsonify({
            "message": f"Model '{model_name}' version '{version}' deleted successfully",
            "deleted_path": target_file,
            "remaining_versions": [f.split('(')[0].replace('.h5', '') 
                                for f in os.listdir(model_dir) if f.endswith('.h5')],
            "status": 200
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to delete model: {str(e)}",
            "status": 500
        }), 500

@app.route('/train_new_model', methods=['POST'])
def train_new_model():
    try:
        # Parse request data
        request_data = request.json
        data = request_data.get('data')
        config = request_data.get('config')
        
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input data must be a non-empty array"}), 400

        # Prepare data
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

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create appropriate hypermodel based on config
        if config['modelType'] == 'CNN':
            hypermodel = CNNHyperModelCustom(config)
        elif config['modelType'] == 'LSTM':
            hypermodel = LSTMHyperModelCustom(config)
        elif config['modelType'] == 'RNN':
            hypermodel = RNNHyperModelCustom(config)
        else:
            return jsonify({"error": "Invalid model type"}), 400

        # Setup tuner
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective='val_loss',
            max_trials=config['maxTrials'],
            executions_per_trial=config['executionsPerTrial'],
            directory=TUNER_DIR,
            project_name=f"{config['modelType'].lower()}_custom_training",
            overwrite=True
        )

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=config['patience'], 
            restore_best_weights=True
        )

        # Run search
        tuner.search(
            X_train, y_train,
            epochs=config['epochs'],
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=2
        )

        # Get best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        
        # Train final model
        history = best_model.fit(
            X_train, y_train,
            epochs=config['epochs'],
            batch_size=best_hps.get('batch_size'),
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=2
        )

        # Create new architecture directory
        model_type = config['modelType'].lower()
        base_dir = os.path.join(CLASSIFICATION_MODEL_DIR, model_type)
        
        # Find next available architecture number
        existing_archis = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        archi_numbers = [int(d.replace('architecture', '')) for d in existing_archis if d.startswith('architecture')]
        next_archi_num = max(archi_numbers) + 1 if archi_numbers else 0
        archi_dir = os.path.join(base_dir, f"architecture{next_archi_num}")
        os.makedirs(archi_dir, exist_ok=True)

        # Save model as v0
        model_path = os.path.join(archi_dir, "v0.h5")
        best_model.save(model_path)
        
        # Evaluate model
        thresholds = [0.5] * len(CLASS_LABELS)
        evaluation_results = evaluate_model(best_model, X_test, y_test, CLASS_LABELS, thresholds)

        # Prepare hyperparameters response
        hyperparameters = {
            "model_type": config['modelType'],
            "architecture": f"architecture{next_archi_num}",
            "num_conv_layers": best_hps.values.get('num_conv_layers'),
            "conv1_filters": best_hps.values.get('conv1_filters'),
            "num_dense_layers": best_hps.values.get('num_dense_layers'),
            "dense1_units": best_hps.values.get('dense1_units'),
            "batch_size": best_hps.values.get('batch_size'),
            "optimizer": best_hps.values.get('optimizer'),
            "learning_rate": best_hps.values.get('learning_rate'),
            "lstm_units": best_hps.values.get('lstm1_units'),
            "num_lstm_layers": best_hps.values.get('num_lstm_layers'),
            "rnn_units": best_hps.values.get('rnn1_units'),
            "num_rnn_layers": best_hps.values.get('num_rnn_layers')
        }

        # Convert numpy types to native Python types
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

        cleaned_evaluation = convert_numpy_types(evaluation_results)

        return jsonify({
            "message": f"New {config['modelType']} model trained successfully",
            "evaluation": {
                f"{config['modelType']}_architecture{next_archi_num}": cleaned_evaluation
            },
            "hyperparameters": hyperparameters,
            "model_path": model_path,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


class CNNHyperModelCustom(kt.HyperModel):
    def __init__(self, config):
        self.config = config
    
    def build(self, hp):
        model = Sequential()
        
        # Conv layers
        num_conv_layers = hp.Choice('num_conv_layers', values=self.config.get('numConvLayers', [1, 2]))
        model.add(Conv1D(
            filters=hp.Choice('conv1_filters', values=self.config['convFilters']),
            kernel_size=1,
            activation="relu",
            input_shape=(1, 18)
        ))
        
        if num_conv_layers == 2:
            model.add(Conv1D(
                filters=hp.Choice('conv2_filters', values=self.config['convFilters']),
                kernel_size=1,
                activation="relu"
            ))
        
        model.add(Flatten())
        
        # Dense layers
        num_dense_layers = hp.Choice('num_dense_layers', values=self.config.get('numDenseLayers', [1, 2]))
        model.add(Dense(
            units=hp.Choice('dense1_units', values=self.config['denseUnits']),
            activation='relu'
        ))
        
        if num_dense_layers == 2:
            model.add(Dense(
                units=hp.Choice('dense2_units', values=self.config['denseUnits']),
                activation='relu'
            ))
        
        # Output layer
        model.add(Dense(len(CLASS_LABELS), activation="sigmoid"))
        
        # Optimizer
        optimizer = hp.Choice('optimizer', values=self.config['optimizers'])
        learning_rate = hp.Choice('learning_rate', values=self.config['learningRates'])
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
            
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[ThresholdedAccuracy(threshold=0.5)])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", self.config['batchSizes']),
            **kwargs,
        )

class ThresholdedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="thresholded_accuracy", **kwargs):
        super(ThresholdedAccuracy, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.int32)
        y_true = tf.cast(y_true, tf.int32)
        matches = tf.reduce_all(tf.equal(y_true, y_pred), axis=1)
        self.correct.assign_add(tf.reduce_sum(tf.cast(matches, tf.float32)))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class LSTMHyperModelCustom(kt.HyperModel):
    def __init__(self, config):
        self.config = config
    
    def build(self, hp):
        model = Sequential()
        
        # LSTM layers
        num_lstm_layers = hp.Choice('num_lstm_layers', values=self.config.get('numLSTMLayers', [1, 2]))
        model.add(LSTM(
            units=hp.Choice('lstm1_units', values=self.config['lstmUnits']),
            activation="tanh",
            return_sequences=(num_lstm_layers == 2),
            input_shape=(1, 18)
        ))
        
        if num_lstm_layers == 2:
            model.add(LSTM(
                units=hp.Choice('lstm2_units', values=self.config['lstmUnits']),
                activation="tanh"
            ))
        
        # Output layer
        model.add(Dense(len(CLASS_LABELS), activation="sigmoid"))
        
        # Optimizer
        optimizer = hp.Choice('optimizer', values=self.config['optimizers'])
        learning_rate = hp.Choice('learning_rate', values=self.config['learningRates'])
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
            
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[ThresholdedAccuracy(threshold=0.5)])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", self.config['batchSizes']),
            **kwargs,
        )


class RNNHyperModelCustom(kt.HyperModel):
    def __init__(self, config):
        self.config = config
    
    def build(self, hp):
        model = Sequential()
        
        # RNN layers
        num_rnn_layers = hp.Choice('num_rnn_layers', values=self.config.get('numRNNLayers', [1, 2]))
        model.add(SimpleRNN(
            units=hp.Choice('rnn1_units', values=self.config['rnnUnits']),
            activation="tanh",
            return_sequences=(num_rnn_layers == 2),
            input_shape=(1,18)
        ))
        
        if num_rnn_layers == 2:
            model.add(SimpleRNN(
                units=hp.Choice('rnn2_units', values=self.config['rnnUnits']),
                activation="tanh"
            ))
        
        # Output layer
        model.add(Dense(len(CLASS_LABELS), activation="sigmoid"))
        
        # Optimizer
        optimizer = hp.Choice('optimizer', values=self.config['optimizers'])
        learning_rate = hp.Choice('learning_rate', values=self.config['learningRates'])
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = SGD(learning_rate=learning_rate, momentum=0.9)
            
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[ThresholdedAccuracy(threshold=0.5)])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", self.config['batchSizes']),
            **kwargs,
        )

@app.route('/train_new_forecast_model', methods=['POST'])
def train_new_forecast_model():
    try:
        request_data = request.json
        data = request_data.get('data')
        config = request_data.get('config')
        
        if not data or not isinstance(data, list):
            return jsonify({"error": "Input data must be a non-empty array"}), 400

        df = pd.DataFrame(data)
        
        # Validate input
        if "Timestamp" not in df.columns:
            return jsonify({"error": "Missing Timestamp"}), 400
            
        if "Power_Output" not in df.columns:
            return jsonify({"error": "Missing Power_Output column"}), 400

        # Prepare data
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp")
        df = df[['Power_Output']].interpolate(method='linear')
        df = df.resample('h').mean()
        df = df.interpolate(method='linear')
        
        # Add MSTL components
        prepared_data = add_mstl_components(df)
        forecast_features = ['Power_Output', 'trend', 'residual', 'seasonal_12', 'seasonal_24']
        
        # Scale features
        scaler = StandardScaler()
        prepared_data[forecast_features] = scaler.fit_transform(prepared_data[forecast_features])
        
        # Split data
        train_size = int(len(prepared_data) * 0.8)
        train, test = prepared_data.iloc[:train_size], prepared_data.iloc[train_size:]
        
        # Create sequences
        X_train, y_train = create_dataset(train[forecast_features], FORECAST_WINDOW, FORECAST_HORIZON)
        X_test, y_test = create_dataset(test[forecast_features], FORECAST_WINDOW, FORECAST_HORIZON)
        test_timestamps = test.index[FORECAST_WINDOW:-FORECAST_HORIZON]

        # Create appropriate hypermodel based on config        
        model_type = config['modelType'].lower()

        if model_type == 'cnn':
            hypermodel = CNNForecastHyperModelCustom(config)
        elif model_type == 'lstm':
            hypermodel = LSTMHyperModelCustom(config)
        elif model_type == 'rnn':
            hypermodel = RNNHyperModelCustom(config)
        else:
            return jsonify({"error": f"Invalid model type {model_type}"}), 400

        # Setup tuner
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective='val_loss',
            max_trials=config.get('maxTrials', 30),
            executions_per_trial=config.get('executionsPerTrial', 1),
            directory=TUNER_DIR,
            project_name=f'{model_type}_forecast_tuning',
            overwrite=True
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.get('patience', 10),
            restore_best_weights=True
        )

        # Run search
        tuner.search(
            X_train, y_train,
            epochs=config.get('epochs', 100),
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=2
        )

        # Get best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        
        # Train final model
        history = best_model.fit(
            X_train, y_train,
            epochs=config.get('epochs', 100),
            batch_size=best_hps.get('batch_size'),
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=2
        )

        # Create new architecture directory
        base_dir = os.path.join(FORECAST_MODEL_DIR, model_type)
        os.makedirs(base_dir, exist_ok=True)
        
        # Find next available architecture number
        existing_archis = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        archi_numbers = [int(d.replace('architecture', '')) for d in existing_archis if d.startswith('architecture')]
        next_archi_num = max(archi_numbers) + 1 if archi_numbers else 0
        archi_dir = os.path.join(base_dir, f"architecture{next_archi_num}")
        os.makedirs(archi_dir, exist_ok=True)

        # Save model as v0
        model_path = os.path.join(archi_dir, "v0.h5")
        best_model.save(model_path)
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        
        # Inverse transform predictions
        def inverse_transform_predictions(predictions):
            dummy = np.zeros((predictions.shape[0] * predictions.shape[1], len(forecast_features)))
            dummy[:, 0] = predictions.ravel()
            inv = scaler.inverse_transform(dummy)
            return inv[:, 0].reshape(predictions.shape)
        
        y_test_inv = inverse_transform_predictions(y_test)
        y_pred_inv = inverse_transform_predictions(y_pred)
        
        # Calculate metrics
        overall_metrics = evaluate_forecast(y_test_inv, y_pred_inv)
        step_metrics = evaluate_forecast_by_step(y_test_inv, y_pred_inv)
        
        # Prepare visualization data
        visualization_data = []
        for i in range(min(len(X_test), 5)):
            input_start = test_timestamps[i]
            input_timestamps = [input_start + pd.Timedelta(hours=j) for j in range(FORECAST_WINDOW)]
            
            true_start = input_start + pd.Timedelta(hours=FORECAST_WINDOW)
            true_timestamps = [true_start + pd.Timedelta(hours=j) for j in range(FORECAST_HORIZON)]
            
            input_values = scaler.inverse_transform(X_test[i])[:, 0].tolist()
            
            visualization_data.append({
                "input_sequence": {
                    "timestamps": [ts.isoformat() for ts in input_timestamps],
                    "values": input_values
                },
                "true_values": {
                    "timestamps": [ts.isoformat() for ts in true_timestamps],
                    "values": y_test_inv[i].tolist()
                },
                "predicted_values": {
                    "timestamps": [ts.isoformat() for ts in true_timestamps],
                    "values": y_pred_inv[i].tolist()
                }
            })

        # Prepare hyperparameters response - model specific
        # Prepare hyperparameters response
        hyperparameters = {
            "model_type": model_type.upper(),
            "architecture": f"architecture{next_archi_num}",
            "optimizer": best_hps.values['optimizer'],
            "learning_rate": best_hps.values['learning_rate'],
            "batch_size": best_hps.values['batch_size']
        }

        if model_type == 'cnn':
            hyperparameters.update({
                "num_conv_layers": best_hps.values['num_conv_layers'],
                "conv1_filters": [best_hps.values[f'filters_{i+1}'] 
                           for i in range(best_hps.values['num_conv_layers'])],
                "kernel_sizes": [best_hps.values[f'kernel_size_{i+1}'] 
                               for i in range(best_hps.values['num_conv_layers'])],
                "num_dense_layers": best_hps.values['num_dense_layers'],
                "dense1_units": [best_hps.values[f'units_{i+1}'] 
                        for i in range(best_hps.values['num_dense_layers'])],
                "use_dropout": best_hps.values.get('use_dropout', False),
                "dropout_rate": best_hps.values.get('dropout_rate', 0)
            })
        elif model_type in ['lstm', 'rnn']:
            hyperparameters.update({
                "num_layers": best_hps.get('num_layers'),
                "units": [best_hps.get(f'units_{i+1}') for i in range(best_hps.get('num_layers'))],
                "activation": best_hps.get('activation'),
                "use_dropout": best_hps.get('use_dropout'),
                "dropout_rate": best_hps.get('dropout_rate') if best_hps.get('use_dropout') else None
            })

        return jsonify({
            "message": f"New {model_type.upper()} forecasting model trained successfully",
            "results": {
                f"{model_type}_architecture{next_archi_num}": {
                    "overall_metrics": overall_metrics,
                    "step_metrics": step_metrics,
                    "visualization_data": visualization_data,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test)
                }
            },
            "hyperparameters": hyperparameters,
            "model_path": model_path,
            "forecast_horizon": FORECAST_HORIZON
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "message": "Failed to train model",
            "config_received": config
        }), 500
    
class CNNForecastHyperModelCustom(kt.HyperModel):
    def __init__(self, config):
        self.config = config
        self.input_shape = (FORECAST_WINDOW, 5)  # (timesteps, features)
        self.output_units = FORECAST_HORIZON
    
    def build(self, hp):
        model = Sequential()
        # Conv1D layers
        num_conv_layers = hp.Int('num_conv_layers', 1, 2)
        for i in range(num_conv_layers):
            model.add(Conv1D(
                filters=hp.Choice(f'filters_{i+1}', values=self.config['convFilters']),
                kernel_size=hp.Int(f'kernel_size_{i+1}', 2, 5),
                activation='relu',
                padding='same',
                input_shape=self.input_shape if i == 0 else None
            ))
            model.add(MaxPooling1D(pool_size=2))

        model.add(Flatten())

        # Dense layers
        num_dense_layers = hp.Int('num_dense_layers', 1, 2)
        for i in range(num_dense_layers):
            model.add(Dense(
                units=hp.Choice(f'units_{i+1}', values=self.config['denseUnits']),
                activation='relu'
            ))
            if hp.Boolean('use_dropout'):
                model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5)))

        model.add(Dense(self.output_units))

        # Optimizer
        optimizer = hp.Choice('optimizer', values=self.config['optimizers'])
        learning_rate = hp.Choice('learning_rate', values=self.config['learningRates'])

        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = SGD(learning_rate=learning_rate, momentum=0.9)

        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', values=self.config['batchSizes']),
            **kwargs
        )
    

class LSTMHyperModelCustom(kt.HyperModel):
    def __init__(self, config):
        self.config = config
    
    def build(self, hp):
        model = Sequential()

        # Number of LSTM layers
        num_layers = hp.Int('num_layers', 
                           min_value=self.config.get('numLSTMLayers', [1])[0], 
                           max_value=self.config.get('numLSTMLayers', [2])[0])
        
        # LSTM layers
        for i in range(num_layers):
            units = hp.Choice(f'units_{i+1}', values=tuple(self.config['lstmUnits']))  # Convert to tuple
            model.add(LSTM(
                units,
                return_sequences=(i < num_layers - 1),
                input_shape=(None, 5) if i == 0 else None,
                activation=hp.Choice('activation', values=tuple(self.config.get('activations', ['tanh'])))  # Convert to tuple
            ))

        # Dropout
        if hp.Boolean('use_dropout'):
            model.add(Dropout(
                hp.Float('dropout_rate', 
                        min_value=self.config.get('dropoutRates', [0.1])[0], 
                        max_value=self.config.get('dropoutRates', [0.5])[0])
            ))

        # Output layer
        model.add(Dense(FORECAST_HORIZON))

        # Optimizer
        optimizer = hp.Choice('optimizer', values=tuple(self.config['optimizers']))  # Convert to tuple
        learning_rate = hp.Choice('learning_rate', values=tuple(self.config['learningRates']))  # Convert to tuple

        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = SGD(learning_rate=learning_rate, momentum=0.9)

        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", tuple(self.config['batchSizes'])),  # Convert to tuple
            **kwargs,
        )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
