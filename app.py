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
import glob
import os

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


@app.route('/evaluate_classification_model/<model_name>/<version>', methods=['POST'])
def evaluate_classification_model(model_name, version):
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
        model_path = os.path.join(CLASSIFICATION_MODEL_DIR, model_name, f"v{version}.h5")
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model {model_name} version {version} not found"}), 404

        model = tf.keras.models.load_model(model_path)
        
        # Evaluate the model
        thresholds = [0.5] * len(CLASS_LABELS)
        evaluation_results = {
            f"{model_name}_v{version}": evaluate_model(model, X_test, y_test, CLASS_LABELS, thresholds)
        }

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
            "message": "Model evaluated successfully",
            "evaluation": cleaned_evaluation_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/evaluate_forecasting_model/<model_name>/<version>', methods=['POST'])
def evaluate_forecasting_model(model_name, version):
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
        model_path = os.path.join(FORECAST_MODEL_DIR, model_name, f"v{version}.h5")
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
            f"{model_name}_v{version}": {
                "overall_metrics": overall_metrics,
                "step_metrics": step_metrics,
                "visualization_data": visualization_data,
                "test_samples": len(X_test),
                "original_version": f"{model_name}_v{version}",
            }
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
    """Get all models in directory and identify current model"""
    models = []
    current_model = None
    
    # Get model type (classification or forecasting)
    model_type = "classification" if "classification" in directory else "forecasting"
    
    # Scan each model subdirectory
    for model_name in os.listdir(directory):
        model_dir = os.path.join(directory, model_name)
        if os.path.isdir(model_dir):
            # Check for model files in this model's directory
            for file in os.listdir(model_dir):
                if file.endswith('.h5'):
                    model_path = os.path.join(model_dir, file)
                    is_current = "(current)" in file.lower()
                    creation_timestamp = os.path.getctime(model_path)
                    creation_date = datetime.datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M')
                    print(creation_timestamp)
                    models.append({
                        "name": model_name,
                        "version": file.replace('.h5', '').replace('(current)', '').strip(),
                        "path": model_path,
                        "is_current": is_current,
                        "is_tuned": "_tuned" in file.lower(),
                        "creation_date": creation_date,
                    })
                    
                    if is_current:
                        current_model = model_name
    
    return {
        "models": models,
        "current_model": current_model
    }

@app.route('/available_classification_models', methods=['GET'])
def get_available_classification_models():
    try:
        model_info = get_model_info(CLASSIFICATION_MODEL_DIR)
        return jsonify({
            "models": model_info["models"],
            "current_model": model_info["current_model"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/available_forecasting_models', methods=['GET'])
def get_available_forecasting_models():
    try:
        model_info = get_model_info(FORECAST_MODEL_DIR)
        return jsonify({
            "models": model_info["models"],
            "current_model": model_info["current_model"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set_active_model/<model_type>/<model_name>/<version>', methods=['POST'])
def set_active_model(model_type, model_name, version):
    try:
        # Validate model type
        valid_types = ["classification", "forecasting"]
        if model_type not in valid_types:
            return jsonify({"error": f"Invalid model type. Must be one of: {valid_types}"}), 400
        
        # Set base directory
        base_dir = os.path.join("models", model_type)
        model_dir = os.path.join(base_dir, model_name)
        
        # Verify model directory exists
        if not os.path.exists(model_dir):
            return jsonify({"error": f"Model directory {model_name} not found"}), 404
        
        # Find the requested model file
        target_file = None
        for file in os.listdir(model_dir):
            file_version = file.split('(')[0].replace('.h5', '')  # Extract version from filename
            if version == file_version and file.endswith('.h5'):
                target_file = os.path.join(model_dir, file)
                break
        
        if not target_file:
            return jsonify({"error": f"Version {version} not found in {model_name}"}), 404
        
        # Remove (current) tag from all models in this model_type
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
            "message": f"{model_name} {version} set as active",
            "current_model": new_path,
            "model_type": model_type
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to set active model: {str(e)}"}), 500



#--------------------------------------------------------------------------------------------------------------------
#---------------------Delete model endpoint---------------------------------------------------


@app.route('/delete_model/<model_type>/<model_name>/<version>', methods=['DELETE'])
def delete_model(model_type, model_name, version):
    try:
        # Validate model type
        valid_types = ["classification", "forecasting"]
        if model_type not in valid_types:
            return jsonify({
                "error": f"Invalid model type. Must be one of: {valid_types}",
                "status": 400
            }), 400
        
        # Set base directory
        base_dir = os.path.join("models", model_type)
        model_dir = os.path.join(base_dir, model_name)
        
        # Verify model directory exists
        if not os.path.exists(model_dir):
            return jsonify({
                "error": f"Model directory '{model_name}' not found",
                "status": 404
            }), 404
        
        # Find the exact version match (handles cases where version is subset of another)
        target_file = None
        for file in os.listdir(model_dir):
            # Extract clean version (handles v1, v1(current), etc.)
            file_version = file.split('(')[0].replace('.h5', '')
            if version == file_version and file.endswith('.h5'):
                target_file = os.path.join(model_dir, file)
                break
        
        if not target_file:
            return jsonify({
                "error": f"Version '{version}' not found in model '{model_name}'",
                "available_versions": [f.split('(')[0].replace('.h5', '') 
                                    for f in os.listdir(model_dir) if f.endswith('.h5')],
                "status": 404
            }), 404
        
        # Prevent deleting current model
        if '(current)' in os.path.basename(target_file):
            return jsonify({
                "error": "Cannot delete the active model. Set another model as active first.",
                "status": 400
            }), 400
        
        # Additional safety check - don't delete if this is the only model
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        if len(model_files) <= 1:
            return jsonify({
                "error": "Cannot delete the only remaining model version",
                "status": 400
            }), 400
        
        # Delete the file
        os.remove(target_file)
        
        # If directory is empty, remove it (with additional checks)
        try:
            if not os.listdir(model_dir):
                os.rmdir(model_dir)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
