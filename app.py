from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

MODEL_PATH = "lstm_multilabel_anomalie_normal_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  
        df = pd.DataFrame(data)

        # Ensure Timestamp is present
        if "Timestamp" not in df.columns:
            return jsonify({"error": "Missing Timestamp"}), 400

        # Extract "Hour" from Timestamp
        df["Month"] = pd.to_datetime(df["Timestamp"]).dt.month

        # Ensure all required features exist
        if not all(col in df.columns for col in FEATURE_COLUMNS):
            return jsonify({"error": "Missing required features"}), 400

        input_data = df[FEATURE_COLUMNS].values.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=1)  # Reshape to (batch_size, 1, features)

        predictions = model.predict(input_data)

        results = []
        for i, pred in enumerate(predictions):
            predicted_classes = [CLASS_LABELS[j] for j, value in enumerate(pred) if value > 0.5]  # Adjust threshold
            results.append({
                "Predicted_Classes": predicted_classes if predicted_classes else ["Normal"]
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
