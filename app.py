from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and training columns
model = joblib.load('iris_model.joblib')
training_columns = joblib.load('training_columns.joblib')

def encode_features(data):
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df, columns=["Procedure_Name", "Gender", "Diabetes_Status", "Wound_Class"], drop_first=True)
    aligned_data = pd.DataFrame(0, index=np.arange(1), columns=training_columns)
    aligned_data.update(df_encoded)
    return aligned_data.to_numpy()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get('data')
        if not isinstance(input_data, dict):
            return jsonify({'error': 'Input data must be a dictionary'}), 400

        input_array = encode_features(input_data)
        print("Input shape:", input_array.shape)  # Debugging: check input shape

        prediction = model.predict(input_array)[0]
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
