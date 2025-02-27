import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check if "features" exist and has exactly 8 values
    if "features" not in data or len(data["features"]) != 8:
        return jsonify({"error": "Invalid input: Expected exactly 8 features"}), 400

    # Convert input to NumPy array and reshape for prediction
    features = np.array(data["features"]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
