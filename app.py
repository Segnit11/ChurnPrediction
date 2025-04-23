import pandas as pd
import os
import sys
import joblib
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.preprocessing import preprocess_data
from src.utils import explain_prediction, generate_email

# Add 'src/' to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

app = Flask(__name__)
CORS(app)

# Download model files if not present
base_dir = os.path.dirname(__file__)
model_files = {
    'churn_model_stacking.pkl': 'https://drive.google.com/uc?export=download&id=153GU_dX3KbqevGw4YYYtNSJ9f1gcFlqF',
    'rf_model.pkl': 'https://drive.google.com/uc?export=download&id=1OTlzKxJovXrn1nVQNzFPh6Sg0mGGKv5X',
    'gbc_model.pkl': 'https://drive.google.com/uc?export=download&id=1WGFjm6zxRdpQ2qjbSrn3gVb_hvKPDV8x',
    'xgb_model.pkl': 'https://drive.google.com/uc?export=download&id=18FQe3ItIFD723mRit3w-buFcBTlrg_h0',
    'scaler.pkl': 'https://drive.google.com/uc?export=download&id=1bVWAMse9OJEUQqKWTwkbEzht5KbzF5pP',
    'geo_encoder.pkl': 'https://drive.google.com/uc?export=download&id=1_okgwDIDbowjhUiFAbxHHRLCh2EO4pZs',
    'gender_encoder.pkl': 'https://drive.google.com/uc?export=download&id=1jYUQopoy35OleSGo7Xt6iEpGgl3BmPP7'
}

for file, url in model_files.items():
    file_path = os.path.join(base_dir, file)
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)

# Load models and preprocessing objects
rf_model = joblib.load(os.path.join(base_dir, "rf_model.pkl"))
gbc_model = joblib.load(os.path.join(base_dir, "gbc_model.pkl"))
xgb_model = joblib.load(os.path.join(base_dir, "xgb_model.pkl"))
stacking_model = joblib.load(os.path.join(base_dir, "churn_model_stacking.pkl"))
encoder_geo = joblib.load(os.path.join(base_dir, "geo_encoder.pkl"))
encoder_gender = joblib.load(os.path.join(base_dir, "gender_encoder.pkl"))
scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))

customer_data = pd.read_csv(os.path.join(base_dir, "data/bank_churn_data.csv"))
customers = customer_data[["CustomerId", "Surname"]].drop_duplicates().to_dict(orient="records")

@app.route("/customers", methods=["GET"])
def get_customers():
    return jsonify(customers)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        feature_order = [
            "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
            "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
            "IsActiveMember", "EstimatedSalary"
        ]
        new_data = pd.DataFrame([data])[feature_order]
        X_new, _, _, _, _, _, _ = preprocess_data(
            new_data,
            training=False,
            encoder_geo=encoder_geo,
            encoder_gender=encoder_gender,
            scaler=scaler
        )
        stacking_pred = stacking_model.predict(X_new)[0]
        stacking_prob = stacking_model.predict_proba(X_new)[0][1]
        rf_prob = rf_model.predict_proba(X_new)[0][1]
        gbc_prob = gbc_model.predict_proba(X_new)[0][1]
        xgb_prob = xgb_model.predict_proba(X_new)[0][1]
        model_probs = {
            "RandomForest": float(rf_prob),
            "GradientBoosting": float(gbc_prob),
            "XGBoost": float(xgb_prob),
            "StackingClassifier": float(stacking_prob)
        }
        surname = data["Surname"]
        explanation = explain_prediction(stacking_prob, data, surname)
        email = generate_email(stacking_prob, data, explanation, surname)
        
        response = {
            "prediction": int(stacking_pred),
            "probability": float(stacking_prob),
            "model_probabilities": model_probs,
            "explanation": explanation,
            "email": email
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)