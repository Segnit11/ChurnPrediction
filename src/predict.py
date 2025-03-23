# src/predict.py
import pandas as pd
import os
import sys
import joblib

# Add the 'src' folder to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from preprocessing import preprocess_data

# Load the trained model and preprocessing objects
model_path = os.path.join(os.path.dirname(__file__), "../churn_model_stacking.pkl")
geo_encoder_path = os.path.join(os.path.dirname(__file__), "../geo_encoder.pkl")
gender_encoder_path = os.path.join(os.path.dirname(__file__), "../gender_encoder.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "../scaler.pkl")

model = joblib.load(model_path)
encoder_geo = joblib.load(geo_encoder_path)
encoder_gender = joblib.load(gender_encoder_path)
scaler = joblib.load(scaler_path)

# Example new data
new_data = pd.DataFrame({
    "id": [1], "CustomerId": [1001], "Surname": ["Smith"],
    "CreditScore": [600], "Geography": ["Germany"], "Gender": ["Male"],
    "Age": [40], "Tenure": [5], "Balance": [10000], "NumOfProducts": [2],
    "HasCrCard": [1], "IsActiveMember": [1], "EstimatedSalary": [50000]
})

# Preprocess new data
X_new, _, _, _, _, _, _ = preprocess_data(new_data, training=False, encoder_geo=encoder_geo, encoder_gender=encoder_gender, scaler=scaler)
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)[:, 1]
print(f"Prediction: {prediction[0]} (Probability of Churn: {probability[0]:.2f})")