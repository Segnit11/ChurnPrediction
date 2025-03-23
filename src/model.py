# src/model.py
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Add the 'src' folder to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from preprocessing import load_data, preprocess_data

# Load and preprocess data
data_path = os.path.join(os.path.dirname(__file__), "../data/bank_churn_data.csv")
df = load_data(data_path)
X_train, X_test, y_train, y_test, encoder_geo, encoder_gender, scaler = preprocess_data(df, training=True)

# Define and train individual base models
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

gbc_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, random_state=42)
gbc_model.fit(X_train, y_train)

xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# Define base models for stacking (using pre-trained models)
base_models = [
    ("rf", rf_model),
    ("gbc", gbc_model),
    ("xgb", xgb_model)
]

# Define stacking classifier with Logistic Regression as meta-model
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Predict and evaluate the stacking model
y_pred = stacking_model.predict(X_test)
y_prob = stacking_model.predict_proba(X_test)[:, 1]
print("StackingClassifier Classification Report:\n", classification_report(y_test, y_pred))
print("StackingClassifier ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Save all models and preprocessing objects
output_dir = os.path.dirname(__file__) + "/.."
joblib.dump(rf_model, os.path.join(output_dir, "rf_model.pkl"))
joblib.dump(gbc_model, os.path.join(output_dir, "gbc_model.pkl"))
joblib.dump(xgb_model, os.path.join(output_dir, "xgb_model.pkl"))
joblib.dump(stacking_model, os.path.join(output_dir, "churn_model_stacking.pkl"))
joblib.dump(encoder_geo, os.path.join(output_dir, "geo_encoder.pkl"))
joblib.dump(encoder_gender, os.path.join(output_dir, "gender_encoder.pkl"))
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
print("Models and preprocessing objects saved: rf_model.pkl, gbc_model.pkl, xgb_model.pkl, churn_model_stacking.pkl, geo_encoder.pkl, gender_encoder.pkl, scaler.pkl")