# src/model.py
import pandas as pd
import os
import sys

# Add the 'src' folder to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load and preprocess data
data_path = os.path.join(os.path.dirname(__file__), "../data/bank_churn_data.csv")
df = load_data(data_path)
X_train, X_test, y_train, y_test = preprocess_data(df)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Print results
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Feature importance
importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
print("\nFeature Importance:\n", importances)