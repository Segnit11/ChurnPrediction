import pandas as pd
import os
import sys
import joblib
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add 'src/' to the path before importing project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.preprocessing import preprocess_data
from src.utils import explain_prediction, generate_email
from src.analytics import (
    compute_portfolio_analytics,
    FEATURE_IMPORTANCE,
    risk_tier,
)

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

# Pre-compute portfolio analytics once at startup (cached).
PORTFOLIO_ANALYTICS = compute_portfolio_analytics(customer_data)

FEATURE_ORDER = [
    "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
    "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary"
]

REQUIRED_FIELDS = [f for f in FEATURE_ORDER if f != "CustomerId"]


def _validate_record(data):
    """Return a list of human-readable validation errors for a prediction payload."""
    errors = []
    if not isinstance(data, dict):
        return ["Payload must be a JSON object."]

    for field in REQUIRED_FIELDS:
        if data.get(field) in (None, ""):
            errors.append(f"Missing required field: {field}")

    numeric_ranges = {
        "CreditScore": (300, 900),
        "Age": (18, 120),
        "Tenure": (0, 50),
        "Balance": (0, None),
        "NumOfProducts": (1, 4),
        "EstimatedSalary": (0, None),
    }
    for field, (lo, hi) in numeric_ranges.items():
        val = data.get(field)
        if val in (None, ""):
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number.")
            continue
        if lo is not None and num < lo:
            errors.append(f"{field} must be ≥ {lo}.")
        if hi is not None and num > hi:
            errors.append(f"{field} must be ≤ {hi}.")

    if data.get("Geography") and data["Geography"] not in ("France", "Germany", "Spain"):
        errors.append("Geography must be France, Germany, or Spain.")
    if data.get("Gender") and data["Gender"] not in ("Male", "Female"):
        errors.append("Gender must be Male or Female.")
    return errors


def _predict_one(data, with_narrative=True):
    """Run the model stack on a single record and assemble the response dict."""
    new_data = pd.DataFrame([data])[FEATURE_ORDER]
    X_new, _, _, _, _, _, _ = preprocess_data(
        new_data,
        training=False,
        encoder_geo=encoder_geo,
        encoder_gender=encoder_gender,
        scaler=scaler,
    )
    stacking_pred = stacking_model.predict(X_new)[0]
    stacking_prob = stacking_model.predict_proba(X_new)[0][1]
    model_probs = {
        "RandomForest": float(rf_model.predict_proba(X_new)[0][1]),
        "GradientBoosting": float(gbc_model.predict_proba(X_new)[0][1]),
        "XGBoost": float(xgb_model.predict_proba(X_new)[0][1]),
        "StackingClassifier": float(stacking_prob),
    }
    # Ensemble agreement = 1 - spread across models (a simple confidence proxy).
    probs = list(model_probs.values())
    confidence = round((1 - (max(probs) - min(probs))) * 100, 1)

    response = {
        "prediction": int(stacking_pred),
        "probability": float(stacking_prob),
        "model_probabilities": model_probs,
        "confidence": confidence,
        "risk": risk_tier(stacking_prob),
    }

    if with_narrative:
        surname = data.get("Surname", "This customer")
        explanation = explain_prediction(stacking_prob, data, surname)
        response["explanation"] = explanation
        response["email"] = generate_email(stacking_prob, data, explanation, surname)
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": ["RandomForest", "GradientBoosting", "XGBoost", "StackingClassifier"],
        "customers": len(customers),
    })


@app.route("/customers", methods=["GET"])
def get_customers():
    """Searchable, paginated customer list.

    Query params: q (search by id/surname), limit (default 100), offset (default 0).
    """
    q = (request.args.get("q") or "").strip().lower()
    try:
        limit = max(1, min(int(request.args.get("limit", 100)), 500))
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return jsonify({"error": "limit and offset must be integers"}), 400

    results = customers
    if q:
        results = [
            c for c in customers
            if q in str(c["CustomerId"]).lower() or q in str(c["Surname"]).lower()
        ]
    page = results[offset:offset + limit]
    return jsonify({
        "total": len(results),
        "limit": limit,
        "offset": offset,
        "results": page,
    })


@app.route("/analytics", methods=["GET"])
def analytics():
    """Aggregate portfolio analytics for dashboard / marketing sections."""
    return jsonify(PORTFOLIO_ANALYTICS)


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """Global feature importances driving churn predictions."""
    return jsonify({"features": FEATURE_IMPORTANCE})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        errors = _validate_record(data)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400
        return jsonify(_predict_one(data, with_narrative=True))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Score multiple customers at once. Body: {"customers": [ {...}, ... ]}.

    Returns lightweight scores (no AI narrative) plus a summary breakdown.
    """
    try:
        payload = request.get_json(silent=True) or {}
        records = payload.get("customers")
        if not isinstance(records, list) or not records:
            return jsonify({"error": "Body must include a non-empty 'customers' array."}), 400
        if len(records) > 500:
            return jsonify({"error": "Batch limited to 500 customers per request."}), 400

        results = []
        tier_counts = {}
        for idx, rec in enumerate(records):
            errs = _validate_record(rec)
            if errs:
                results.append({"index": idx, "error": "Validation failed", "details": errs})
                continue
            scored = _predict_one(rec, with_narrative=False)
            tier = scored["risk"]["tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            results.append({
                "index": idx,
                "CustomerId": rec.get("CustomerId"),
                "Surname": rec.get("Surname"),
                "probability": scored["probability"],
                "prediction": scored["prediction"],
                "riskTier": tier,
            })

        valid = [r for r in results if "probability" in r]
        summary = {
            "count": len(records),
            "scored": len(valid),
            "atRisk": sum(1 for r in valid if r["prediction"] == 1),
            "avgProbability": round(sum(r["probability"] for r in valid) / len(valid), 4) if valid else 0,
            "byTier": tier_counts,
        }
        return jsonify({"summary": summary, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host="0.0.0.0", port=port)
