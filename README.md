---
title: ChurnGuard API
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<!-- The YAML block above is metadata for the Hugging Face Space (backend).
     GitHub renders it as a small header; it is required by HF for Docker Spaces. -->

# ChurnGuard — AI Churn Prediction & Retention Intelligence

A four-model ML ensemble that scores bank customers for churn risk, explains the
drivers in plain language, drafts a personalized retention email, and surfaces
portfolio-wide analytics — wrapped in a modern, animated React UI.

## Run locally

You need **two terminals** — one for the Flask API, one for the React UI.

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** and npm
- ~700 MB free disk space (large model artifacts download on first run)

### 1. Backend (Flask API → http://localhost:5001)

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: enable Gemini-generated explanations/retention emails.
# Without it, the app falls back to rule-based text and still works.
export GEMINI_API_KEY=your_key       # Windows: set GEMINI_API_KEY=your_key

python app.py                        # serves on http://localhost:5001
```

On first run, the model artifacts that aren't committed to the repo
(`churn_model_stacking.pkl`, `rf_model.pkl`, etc.) are **downloaded from Google
Drive automatically** via `gdown`. This can take a few minutes. The server
listens on port `5001` by default (override with the `PORT` env var).

Check it's up: `curl http://localhost:5001/health`

### 2. Frontend (React + Tailwind v4 + Framer Motion → http://localhost:3000)

```bash
cd frontend
npm install
npm start                            # opens http://localhost:3000
```

The UI talks to `http://localhost:5001` by default. Point it at another backend
with `REACT_APP_API_URL` (e.g. create `frontend/.env` with
`REACT_APP_API_URL=http://localhost:5001`).

> **Styling note:** Tailwind v4 is compiled with the official Tailwind CLI
> (`src/styles/tailwind.css` → `src/styles/output.css`) because Create React
> App's webpack pipeline doesn't support the v4 PostCSS plugin. The `start` and
> `build` scripts run the compiler automatically (see `tailwind:build`).

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/health` | Service + loaded-model status. |
| GET  | `/customers?q=&limit=&offset=` | Searchable, paginated customer list. |
| GET  | `/analytics` | Portfolio churn analytics (by geography, age, products, activity, credit band). |
| GET  | `/feature-importance` | Global feature importances driving predictions. |
| POST | `/predict` | Score one customer → probability, per-model breakdown, **confidence**, **risk tier + action playbook**, explanation, retention email. |
| POST | `/predict/batch` | Score up to 500 customers at once with a summary breakdown by risk tier. |

`/predict` requests are validated (required fields + sensible numeric ranges) and
return `400` with a `details` list on bad input.

## Frontend highlights
- Cryptgen-style dark fintech marketing layout: animated aurora background, glass
  cards, gradient typography.
- **Framer Motion** spring entrance animations, staggered reveals, animated KPI
  counters, and a floating dashboard preview — all `prefers-reduced-motion` aware.
- Live **Insights** section rendered from `/analytics`, and an interactive
  **Predictor** with a results dashboard (gauge, model breakdown, risk playbook,
  explanation, copy-able email).

---

### **Detailed System Design: Churn Prediction Web Application**

#### Goals
- Allow users to select a customer and input their data.
- Predict churn in real-time using your `StackingClassifier`.
- Display results visually (gauge) and analytically (model breakdown, explanation, email).

#### Components

1. **Frontend (React.js)**:
   - **Purpose**: Interactive UI for data input and result visualization.
   - **Features**:
     - **Form**:
       - **Select a Customer**: Dropdown to pick a customer (e.g., by `CustomerId` or `Surname`).
       - **Inputs**: `CreditScore`, `Balance`, `Location` (Geography), `Number of Products`, `Gender`, `Has Credit Card`, `Is Active Member`, `Age`, `Estimated Salary`, `Tenure (Years)`—11 fields total.
       - **Submit Button**: Triggers prediction request.
     - **Gauge**: Visualizes churn probability (e.g., 0-100%, color-coded: green low, red high).
     - **Churn Probability by Model**: Table or list showing probabilities from Random Forest, GradientBoostingClassifier, XGBoost, and StackingClassifier.
     - **Explanation of Prediction**: Text explaining why the churn probability is high/low (e.g., “High `Age` and low `NumOfProducts` suggest retention”).
     - **Personalized Email**: Text box with a draft email to the customer (e.g., “Dear [Surname], we’ve noticed…”).
   - **Tech**: React.js with libraries:
     - **Form Handling**: `react-hook-form` for simplicity.
     - **Gauge**: `react-gauge-chart` or `react-circular-progressbar`.
     - **Styling**: CSS or Tailwind CSS for a clean look.
   - **Interaction**: Sends a POST request to the Flask backend with form data as JSON, receives prediction response.the **Flask backend** with form data as JSON, receives prediction response.

2. **Backend (Flask)**:

   - **Purpose**: Processes inputs, runs predictions, generates explanations and emails.
   - **Endpoints**:
     - **`/customers` (GET)**: Returns a list of customer IDs/names for the dropdown (could be static or from a DB).
     - **`/predict` (POST)**: Accepts form data, preprocesses it, runs all models, and returns results.
   - **Logic**:
     - **Preprocessing**: Uses `preprocessing.py` with loaded `geo_encoder.pkl`, `gender_encoder.pkl`, `scaler.pkl`.
     - **Inference**: 
       - Loads `churn_model_stacking.pkl` (StackingClassifier).
       - Also loads individual models (RF, GBC, XGBoost) for per-model probabilities—requires saving these separately.
     - **Explanation**: Rule-based logic (e.g., if `NumOfProducts >= 3`, “High product count increases churn risk”).
     - **Email**: Template with placeholders (e.g., “Dear {Surname}, your churn risk is {probability}%…”).
   - **Response**: JSON with:
     - `prediction` (0/1).
     - `probability` (StackingClassifier).
     - `model_probabilities` (RF, GBC, XGBoost, Stacking).
     - `explanation` (text).
     - `email` (text).
   - **Tech**: Flask, `joblib` for model loading, Python for logic.


3. **Model Artifacts**:

   - **Files**:
     - Current: `churn_model_stacking.pkl`, `geo_encoder.pkl`, `gender_encoder.pkl`, `scaler.pkl`.
     - New: Save individual models (`rf_model.pkl`, `gbc_model.pkl`, `xgb_model.pkl`) from `model.py`.
   - **Role**: StackingClassifier for main prediction, individual models for breakdown.

4. **Deployment**:
   - **Local**: Flask dev server + React dev server (via `npm start`).
   - **Production**: Heroku (Flask backend + React build), with static files served via Flask or a CDN.

---


### System Workflow

1. **User Interaction**:
   - Loads the React app, sees a form.
   - Selects a customer from the dropdown (e.g., “Smith, 1001”).
   - Fills in fields (e.g., `Age=40`, `NumOfProducts=2`).
   - Clicks “Predict”.

2. **Frontend**:
   - Sends POST to `http://backend/predict` with JSON:
     ```json
     {
       "CustomerId": 1001, "Surname": "Smith", "CreditScore": 600, "Geography": "Germany",
       "Gender": "Male", "Age": 40, "Tenure": 5, "Balance": 10000, "NumOfProducts": 2,
       "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 50000
     }

3. **Backend**:
   - Receives JSON, converts to DataFrame.
   - Preprocesses using preprocess_data(training=False, ...) → 11 features.
   - Runs predictions:
       - StackingClassifier → prediction, probability.
       - RF, GBC, XGBoost → individual probabilities.
   - Generates:
       - Explanation (e.g., “Low NumOfProducts reduces churn risk”).
       - Email (e.g., “Dear Smith, your churn risk is 16%…”).

4. **Frontend**:
   - Updates **UI** with:
     - **Gauge visualization**.
     - **Model probability breakdown**.
     - **Explanation text**.
     - **Email draft**.

### **Architecture Diagram**

```
[User]
   |
[Frontend: React.js]
   ├── Form: Customer Select, Inputs
   ├── Gauge: Churn Probability
   ├── Table: Model Probabilities
   ├── Text: Explanation
   ├── Text: Email Draft
   |    GET /customers
   |    POST /predict (JSON)
   v
[Backend: Flask]
   ├── /customers: Returns customer list
   ├── /predict:
   |    ├── Preprocess: preprocessing.py + .pkl files
   |    ├── Predict: Stacking + RF, GBC, XGBoost models
   |    ├── Explain: Rule-based logic
   |    ├── Email: Template
   |    └── Response: JSON
   |
[Model Artifacts]
   ├── churn_model_stacking.pkl
   ├── rf_model.pkl, gbc_model.pkl, xgb_model.pkl
   └── geo_encoder.pkl, gender_encoder.pkl, scaler.pkl
```
