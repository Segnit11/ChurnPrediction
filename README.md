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
