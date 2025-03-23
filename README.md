# Bank Churn Prediction
Predicts customer churn using a stacked machine learning model.

## Overview
- **Dataset**: 165,034 customers, 21.2% churn rate (130,113 non-churned, 34,921 churned).
- **Key Findings**:
  - `NumOfProducts`: 6% churn at 2 products, 88% at 3—strong non-linear driver.
  - `Age`: 7-year gap (44 vs. 37) between churned and non-churned customers.
- **Model**: StackingClassifier combining Random Forest, GradientBoostingClassifier, and XGBoost, with Logistic Regression as the meta-model.
  - **Accuracy**: 90%.
  - **ROC-AUC**: 0.9655 (highest among tested models).
  - **Features**: 11 (pruned from 14 for efficiency).

## Usage
1. **Train the Model**:
   ```bash
   python src/model.py
   Trains the StackingClassifier and saves the model and preprocessing objects:
    - churn_model_stacking.pkl
    - geo_encoder.pkl
    - gender_encoder.pkl
    - scaler.pkl
   Outputs performance metrics (e.g., accuracy: 0.90, ROC-AUC: 0.9655).

2. **Make Predictions**:
    ```bash
    python src/predict.py
     - Loads the saved model and preprocessing objects.
     - Predicts churn for new data (example output: Prediction: 0 (Probability of Churn: 0.16)).
    

## Results
**Performance**: Outperforms individual models:
    - Random Forest: 0.963 ROC-AUC.
    - GradientBoostingClassifier: 0.9628 ROC-AUC.
    - XGBoost: 0.952 ROC-AUC.
**Key Drivers**: Age, NumOfProducts, IsActiveMember (inferred from base models).
**Why Stacking?**: Combines Random Forest’s balanced approach, GradientBoosting’s focus on Age, and XGBoost’s emphasis on NumOfProducts for superior predictive power.

## Files
    - src/preprocessing.py: Data preprocessing logic (handles training and prediction).
    - src/model.py: Trains and evaluates the StackingClassifier, saves model and preprocessing objects.
    - src/predict.py: Loads saved objects and predicts churn on new data.
**Saved Objects (in churn_prediction/)**:
    - churn_model_stacking.pkl: Trained StackingClassifier model.
    - geo_encoder.pkl: Geography encoder.
    - gender_encoder.pkl: Gender encoder.
    - scaler.pkl: Standard scaler for numerical features.