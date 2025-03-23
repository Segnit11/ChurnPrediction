# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df, training=True, encoder_geo=None, encoder_gender=None, scaler=None):
    """Preprocess the churn dataset. If training=False, use provided encoders/scaler."""
    print("Columns in dataset:", df.columns.tolist())
    
    # Drop non-predictive columns
    df = df.drop(columns=["id", "CustomerId", "Surname"], errors="ignore")

    # Handle missing values
    for col in ["CreditScore", "Balance", "EstimatedSalary"]:
        df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Product_Usage"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]

    # Encode categorical variables
    if training:
        encoder_geo = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        encoder_gender = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        geo_encoded = encoder_geo.fit_transform(df[["Geography"]])
        gender_encoded = encoder_gender.fit_transform(df[["Gender"]])
    else:
        if encoder_geo is None or encoder_gender is None:
            raise ValueError("Encoders must be provided for prediction")
        geo_encoded = encoder_geo.transform(df[["Geography"]])
        gender_encoded = encoder_gender.transform(df[["Gender"]])
    geo_df = pd.DataFrame(geo_encoded, columns=encoder_geo.get_feature_names_out(["Geography"]))
    gender_df = pd.DataFrame(gender_encoded, columns=encoder_gender.get_feature_names_out(["Gender"]))

    # Combine and drop original categorical columns
    df = pd.concat([df.drop(["Geography", "Gender"], axis=1), geo_df, gender_df], axis=1)

    # Scale numerical features
    numerical_cols = ["CreditScore", "Age", "Balance", "NumOfProducts", "EstimatedSalary", 
                      "Balance_to_Salary", "Product_Usage", "Age_Tenure"]
    if training:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for prediction")
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Split features and target
    if training:
        X = df.drop(columns=["Exited", "Tenure", "HasCrCard", "Geography_Spain"], errors="ignore")
        y = df["Exited"]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)
        return X_train, X_test, y_train, y_test, encoder_geo, encoder_gender, scaler
    else:
        X = df.drop(columns=["Tenure", "HasCrCard", "Geography_Spain"], errors="ignore")
        return X, None, None, None, None, None, None

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../data/bank_churn_data.csv")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, _, _, _ = preprocess_data(df, training=True)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)