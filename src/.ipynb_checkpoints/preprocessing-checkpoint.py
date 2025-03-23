# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the churn dataset."""
    print("Columns in dataset:", df.columns.tolist())
    
    # Drop non-predictive columns
    df = df.drop(columns=["id", "CustomerId", "Surname"])

    # Handle missing values (avoid inplace=True)
    for col in ["CreditScore", "Balance", "EstimatedSalary"]:
        df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["Product_Usage"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["Age_Tenure"] = df["Age"] * df["Tenure"]

    # Encode categorical variables (use sparse_output instead of sparse)
    encoder_geo = OneHotEncoder(sparse_output=False, drop="first")
    geo_encoded = encoder_geo.fit_transform(df[["Geography"]])
    geo_df = pd.DataFrame(geo_encoded, columns=encoder_geo.get_feature_names_out(["Geography"]))

    encoder_gender = OneHotEncoder(sparse_output=False, drop="first")
    gender_encoded = encoder_gender.fit_transform(df[["Gender"]])
    gender_df = pd.DataFrame(gender_encoded, columns=encoder_gender.get_feature_names_out(["Gender"]))

    df = pd.concat([df.drop(["Geography", "Gender"], axis=1), geo_df, gender_df], axis=1)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", 
                      "Balance_to_Salary", "Product_Usage", "Age_Tenure"]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Split features and target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test the preprocessing
    data_path = os.path.join(os.path.dirname(__file__), "../data/bank_churn_data.csv")
    print("Looking for file at:", data_path)
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Preprocessed X_train sample:\n", X_train.head())