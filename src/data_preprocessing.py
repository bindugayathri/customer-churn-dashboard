# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(file_path):
    """
    Loads data, cleans it, and prepares it for model training.
    
    Args:
        file_path (str): The path to the raw CSV data file.
        
    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test, and the fitted preprocessor.
    """
    # Load data
    df = pd.read_csv(file_path)

    # --- Data Cleaning ---
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Fill missing values with the median, as it's robust to outliers
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Drop customerID as it is not a feature
    df.drop(columns=['customerID'], inplace=True)

    # Convert target variable 'Churn' to binary (0 or 1)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # --- Feature and Target Split ---
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # --- Define Numerical and Categorical Features ---
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # --- Create Preprocessing Pipelines for Numerical and Categorical Data ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # --- Split Data into Training and Testing sets ---
    # We use stratify=y to ensure the train and test sets have a similar proportion of churned users.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Only transform the test data (to prevent data leakage)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor