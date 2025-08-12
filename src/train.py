# src/train.py

import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib
from data_preprocessing import preprocess_data

print("Starting the training process...")

# 1. Preprocess the data using the function from our other script
file_path = './data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(file_path)

# 2. Handle Class Imbalance
# We calculate the ratio of negative to positive class for the 'scale_pos_weight' parameter in XGBoost.
# This tells the model to pay more attention to the minority class (Churn=Yes).
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# 3. Define and Train the XGBoost Model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

print("Training the XGBoost model...")
model.fit(X_train, y_train)

# 4. Evaluate the Model
print("Evaluating the model...")
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Save the Model and Preprocessor Artifacts
print("Saving model and preprocessor artifacts...")

# Define the directory and create it if it doesn't exist
artifacts_dir = './artifacts'
os.makedirs(artifacts_dir, exist_ok=True)

# Save the model and the preprocessor object
joblib.dump(model, os.path.join(artifacts_dir, 'churn_model.pkl'))
joblib.dump(preprocessor, os.path.join(artifacts_dir, 'preprocessor.pkl'))

print("\nTraining complete. Artifacts saved successfully!")