#The code saves the trained model and scaler so they can be used later without retraining.
import os
import joblib

# Create 'models' directory if it does not exist
os.makedirs("models", exist_ok=True)

# Save model and scaler
joblib.dump(log_reg_balanced, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully!")

#--------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib



target_variable = "Churn"  # Define target variable

# Split data into features and target
X = df.drop(columns=[target_variable])
y = df[target_variable]

# Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model with different parameters
log_reg = LogisticRegression(C=0.8, solver='liblinear', random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(log_reg, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

def evaluate_model(model, X_test, y_test, thresholds):
    """
    Evaluates the model for multiple thresholds and prints classification reports.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print("\n=== Model Evaluation for Different Thresholds ===")
    for threshold in thresholds:
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability estimates
        y_pred_adjusted = (y_pred_proba >= threshold).astype(int)  # Apply threshold

        accuracy = accuracy_score(y_test, y_pred_adjusted)
        conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
        class_report = classification_report(y_test, y_pred_adjusted)

        print(f"\nThreshold = {threshold}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

# Evaluate model with multiple thresholds (0.2, 0.3, 0.4)
thresholds = [0.2, 0.3, 0.4]
evaluate_model(log_reg, X_test_scaled, y_test, thresholds)

# Default Threshold Evaluation
y_pred = log_reg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print(f"Model Accuracy (Default Threshold = 0.5): {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
