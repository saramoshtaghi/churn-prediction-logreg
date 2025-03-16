import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
csv_file = "/content/sample_data/Customer Churn.csv"  # Adjust if filename differs
df = pd.read_csv(csv_file)

target_variable = "Churn"  # Ensure this is the correct column name
X = df.drop(columns=[target_variable])
y = df[target_variable]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model WITHOUT SMOTE
log_reg_no_smot = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
log_reg_no_smot.fit(X_train_scaled, y_train)

y_pred_no_smot_proba = log_reg_no_smot.predict_proba(X_test_scaled)[:, 1]

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)

# Train Logistic Regression model WITH SMOTE
log_reg_smot = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
log_reg_smot.fit(X_train_balanced_scaled, y_train_balanced)

y_pred_smot_proba = log_reg_smot.predict_proba(X_test_scaled)[:, 1]

# Define function to evaluate both models
def evaluate_models(models, X_test, y_test, thresholds):
    """
    Evaluates models for different decision thresholds and prints classification reports.
    """
    for model_name, model_proba in models.items():
        print(f"\n=== Model Evaluation: {model_name} ===")
        for threshold in thresholds:
            y_pred_adjusted = (model_proba >= threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred_adjusted)
            conf_matrix = confusion_matrix(y_test, y_pred_adjusted)
            class_report = classification_report(y_test, y_pred_adjusted)
            roc_auc = roc_auc_score(y_test, model_proba)
            fpr, tpr, _ = roc_curve(y_test, model_proba)
            
            print(f"\nThreshold = {threshold}")
            print(f"Accuracy: {accuracy:.4f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            print("Classification Report:")
            print(class_report)
            print(f"ROC-AUC Score: {roc_auc:.4f}")
            
            # Plot ROC Curve
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title(f"ROC Curve - {model_name} (Threshold = {threshold})")
            plt.legend()
            plt.show()

# Define thresholds to evaluate
thresholds_to_test = [0.2, 0.3, 0.4, 0.5]

# Compare models at different thresholds
evaluate_models({
    "Without SMOTE": y_pred_no_smot_proba,
    "With SMOTE": y_pred_smot_proba
}, X_test_scaled, y_test, thresholds_to_test)

# Save models and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(log_reg_no_smot, "models/logistic_regression_no_smot.pkl")
joblib.dump(log_reg_smot, "models/logistic_regression_with_smot.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nBoth models and scaler saved successfully!")
