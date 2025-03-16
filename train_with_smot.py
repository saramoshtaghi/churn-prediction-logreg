# Re-import necessary libraries after execution state reset
import os
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset again
import pandas as pd

# Update file path if needed
csv_file = "/content/sample_data/Customer Churn.csv"  # Adjust if filename differs
df = pd.read_csv(csv_file)

# Step 1: Split data into features (X) and target variable (y)
target_variable = "Churn"  # Ensure this is the correct column name

X = df.drop(columns=[target_variable])  # Features
y = df[target_variable]  # Target

# Step 2: Split into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 4: Standardize Features
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train Logistic Regression Model on Balanced Data
log_reg_balanced = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
log_reg_balanced.fit(X_train_balanced_scaled, y_train_balanced)

# Define function to evaluate model at multiple thresholds
def evaluate_multiple_thresholds(model, X_test, y_test, thresholds):
    """
    Evaluates the model for different decision thresholds and prints classification reports.
    """
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

# Define thresholds to evaluate
thresholds_to_test = [0.2, 0.3, 0.4, 0.5]  # Including default threshold (0.5)

# Evaluate model at multiple thresholds
evaluate_multiple_thresholds(log_reg_balanced, X_test_scaled, y_test, thresholds_to_test)
