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
csv_file = "data/processed/telecom_churn_processed.csv"  # Adjust if filename differs
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
