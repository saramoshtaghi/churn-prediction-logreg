
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
