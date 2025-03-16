# Customer Churn Prediction Project

## 📌 Project Overview
This project focuses on predicting customer churn using Logistic Regression. It includes two versions:
1. **Without SMOTE** - The model is trained on the original imbalanced dataset.
2. **With SMOTE** - The model is trained using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

Both versions allow for threshold tuning to optimize recall and precision.

---

## 📂 Folder Structure
```
/customer_churn_project/
│── /models/                 # Saved trained models
│   ├── logistic_regression_no_smot.pkl
│   ├── logistic_regression_with_smot.pkl
│   ├── scaler_no_smot.pkl
│   ├── scaler_with_smot.pkl
│── /src/                    # Source code for training and evaluation
│   ├── train_no_smot.py      # Training script without SMOTE
│   ├── train_with_smot.py    # Training script with SMOTE
│   ├── evaluate_model.py     # Evaluation script
│── /data/                    # Dataset storage
│   ├── customer_churn.csv    # Raw dataset
│── README.md                 # Project documentation
```

---

## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/customer_churn_project.git
cd customer_churn_project
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Train the Model**
#### **Without SMOTE**
```bash
python src/train_no_smot.py
```
#### **With SMOTE**
```bash
python src/train_with_smot.py
```

### **4️⃣ Evaluate the Model**
```bash
python src/evaluate_model.py
```

---

## 🔍 Model Training Details
- **`train_no_smot.py`**: Trains a Logistic Regression model on the original dataset.
- **`train_with_smot.py`**: Applies SMOTE to balance the dataset before training.
- **Hyperparameters:** C=1.0, solver='liblinear'
- **Feature Scaling:** StandardScaler applied to numeric features.
- **Decision Threshold Adjustment:** Threshold tuning for optimizing Precision vs. Recall.

---

## 📊 Model Evaluation
We evaluate both models using:
- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC-AUC Curve**
- **Confusion Matrix Analysis**

### **Compare Performance**
| Model | Precision | Recall | F1-score |
|--------|------------|---------|----------|
| No SMOTE | High Precision, Low Recall | More False Negatives | Balanced |
| With SMOTE | Lower Precision, Higher Recall | Captures More Churners | Higher Recall |

**Choosing the Model:**
- If you care more about **capturing churners**, use **With SMOTE** (higher recall).
- If you want fewer **false churn alerts**, use **Without SMOTE** (higher precision).

---

## 🔥 Next Steps
- [ ] Experiment with **other models** (Random Forest, XGBoost).
- [ ] Tune **decision thresholds dynamically**.
- [ ] Deploy as a **Flask API or Streamlit app**.

---

## 📩 Contributing
Feel free to submit PRs or open issues. For discussions, reach out via **sarahmoshtaq@gmail.com**.

---

## 📜 License
This project is open-source under the MIT License. See **LICENSE** for details.

---

**🚀 Happy Coding!**

