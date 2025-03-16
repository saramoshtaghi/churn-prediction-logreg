# Customer Churn Prediction Project

## 📌 Project Overview
This project focuses on predicting customer churn using Logistic Regression. It includes two versions:
1. **Without SMOTE** - The model is trained on the original imbalanced dataset.
2. **With SMOTE** - The model is trained using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

The project evaluates both models at multiple decision thresholds (0.2, 0.3, 0.4, 0.5) and compares their performance using:
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **ROC-AUC Curve**

---

## 📂 Folder Structure
```
/customer_churn_project/
│── /models/                 # Saved trained models
│   ├── logistic_regression_no_smot.pkl
│   ├── logistic_regression_with_smot.pkl
│   ├── scaler.pkl
│── /src/                    # Source code for training and evaluation
│   ├── train.py              # Training script (handles both SMOTE and no-SMOTE versions)
│   ├── eval.py               # Evaluation script for both models
│── /data/                    # Dataset storage
│   ├── customer_churn.csv    # Raw dataset
│── README.md                 # Project documentation
│── requirements.txt          # Dependencies list
│── LICENSE                   # License file
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

### **3️⃣ Preprocess Data**
- The script **automatically handles missing values** and scales numerical features using `StandardScaler`.
- SMOTE is applied in the training phase **only when specified**.

---

## 🎯 Model Training
Run the training script to generate models:
```bash
python src/train.py --use_smot  # Runs training with SMOTE
python src/train.py  # Runs training without SMOTE
```
- The trained models and scalers are saved in the `/models/` directory.

---

## 📊 Model Evaluation
To evaluate and compare models at different thresholds (0.2, 0.3, 0.4, 0.5):
```bash
python src/eval.py
```
This script computes:
- **Accuracy, Precision, Recall, F1-score**
- **ROC-AUC Score**
- **Plots ROC Curves**

---


## 📩 Contributing
Feel free to submit PRs or open issues. For discussions, reach out via **sarahmoshtaq@gmail.com**.

---

## 📜 License
This project is open-source under the MIT License. See **LICENSE** for details.

---

**🚀 Happy Coding!**

