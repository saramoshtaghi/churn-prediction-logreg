# README.md

# Churn Prediction using Logistic Regression

## Overview
This project implements a logistic regression model to predict customer churn. The model is trained using a structured dataset and evaluated with multiple decision thresholds to optimize performance. Additionally, **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance the dataset, improving model performance and robustness.

## Project Structure
```
churn-prediction-logreg/
│── src/
│   ├── train.py          # Training and model saving script
│   ├── evaluation.py     # Evaluation script with multiple thresholds
│
│── models/
│   ├── logistic_regression_model.pkl  # Trained model
│   ├── scaler.pkl                     # StandardScaler object
│
│── data/
│   ├── dataset.csv       # Dataset (Not included in repo, add manually)
│
│── notebooks/            # Jupyter notebooks (if used for analysis)
│
│── docs/                 # Documentation files
│
│── requirements.txt       # Required dependencies
│── README.md              # Project documentation
│── LICENSE                # MIT License
│── .gitignore             # Ignore unnecessary files
```

## Features
- **Preprocessing**: Standardization of features.
- **Training**: Logistic regression using `sklearn`.
- **Data Balancing**: **SMOTE applied** to handle class imbalance.
- **Evaluation**:
  - Accuracy, Confusion Matrix, and Classification Report.
  - ROC Curve and AUC Score.
  - Different decision thresholds (default: 0.3, 0.5, 0.7).
  - Comparison of performance with and without SMOTE.

## Setup & Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/churn-prediction-logreg.git
   cd churn-prediction-logreg
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Add dataset**: Place `dataset.csv` inside the `data/` folder.

## Usage
### **Train the model**
Run the training script to preprocess data, train the model, and save it.
```bash
python src/train.py
```
### **Evaluate the model**
Run the evaluation script to test different decision thresholds.
```bash
python src/evaluation.py
```

## Adjusting Decision Thresholds
Modify the `thresholds` list in `evaluation.py` to test different values:
```python
thresholds = [0.2, 0.4, 0.6, 0.8]  # Example thresholds
```

## Output
- **`models/logistic_regression_model.pkl`** → Trained model
- **`models/scaler.pkl`** → Scaler object
- **Evaluation metrics and ROC curve plots**
- **Performance comparison with and without SMOTE**

## License
This project is licensed under the **MIT License**. See the LICENSE file for details.

---
Feel free to customize this repository further as needed. 🚀
