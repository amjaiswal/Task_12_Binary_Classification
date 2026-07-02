# Task 12 - Binary Classification

## 📌 Objective
The objective of this project is to build a production-ready binary classification model using the Titanic dataset. The project includes probability calibration, threshold optimization, hyperparameter tuning, and comprehensive model evaluation.

---

## 📂 Dataset
- Titanic Dataset
- Target Variable: **Survived**

---

## 🚀 Features
- Data Preprocessing
- Missing Value Handling
- Feature Selection
- Label Encoding
- Train-Test Split
- Random Forest Classifier
- Hyperparameter Tuning using GridSearchCV
- Probability Calibration using CalibratedClassifierCV
- Threshold Optimization
- ROC-AUC Evaluation
- Confusion Matrix
- Calibration Curve
- ROC Curve
- Segment-wise Performance Analysis
- Model Serialization using Joblib
- Error Handling

---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## 📁 Project Structure

```
Task_12_Binary_Classification/
│
├── data/
│   └── Titanic-Dataset.csv
│
├── models/
│   └── calibrated_model.pkl
│
├── outputs/
│   ├── calibration_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── threshold_plot.png
│
├── task12.py
├── Task12.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python task12.py
```

---

## 📊 Model Performance

- Accuracy: **81.01%**
- Precision: **72.73%**
- Recall: **81.16%**
- F1 Score: **76.71%**
- ROC-AUC Score: **84.08%**
- Best Threshold: **0.3236**

---

## 📦 Output Files

The project generates:

- calibration_curve.png
- threshold_plot.png
- confusion_matrix.png
- roc_curve.png
- calibrated_model.pkl

---

## ✅ Conclusion

This project demonstrates a complete binary classification pipeline, including data preprocessing, model training, hyperparameter tuning, probability calibration, threshold optimization, evaluation, and model serialization. The final model is suitable for deployment and decision-making scenarios.