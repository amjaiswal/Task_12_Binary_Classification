# %% [markdown]
# # Import Library

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

import joblib

# %% [markdown]
# # Load Dataset

# %%
df = pd.read_csv("data/Titanic-Dataset.csv")

df.head()

# %% [markdown]
# # Dataset Information

# %%
df.info()

# %% [markdown]
# # Missing Values

# %%
df.isnull().sum()

# %% [markdown]
# # Fill Missing Values

# %%
# Fill Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df.drop("Cabin", axis=1, inplace=True)

print("Missing values handled successfully!")

# %% [markdown]
# # Label Encoding

# %%
label_encoder = LabelEncoder()

df["Sex"] = label_encoder.fit_transform(df["Sex"])
df["Embarked"] = label_encoder.fit_transform(df["Embarked"])

# %%
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# %%
df.head()

# %% [markdown]
# # Features and Target Variable

# %%
X = df.drop("Survived", axis=1)
y = df["Survived"]

print("Features Shape :", X.shape)
print("Target Shape   :", y.shape)

# %% [markdown]
# # Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape)
print("Testing Samples  :", X_test.shape)

# %% [markdown]
# # Train Random Forest Classifier

# %%
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")

# %% [markdown]
# # Initial Prediction

# %%
y_pred = rf_model.predict(X_test)

print("Prediction completed successfully!")

# %% [markdown]
# # Initial Model Performance

# %%
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# %% [markdown]
# # Hyperparameter Tuning

# %%
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest F1 Score:")
print(grid_search.best_score_)

# %% [markdown]
# ## Best Tuned Model

# %%
best_model = grid_search.best_estimator_

print(best_model)

# %% [markdown]
# ## Prediction using Tuned Model

# %%
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Performance of Tuned Model")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# %% [markdown]
# # Probability Calibration

# %%
calibrated_model = CalibratedClassifierCV(
    estimator=best_model,
    method="sigmoid",   # Platt Scaling
    cv=5
)

calibrated_model.fit(X_train, y_train)

print("Model calibrated successfully!")

# %% [markdown]
# # Predict Calibrated Probabilities

# %%
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

print("First 10 Predicted Probabilities:")
print(y_prob[:10])

# %% [markdown]
# # Calibration Curve

# %%
prob_true, prob_pred = calibration_curve(
    y_test,
    y_prob,
    n_bins=10
)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true, marker='o', label="Calibrated Model")
plt.plot([0,1],[0,1], linestyle='--', label="Perfect Calibration")

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")

plt.legend()

plt.savefig("outputs/calibration_curve.png")

plt.show()

# %% [markdown]
# # ROC-AUC Score

# %%
roc_auc = roc_auc_score(y_test, y_prob)

print(f"ROC-AUC Score : {roc_auc:.4f}")

# %% [markdown]
# # Find the Best Threshold

# %%
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Calculate F1 Score for each threshold
f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"Best Threshold : {best_threshold:.4f}")
print(f"Best F1 Score  : {f1_scores[best_index]:.4f}")

# %% [markdown]
# # Predictions Using Best Threshold

# %%
y_pred_best = (y_prob >= best_threshold).astype(int)

# %% [markdown]
# # Evaluation After Threshold Optimization

# %%
accuracy = accuracy_score(y_test, y_pred_best)
precision = precision_score(y_test, y_pred_best)
recall = recall_score(y_test, y_pred_best)
f1 = f1_score(y_test, y_pred_best)

print("Performance After Threshold Optimization")
print("-" * 45)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# %% [markdown]
# # Threshold vs F1 Score

# %%
plt.figure(figsize=(8,5))

plt.plot(thresholds, f1_scores, linewidth=2)
plt.scatter(best_threshold, f1_scores[best_index], marker='o')

plt.title("Threshold vs F1 Score")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")

plt.grid(True)

plt.savefig("outputs/threshold_plot.png")

plt.show()

# %% [markdown]
# # Confusion Matrix

# %%
cm = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.show()

# %% [markdown]
# # ROC Curve

# %%
RocCurveDisplay.from_predictions(y_test, y_prob)

plt.title("ROC Curve")

plt.savefig("outputs/roc_curve.png")

plt.show()

# %% [markdown]
# # Segment-wise Analysis

# %%
results = X_test.copy()

results["Actual"] = y_test.values
results["Prediction"] = y_pred_best

print("Performance by Gender")

for gender in sorted(results["Sex"].unique()):

    subset = results[results["Sex"] == gender]

    acc = accuracy_score(
        subset["Actual"],
        subset["Prediction"]
    )

    gender_name = "Female" if gender == 0 else "Male"

    print(f"{gender_name} Accuracy : {acc:.4f}")

# %% [markdown]
# # Passenger Class Performance

# %%
print("Performance by Passenger Class")

for pclass in sorted(results["Pclass"].unique()):

    subset = results[results["Pclass"] == pclass]

    acc = accuracy_score(
        subset["Actual"],
        subset["Prediction"]
    )

    print(f"Class {pclass} Accuracy : {acc:.4f}")

# %% [markdown]
# # Save Model

# %%
joblib.dump(
    calibrated_model,
    "models/calibrated_model.pkl"
)

print("Model saved successfully!")

# %% [markdown]
# # Error Handling

# %%
try:
    test_prob = calibrated_model.predict_proba(X_test)
    print("Prediction successful!")
except Exception as e:
    print("Error:", e)

# %% [markdown]
# # Verify Saved Model

# %%
loaded_model = joblib.load("models/calibrated_model.pkl")

sample_prediction = loaded_model.predict(X_test[:5])

print("Sample Predictions:")
print(sample_prediction)

# %% [markdown]
# # Final Summary

# %%
print("=" * 60)
print("TASK 12 - BINARY CLASSIFICATION COMPLETED")
print("=" * 60)

print(f"Best Threshold : {best_threshold:.4f}")
print(f"ROC-AUC Score  : {roc_auc:.4f}")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")

print("\nOutput Files Generated:")
print("- calibration_curve.png")
print("- threshold_plot.png")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- calibrated_model.pkl")

print("\nTask completed successfully!")

# %%



