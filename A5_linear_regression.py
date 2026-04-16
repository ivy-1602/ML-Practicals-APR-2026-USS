# ============================================================
#   ASSIGNMENT 5 — Simple Linear Regression
#   Case Study: Predicting Student Scores Based on Study Hours
#   Metrics: MAE, MSE, RMSE, R²
#   Library: scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)

# -----------------------------------------------
# DATASET — Student Study Hours vs Scores
# -----------------------------------------------
np.random.seed(42)
n = 100

study_hours = np.random.uniform(0.5, 12, n)

scores = np.where(
    study_hours <= 2,
    85 + (study_hours * 3) + np.random.normal(0, 3, n),
    np.where(
        study_hours <= 5,
        90 - ((study_hours - 2) * 4) + np.random.normal(0, 4, n),
        78 - ((study_hours - 5) * 3) + np.random.normal(0, 5, n)
    )
)
scores = np.clip(scores, 0, 100)

data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Score':       scores
})

print("=" * 55)
print("   PREDICTING STUDENT SCORES — STUDY HOURS")
print("=" * 55)
print(f"\n Dataset Shape   : {data.shape}")
print(f" Avg Study Hours : {data['Study_Hours'].mean():.2f} hrs")
print(f" Avg Score       : {data['Score'].mean():.2f}%")
print("\n Sample Data:")
print(data.head())

# -----------------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------------
X = data[['Study_Hours']]
y = data['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# LINEAR REGRESSION
# -----------------------------------------------
print("\n" + "=" * 55)
print("   SIMPLE LINEAR REGRESSION")
print("=" * 55)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Metrics
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n Coefficient (slope)  : {lr.coef_[0]:.4f}")
print(f" Intercept            : {lr.intercept_:.4f}")
print(f"\n --- Evaluation Metrics ---")
print(f" MAE  (Mean Absolute Error)       : {mae:.4f}")
print(f" MSE  (Mean Squared Error)        : {mse:.4f}")
print(f" RMSE (Root Mean Squared Error)   : {rmse:.4f}")
print(f" R²   (R-Squared Score)           : {r2:.4f}")

# Predict for specific hours
print("\n --- Predictions for specific study hours ---")
for hrs in [1, 2, 3, 5, 7, 10]:
    pred = lr.predict([[hrs]])[0]
    print(f"   Study {hrs:>2} hrs → Predicted Score : {pred:.2f}%")

# -----------------------------------------------
# PLOTS
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Assignment 5 — Linear Regression: Student Score Prediction",
             fontsize=13)

# Plot 1 — Regression Line
axes[0].scatter(X_train, y_train, color='steelblue',
                alpha=0.6, label='Train Data', s=50)
axes[0].scatter(X_test, y_test, color='coral',
                alpha=0.8, label='Test Data', s=50)
x_line = np.linspace(0.5, 12, 200).reshape(-1, 1)
axes[0].plot(x_line, lr.predict(x_line),
             color='black', linewidth=2, label='Regression Line')
axes[0].set_title("Study Hours vs Score")
axes[0].set_xlabel("Study Hours")
axes[0].set_ylabel("Score (%)")
axes[0].legend()

# Plot 2 — Actual vs Predicted
axes[1].scatter(y_test, y_pred, color='mediumpurple',
                edgecolors='k', alpha=0.8, s=60)
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title("Actual vs Predicted Scores")
axes[1].set_xlabel("Actual Score")
axes[1].set_ylabel("Predicted Score")
axes[1].legend()

# Plot 3 — Metrics Bar Chart
metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
axes[2].bar(metrics.keys(), metrics.values(),
            color=['steelblue', 'coral', 'mediumseagreen'])
axes[2].set_title(f"Error Metrics  (R² = {r2:.4f})")
axes[2].set_ylabel("Error Value")
for i, (k, v) in enumerate(metrics.items()):
    axes[2].text(i, v + 0.3, f"{v:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("A5_output.png", dpi=150)
plt.show()

print("\n Plot saved as A5_output.png ✓")
print(" Assignment 5 Complete! ✓")