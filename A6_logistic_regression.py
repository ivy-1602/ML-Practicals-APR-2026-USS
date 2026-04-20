# ============================================================
#   ASSIGNMENT 6 — Logistic Regression Binary Classifier
#   Case Study: Predicting Eye Color in Children
#   Based on: Parent eye colors, melanin levels, genetic markers
#   Library: scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)

# -----------------------------------------------
# DATASET — Eye Color Genetics Data
# -----------------------------------------------
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'Parent1_Melanin':    np.random.uniform(0.5, 1.0, n),  # brown = high melanin
    'Parent2_Melanin':    np.random.uniform(0.0, 0.6, n),  # hazel/blue = low
    'OCA2_Gene':          np.random.uniform(0, 1, n),       # key eye color gene
    'HERC2_Gene':         np.random.uniform(0, 1, n),       # controls OCA2
    'Child_Melanin':      np.random.uniform(0.0, 1.0, n),  # child's melanin
    'Dominant_Alleles':   np.random.randint(0, 3, n),       # 0,1,2
})

# Realistic logic:
# Blue/Hazel = low melanin + low OCA2 + low dominant alleles
data['Eye_Color'] = (
    (data['Child_Melanin'] < 0.45) &
    (data['OCA2_Gene'] < 0.5) &
    (data['Dominant_Alleles'] <= 1)
).astype(int)  # 1 = Blue/Hazel, 0 = Brown

print("=" * 55)
print("      EYE COLOR PREDICTION IN CHILDREN")
print("=" * 55)
print(f"\n Dataset Shape    : {data.shape}")
print(f" Blue/Hazel Eyes  : {data['Eye_Color'].sum()}")
print(f" Brown Eyes       : {n - data['Eye_Color'].sum()}")
print("\n Sample Data:")
print(data.head())

# -----------------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------------
X = data.drop('Eye_Color', axis=1)
y = data['Eye_Color']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------------------------
# LOGISTIC REGRESSION
# -----------------------------------------------
print("\n" + "=" * 55)
print("   LOGISTIC REGRESSION CLASSIFIER")
print("=" * 55)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred      = lr.predict(X_test_scaled)
y_pred_prob = lr.predict_proba(X_test_scaled)[:, 1]

print(f"\n Training Samples : {len(X_train)}")
print(f" Testing Samples  : {len(X_test)}")
print(f"\n Accuracy Score   : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Brown', 'Blue/Hazel']))

# Feature coefficients
print("\n Feature Coefficients:")
for feat, coef in sorted(zip(X.columns, lr.coef_[0]),
                          key=lambda x: abs(x[1]), reverse=True):
    print(f"   {feat:<22} : {coef:.4f}")

# -----------------------------------------------
# PREDICT FOR OUR SPECIFIC CASE
# Brown eyed parent + Hazel/Blue eyed parent
# -----------------------------------------------
print("\n" + "=" * 55)
print("   REAL CASE: Brown Parent + Hazel/Blue Parent")
print("=" * 55)

test_cases = pd.DataFrame({
    'Parent1_Melanin':  [0.85, 0.85, 0.75],
    'Parent2_Melanin':  [0.20, 0.35, 0.15],
    'OCA2_Gene':        [0.35, 0.55, 0.30],
    'HERC2_Gene':       [0.40, 0.60, 0.35],
    'Child_Melanin':    [0.30, 0.50, 0.25],
    'Dominant_Alleles': [1,    1,    0   ],
})

test_scaled = scaler.transform(test_cases)
predictions = lr.predict(test_scaled)
probabilities = lr.predict_proba(test_scaled)

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "Blue/Hazel 👁️" if pred == 1 else "Brown 👁️"
    print(f"\n Child {i+1}:")
    print(f"   Predicted Eye Color : {label}")
    print(f"   Brown probability   : {prob[0]*100:.1f}%")
    print(f"   Blue/Hazel prob     : {prob[1]*100:.1f}%")

# -----------------------------------------------
# PLOTS
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Assignment 6 — Logistic Regression: Eye Color Prediction",
             fontsize=13)

# Plot 1 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Brown', 'Blue/Hazel'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")

# Plot 2 — ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='gray', lw=1.5,
             linestyle='--', label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='steelblue')
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

# Plot 3 — Feature Coefficients
coef_series = pd.Series(lr.coef_[0], index=X.columns)
colors = ['coral' if c < 0 else 'steelblue' for c in coef_series.sort_values()]
coef_series.sort_values().plot(
    kind='barh', ax=axes[2], color=colors
)
axes[2].axvline(x=0, color='black', linewidth=0.8)
axes[2].set_title("Feature Coefficients\n(+ve = Blue/Hazel, -ve = Brown)")
axes[2].set_xlabel("Coefficient Value")

plt.tight_layout()
plt.savefig("A6_output.png", dpi=150)
plt.show()

print("\n Plot saved as A6_output.png ✓")
print(" Assignment 6 Complete! ✓")