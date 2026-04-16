# ============================================================
#   ASSIGNMENT 4 — Decision Tree Classifier
#   Case Study: Song Release Success Prediction
#   Library: scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# -----------------------------------------------
# DATASET — Song Features
# -----------------------------------------------
np.random.seed(42)
n = 300

data = pd.DataFrame({
    'Tempo':          np.random.randint(60, 200, n),       # BPM
    'Energy':         np.random.uniform(0.1, 1.0, n),      # 0-1
    'Danceability':   np.random.uniform(0.1, 1.0, n),      # 0-1
    'Loudness':       np.random.uniform(-20, 0, n),        # dB
    'Valence':        np.random.uniform(0.1, 1.0, n),      # positivity 0-1
    'Duration_sec':   np.random.randint(120, 300, n),      # seconds
    'Artist_Followers': np.random.randint(1000, 5000000, n),
})

# Realistic hit logic
data['Is_Hit'] = (
    (data['Energy'] > 0.6) &
    (data['Danceability'] > 0.5) &
    (data['Artist_Followers'] > 500000) &
    (data['Valence'] > 0.4)
).astype(int)  # 1 = Hit, 0 = Flop

print("=" * 55)
print("      SONG RELEASE SUCCESS PREDICTION")
print("=" * 55)
print(f"\n Dataset Shape : {data.shape}")
print(f" Hits          : {data['Is_Hit'].sum()}")
print(f" Flops         : {n - data['Is_Hit'].sum()}")
print("\n Sample Data:")
print(data.head())

# -----------------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------------
X = data.drop('Is_Hit', axis=1)
y = data['Is_Hit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# DECISION TREE CLASSIFIER
# -----------------------------------------------
print("\n" + "=" * 55)
print("   DECISION TREE CLASSIFIER")
print("=" * 55)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print(f"\n Training Samples : {len(X_train)}")
print(f" Testing Samples  : {len(X_test)}")
print(f"\n Accuracy Score   : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Flop', 'Hit']))

# Feature Importance
print("\n Feature Importances:")
for feat, imp in sorted(zip(X.columns, dt.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"   {feat:<22} : {imp:.4f}")

# -----------------------------------------------
# PLOTS
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Assignment 4 — Decision Tree: Song Hit Prediction", fontsize=13)

# Plot 1 — Decision Tree Structure
plot_tree(dt, feature_names=X.columns,
          class_names=['Flop', 'Hit'],
          filled=True, rounded=True,
          ax=axes[0], fontsize=7)
axes[0].set_title("Decision Tree Structure")

# Plot 2 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Flop', 'Hit'])
disp.plot(ax=axes[1], colorbar=False, cmap='Purples')
axes[1].set_title("Confusion Matrix")

# Plot 3 — Feature Importances
feat_imp = pd.Series(dt.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', ax=axes[2], color='mediumpurple')
axes[2].set_title("Feature Importances")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("A4_output.png", dpi=150)
plt.show()

print("\n Plot saved as A4_output.png ✓")
print(" Assignment 4 Complete! ✓")