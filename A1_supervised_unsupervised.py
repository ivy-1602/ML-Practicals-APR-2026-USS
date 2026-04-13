# ============================================================
#   ASSIGNMENT 1 — Supervised & Unsupervised Machine Learning
#   Case Study: Customer Segmentation for a Retail Store
#   Dataset: Iris (Supervised) + Wine (Unsupervised)
#   Library: scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris, load_wine

# -----------------------------------------------
# PART 1 — SUPERVISED LEARNING (KNN on Iris)
# -----------------------------------------------
print("=" * 55)
print("   PART 1 : SUPERVISED LEARNING — KNN Classifier")
print("   Dataset : Iris Flower Dataset")
print("=" * 55)

# Load real dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print(f"\n Dataset Shape  : {X.shape}")
print(f" Classes        : {iris.target_names}")
print(f"\n Sample Data:")
print(X.head())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Results
print(f"\n Training Samples : {len(X_train)}")
print(f" Testing Samples  : {len(X_test)}")
print(f"\n Accuracy Score   : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=iris.target_names))

# -----------------------------------------------
# PART 2 — UNSUPERVISED LEARNING (KMeans on Wine)
# -----------------------------------------------
print("\n" + "=" * 55)
print("  PART 2 : UNSUPERVISED LEARNING — KMeans Clustering")
print("  Dataset : Wine Dataset")
print("=" * 55)

# Load real dataset
wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)

# Use 2 features for easy visualization
X_cluster = X_wine[['alcohol', 'color_intensity']]

# Scale
scaler2 = StandardScaler()
X_cluster_scaled = scaler2.fit_transform(X_cluster)

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster_scaled)

print(f"\n Dataset Shape  : {X_wine.shape}")
print(f" Number of Clusters : 3")
print("\n Customers per Cluster:")
print(pd.Series(clusters).value_counts().sort_index())

# -----------------------------------------------
# PLOTS
# -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Assignment 1 — Supervised & Unsupervised ML", fontsize=14)

# Plot 1 — Supervised: KNN on Iris (sepal length vs width)
scatter = axes[0].scatter(
    X_test['sepal length (cm)'],
    X_test['sepal width (cm)'],
    c=y_pred, cmap='Set1', alpha=0.8, edgecolors='k', s=80
)
axes[0].set_title("Supervised — KNN on Iris Dataset")
axes[0].set_xlabel("Sepal Length (cm)")
axes[0].set_ylabel("Sepal Width (cm)")
legend1 = axes[0].legend(
    handles=scatter.legend_elements()[0],
    labels=list(iris.target_names),
    title="Species"
)
axes[0].add_artist(legend1)

# Plot 2 — Unsupervised: KMeans on Wine
for i in range(3):
    mask = clusters == i
    axes[1].scatter(
        X_cluster['alcohol'][mask],
        X_cluster['color_intensity'][mask],
        label=f'Cluster {i}', alpha=0.7, edgecolors='k', s=80
    )
# Plot centroids (inverse transform back to original scale)
centers = scaler2.inverse_transform(kmeans.cluster_centers_)
axes[1].scatter(
    centers[:, 0], centers[:, 1],
    c='black', marker='X', s=250, zorder=5, label='Centroids'
)
axes[1].set_title("Unsupervised — KMeans on Wine Dataset")
axes[1].set_xlabel("Alcohol")
axes[1].set_ylabel("Color Intensity")
axes[1].legend()

plt.tight_layout()
plt.savefig("A1_output.png", dpi=150)
plt.show()

print("\n Plot saved as A1_output.png ✓")
print(" Assignment 1 Complete! ✓")
