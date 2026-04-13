# ============================================================
#   ASSIGNMENT 2 — Feature Extraction Techniques
#   Case Study: Sentiment Analysis of Movie Reviews
#   Libraries: scikit-learn, pandas, matplotlib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------
# DATASET — Movie Reviews
# -----------------------------------------------
reviews = [
    "This movie was absolutely fantastic and amazing",
    "Terrible film, waste of time and money",
    "Great acting and wonderful storyline",
    "Boring and slow, did not enjoy it at all",
    "One of the best movies I have ever seen",
    "Awful direction and poor screenplay",
    "Brilliant cinematography and superb performance",
    "Disappointing ending ruined the whole experience",
    "Loved every minute of this masterpiece",
    "Horrible plot with bad acting throughout",
    "Outstanding film with great emotional depth",
    "Dull and predictable, not recommended at all",
    "Incredible visuals and gripping storyline",
    "Very bad movie, completely boring experience",
    "Heartwarming story with excellent performances",
]

sentiments = [
    'positive', 'negative', 'positive', 'negative', 'positive',
    'negative', 'positive', 'negative', 'positive', 'negative',
    'positive', 'negative', 'positive', 'negative', 'positive',
]

df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

print("=" * 55)
print("    SENTIMENT ANALYSIS — MOVIE REVIEWS")
print("=" * 55)
print(f"\n Total Reviews  : {len(df)}")
print(f" Positive       : {df['Sentiment'].value_counts()['positive']}")
print(f" Negative       : {df['Sentiment'].value_counts()['negative']}")
print("\n Sample Reviews:")
print(df.head())

# -----------------------------------------------
# FEATURE EXTRACTION 1 — Bag of Words (CountVectorizer)
# -----------------------------------------------
print("\n" + "=" * 55)
print("  TECHNIQUE 1 : Bag of Words (CountVectorizer)")
print("=" * 55)

bow = CountVectorizer(stop_words='english')
X_bow = bow.fit_transform(df['Review'])

bow_df = pd.DataFrame(
    X_bow.toarray(),
    columns=bow.get_feature_names_out()
)
print(f"\n Vocabulary Size : {len(bow.vocabulary_)}")
print(f" Feature Matrix  : {X_bow.shape}")
print("\n Top 10 words:")
word_counts = bow_df.sum().sort_values(ascending=False)
print(word_counts.head(10))

# -----------------------------------------------
# FEATURE EXTRACTION 2 — TF-IDF
# -----------------------------------------------
print("\n" + "=" * 55)
print("  TECHNIQUE 2 : TF-IDF (Term Frequency-Inverse Document Frequency)")
print("=" * 55)

tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(df['Review'])

tfidf_df = pd.DataFrame(
    X_tfidf.toarray(),
    columns=tfidf.get_feature_names_out()
)
print(f"\n Vocabulary Size : {len(tfidf.vocabulary_)}")
print(f" Feature Matrix  : {X_tfidf.shape}")
print("\n Top 10 TF-IDF weighted words:")
tfidf_scores = tfidf_df.sum().sort_values(ascending=False)
print(tfidf_scores.head(10))

# -----------------------------------------------
# FEATURE EXTRACTION 3 — PCA (Dimensionality Reduction)
# -----------------------------------------------
print("\n" + "=" * 55)
print("  TECHNIQUE 3 : PCA (Dimensionality Reduction)")
print("=" * 55)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

print(f"\n Original Features  : {X_tfidf.shape[1]}")
print(f" Reduced Features   : 2")
print(f" Variance Explained : {pca.explained_variance_ratio_.sum() * 100:.2f}%")

# -----------------------------------------------
# PLOTS
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Assignment 2 — Feature Extraction for Sentiment Analysis", fontsize=13)

# Plot 1 — Top words BoW
top_words = word_counts.head(10)
axes[0].barh(top_words.index[::-1], top_words.values[::-1], color='steelblue')
axes[0].set_title("Bag of Words — Top 10 Words")
axes[0].set_xlabel("Frequency")

# Plot 2 — Top TF-IDF scores
top_tfidf = tfidf_scores.head(10)
axes[1].barh(top_tfidf.index[::-1], top_tfidf.values[::-1], color='coral')
axes[1].set_title("TF-IDF — Top 10 Words")
axes[1].set_xlabel("TF-IDF Score")

# Plot 3 — PCA visualization
colors = ['green' if s == 'positive' else 'red' for s in sentiments]
axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=100,
                edgecolors='k', alpha=0.8)
axes[2].set_title("PCA — Review Clusters")
axes[2].set_xlabel("Principal Component 1")
axes[2].set_ylabel("Principal Component 2")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Positive'),
    Patch(facecolor='red',   label='Negative')
]
axes[2].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig("A2_output.png", dpi=150)
plt.show()

print("\n Plot saved as A2_output.png ✓")
print(" Assignment 2 Complete! ✓")