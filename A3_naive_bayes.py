# ============================================================
#   ASSIGNMENT 3 — Naive Bayes Classifier + Evaluate
#   Case Study: SMS Spam Detection
#   Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
#   Library: scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

# -----------------------------------------------
# DATASET — SMS Messages
# -----------------------------------------------
messages = [
    # SPAM
    "Win a free iPhone now click here",
    "You have won 1 million dollars claim now",
    "Free entry to win cash prize call now",
    "Congratulations you are selected for free gift",
    "Get rich quick scheme limited time offer",
    "Click here to claim your free vacation",
    "You are our lucky winner of 500 pounds",
    "Free ringtones download now limited offer",
    "Urgent your account will be suspended click now",
    "Buy cheap meds online no prescription needed",
    "Hot singles in your area click to meet",
    "You have been chosen for exclusive cash reward",
    "Send your bank details to claim lottery prize",
    "Double your income working from home click here",
    "Free credit score check limited time only",
    "Alert your card has been compromised click now",
    "Win big at our online casino join free",
    "Exclusive deal just for you expires tonight",
    "Claim your prize before it expires today",
    "Your phone number won our weekly draw",
    # HAM
    "Hey are you coming to the party tonight",
    "Can you pick up some milk on the way home",
    "The meeting has been rescheduled to tomorrow",
    "Happy birthday hope you have a great day",
    "I will be late for dinner see you at 8",
    "Did you finish the assignment for tomorrow",
    "Let me know when you reach home safely",
    "Can we reschedule our lunch to next week",
    "The movie starts at 7 see you there",
    "Thanks for your help yesterday really appreciated",
    "Are you free this weekend for a walk",
    "I sent you the document check your email",
    "Mom says dinner is ready come home soon",
    "Just checking in how are you doing today",
    "The train is delayed will be there in 20",
    "Can you call me when you get a chance",
    "Looking forward to seeing you at the reunion",
    "Please bring your notes to class tomorrow",
    "Got your message will reply later tonight",
    "Great job on the presentation today well done",
]

labels = ['spam'] * 20 + ['ham'] * 20

df = pd.DataFrame({'Message': messages, 'Label': labels})

print("=" * 55)
print("        SMS SPAM DETECTION — NAIVE BAYES")
print("=" * 55)
print(f"\n Total Messages : {len(df)}")
print(f" Spam           : {df['Label'].value_counts()['spam']}")
print(f" Ham            : {df['Label'].value_counts()['ham']}")
print("\n Sample Messages:")
print(df.head(6).to_string(index=False))

# -----------------------------------------------
# FEATURE EXTRACTION — TF-IDF
# -----------------------------------------------
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X = tfidf.fit_transform(df['Message'])
y = df['Label']

print(f"\n Vocabulary Size  : {len(tfidf.vocabulary_)}")
print(f" Feature Matrix   : {X.shape}")

# -----------------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------------------------
# NAIVE BAYES CLASSIFIER
# -----------------------------------------------
print("\n" + "=" * 55)
print("   MULTINOMIAL NAIVE BAYES CLASSIFIER")
print("=" * 55)

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# -----------------------------------------------
# EVALUATION METRICS
# -----------------------------------------------
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='spam')
rec  = recall_score(y_test, y_pred, pos_label='spam')
f1   = f1_score(y_test, y_pred, pos_label='spam')

print(f"\n Training Samples : {len(y_train)}")
print(f" Testing Samples  : {len(y_test)}")
print(f"\n --- Evaluation Metrics ---")
print(f" Accuracy         : {acc  * 100:.2f}%")
print(f" Precision        : {prec * 100:.2f}%")
print(f" Recall           : {rec  * 100:.2f}%")
print(f" F1 Score         : {f1   * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------------------------
# TEST ON NEW MESSAGES
# -----------------------------------------------
print("\n" + "=" * 55)
print("   REAL-TIME PREDICTION")
print("=" * 55)

new_messages = [
    "Congratulations you won a free iPhone click now",
    "Hey can we meet for coffee tomorrow morning",
    "Claim your cash prize before it expires today",
    "Are you coming to class tomorrow",
    "Free money just enter your bank details here",
]

new_tfidf = tfidf.transform(new_messages)
new_preds = nb.predict(new_tfidf)
new_probs = nb.predict_proba(new_tfidf)

for msg, pred, prob in zip(new_messages, new_preds, new_probs):
    emoji = "🚨 SPAM" if pred == 'spam' else "✅ HAM"
    print(f"\n Message : {msg[:45]}...")
    print(f" Result  : {emoji}")
    print(f" Confidence : {max(prob)*100:.1f}%")

# -----------------------------------------------
# PLOTS
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Assignment 3 — Naive Bayes: SMS Spam Detection", fontsize=13)

# Plot 1 — Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=['spam', 'ham'])
disp = ConfusionMatrixDisplay(cm, display_labels=['Spam', 'Ham'])
disp.plot(ax=axes[0], colorbar=False, cmap='Reds')
axes[0].set_title("Confusion Matrix")

# Plot 2 — Metrics Bar Chart
metrics = {
    'Accuracy':  acc,
    'Precision': prec,
    'Recall':    rec,
    'F1 Score':  f1
}
bars = axes[1].bar(metrics.keys(), metrics.values(),
                   color=['steelblue', 'coral', 'mediumseagreen', 'mediumpurple'],
                   edgecolor='k', alpha=0.85)
axes[1].set_title("Evaluation Metrics")
axes[1].set_ylabel("Score")
axes[1].set_ylim(0, 1.15)
for bar, val in zip(bars, metrics.values()):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val*100:.1f}%", ha='center', fontsize=10)

# Plot 3 — Top Spam vs Ham words
spam_indices = [i for i, l in enumerate(labels) if l == 'spam']
ham_indices  = [i for i, l in enumerate(labels) if l == 'ham']

spam_text = ' '.join([messages[i] for i in spam_indices])
ham_text  = ' '.join([messages[i] for i in ham_indices])

spam_vec = tfidf.transform([spam_text]).toarray()[0]
ham_vec  = tfidf.transform([ham_text]).toarray()[0]

feature_names = tfidf.get_feature_names_out()
top_n = 8

top_spam_idx = spam_vec.argsort()[-top_n:][::-1]
top_ham_idx  = ham_vec.argsort()[-top_n:][::-1]

top_spam_words  = [feature_names[i] for i in top_spam_idx]
top_spam_scores = [spam_vec[i] for i in top_spam_idx]
top_ham_words   = [feature_names[i] for i in top_ham_idx]
top_ham_scores  = [ham_vec[i] for i in top_ham_idx]

x = np.arange(top_n)
width = 0.35
axes[2].bar(x - width/2, top_spam_scores, width,
            label='Spam', color='coral', edgecolor='k', alpha=0.85)
axes[2].bar(x + width/2, top_ham_scores, width,
            label='Ham', color='steelblue', edgecolor='k', alpha=0.85)
axes[2].set_title("Top Words — Spam vs Ham")
axes[2].set_xticks(x)
axes[2].set_xticklabels(top_spam_words, rotation=35, ha='right', fontsize=8)
axes[2].set_ylabel("TF-IDF Score")
axes[2].legend()

plt.tight_layout()
plt.savefig("A3_output.png", dpi=150)
plt.show()

print("\n Plot saved as A3_output.png ✓")
print(" Assignment 3 Complete! ✓")