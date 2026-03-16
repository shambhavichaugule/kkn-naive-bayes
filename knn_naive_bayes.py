import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

# Load same dataset as SVM project
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
df = dataset['train'].to_pandas()

print("Shape:", df.shape)
print("\nTarget distribution:")
print(df['label'].value_counts())

# Features and target
X = df['text']
y = df['label']

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF — same config as best SVM experiment
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Feature matrix shape: {X_train_tfidf.shape}")

# ── KNN ──────────────────────────────────────────────────
print("\n" + "="*50)
print("KNN RESULTS")
print("="*50)

# Start with k=5 (5 nearest neighbors)
knn_model = KNeighborsClassifier(n_neighbors=20, metric='cosine')
knn_model.fit(X_train_tfidf, y_train)
knn_pred = knn_model.predict(X_test_tfidf)

print(f"Accuracy: {accuracy_score(y_test, knn_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, knn_pred,
      target_names=['Negative', 'Positive', 'Neutral']))

# Find best k
# print("\nFinding best k...")
# k_values = [1, 5, 20, 50, 100]
# results = []

# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
#     knn.fit(X_train_tfidf, y_train)
#     pred = knn.predict(X_test_tfidf)
#     acc = accuracy_score(y_test, pred)
#     results.append((k, acc))
#     print(f"k={k:2d} | accuracy={acc:.4f}")

# best_k = max(results, key=lambda x: x[1])
# print(f"\nBest k: {best_k[0]} with accuracy: {best_k[1]:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, pred,
#       target_names=['Negative', 'Positive', 'Neutral']))

# ── NAIVE BAYES ──────────────────────────────────────────
print("\n" + "="*50)
print("NAIVE BAYES RESULTS")
print("="*50)

# nb_model = MultinomialNB()
# nb_model.fit(X_train_tfidf, y_train)
# nb_pred = nb_model.predict(X_test_tfidf)

# Calculate class weights manually
neg_prior = 1 / df['label'].value_counts(normalize=True)[0]
pos_prior = 1 / df['label'].value_counts(normalize=True)[1]
neu_prior = 1 / df['label'].value_counts(normalize=True)[2]

total = neg_prior + pos_prior + neu_prior

nb_model = MultinomialNB(
    class_prior=[neg_prior/total, pos_prior/total, neu_prior/total]
)

nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, nb_pred,
      target_names=['Negative', 'Positive', 'Neutral']))

# Confusion Matrix
cm = confusion_matrix(y_test, nb_pred)
print("\nConfusion Matrix:")
print(cm)

# Use ComplementNB to balance classes
print("\n" + "="*50)
print("COMPLEMENT NAIVE BAYES RESULTS")
print("="*50)

cnb_model = ComplementNB()
cnb_model.fit(X_train_tfidf, y_train)
cnb_pred = cnb_model.predict(X_test_tfidf)

print("\nComplement Naive Bayes Results:")
print(f"Accuracy: {accuracy_score(y_test, cnb_pred):.4f}")
print(classification_report(y_test, cnb_pred,
      target_names=['Negative', 'Positive', 'Neutral']))

# ── FINAL COMPARISON ─────────────────────────────────────
print("\n" + "="*50)
print("FULL MODEL COMPARISON")
print("="*50)

from sklearn.metrics import f1_score

models = {
    'KNN (k=20)': knn_pred,
    'MultinomialNB': nb_pred,
    'ComplementNB': cnb_pred
}

print(f"\n{'Model':<20} {'Accuracy':>10} {'Neg Recall':>12} {'Pos Recall':>12} {'Neu Recall':>12} {'Macro F1':>10}")
print("-" * 80)

for name, pred in models.items():
    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    neg_recall = report['0']['recall']
    pos_recall = report['1']['recall']
    neu_recall = report['2']['recall']
    macro_f1 = report['macro avg']['f1-score']
    print(f"{name:<20} {acc:>10.4f} {neg_recall:>12.4f} {pos_recall:>12.4f} {neu_recall:>12.4f} {macro_f1:>10.4f}")