# KNN & Naive Bayes — Twitter Financial News Sentiment Analysis

## Project Summary

Built and compared K-Nearest Neighbors (KNN) and Naive Bayes classifiers for financial tweet sentiment analysis. This project uses the same dataset as the SVM project — enabling direct comparison across all classical ML algorithms on the same task.

**Dataset:** [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

**Business Problem:** Can we automatically classify the sentiment of financial news tweets — so traders, analysts, and product teams can monitor market sentiment at scale?

**Previous Project:** [SVM — Twitter Financial News Sentiment](https://github.com/shambhavichaugule/svm)

---

## What I Built

Two classifiers (KNN and Naive Bayes) predicting:
- `0` → Negative sentiment
- `1` → Positive sentiment
- `2` → Neutral sentiment

---

## Key Learnings

### 1. What is KNN?

KNN (K-Nearest Neighbors) is the simplest ML algorithm — it doesn't build a model at all. It just memorizes all training data and for every new point finds the K most similar examples and takes a majority vote.

```
New tweet arrives
→ Find 20 most similar tweets in training data
→ 14 are Neutral, 4 are Positive, 2 are Negative
→ Predict Neutral (majority vote)
```

**No training phase** — just memorization. This makes it:
- Very fast to "train" (nothing to train)
- Very slow to predict (must compare against all training examples)
- Memory intensive — must store entire training dataset

### 2. Cosine Similarity for Text

For numeric data KNN uses euclidean distance (straight line between points).

For text data we use **cosine similarity** — measures the angle between two tweet vectors:

```python
KNeighborsClassifier(n_neighbors=20, metric='cosine')
```

```
Short tweet: "stock upgraded"
Long tweet:  "stock upgraded by Goldman Sachs with price target raised"
```

Euclidean distance says these are far apart because one is longer.
Cosine similarity correctly identifies them as similar — they point in the same direction.

### 3. Finding The Best K — Bias Variance Tradeoff

Tested multiple values of K:

```
k=1   → 73% — overfitting, too sensitive to individual noisy points
k=5   → 76% — better generalization
k=20  → 77% — sweet spot ✅
k=50  → 73% — underfitting, too many neighbors dilute signal
k=100 → 71% — signal almost completely diluted
```

Same bias-variance tradeoff as `max_depth` in decision trees — just a different parameter name.

**Best k = 20**

### 4. KNN Limitation — No Class Weight Support

KNN has no `class_weight='balanced'` option unlike other sklearn models. This makes it inherently biased towards majority classes:

```
Neutral recall:  97% — excellent (65% of training data)
Negative recall: 30% — poor (only 15% of training data)
```

For imbalanced datasets KNN will always favour the majority class. This is a fundamental limitation, not a tuning problem.

### 5. What is Naive Bayes?

Naive Bayes is based on probability. For each tweet it asks:

```
P(Negative | words in tweet) = ?
P(Positive | words in tweet) = ?
P(Neutral  | words in tweet) = ?
```

Predict the class with highest probability.

Called **"naive"** because it assumes every word is independent — "price" and "cuts" are treated as unrelated even though "price cuts" means something specific together. Despite this unrealistic assumption it works surprisingly well for text.

### 6. Three Variants of Naive Bayes Tested

**MultinomialNB** — standard Naive Bayes for text:
```
Accuracy: 58%
Negative recall: 79% — catches most negative tweets
Neutral recall:  48% — fails badly on majority class
```
Goes the opposite extreme — flags too many tweets as negative/positive, misses neutral.

**ComplementNB** — designed specifically for imbalanced text:
```
Accuracy: 80%
Negative recall: 53%
Neutral recall:  89%
```

Instead of learning "what words predict this class" it learns "what words predict everything **except** this class." Much better balance for imbalanced datasets.

### 7. Full Model Comparison — All Algorithms

Using identical dataset, split, and TF-IDF configuration across all models:

| Model | Accuracy | Neg Recall | Pos Recall | Neu Recall | Macro F1 |
|---|---|---|---|---|---|
| SVM Linear (baseline) | 78% | 68% | 71% | 83% | 0.72 |
| SVM Linear (tuned) | 80% | 69% | 72% | 85% | 0.74 |
| SVM RBF (tuned) | 82% | 56% | 66% | 93% | 0.74 |
| KNN (k=20) | 77% | 30% | 50% | 97% | 0.63 |
| MultinomialNB | 58% | 79% | 73% | 48% | 0.57 |
| ComplementNB | 80% | 53% | 69% | 89% | 0.72 |

### 8. Algorithm Personalities

Each algorithm has a distinct personality that makes it suitable for different problems:

| Algorithm | Personality | Best Used When |
|---|---|---|
| KNN | Lazy, intuitive, biased towards majority | Small balanced datasets |
| MultinomialNB | Trigger happy, catches minority classes aggressively | When recall matters more than precision |
| ComplementNB | Balanced, handles imbalance well | Imbalanced text classification |
| SVM | Principled, finds best boundary | Most text classification tasks |

### 9. Which Model To Ship?

**For a financial sentiment product — SVM Linear tuned wins:**
- 80% accuracy
- 69% negative recall — best among production viable models
- Fast at prediction time
- Interpretable — can explain why a tweet was classified negative

MultinomialNB has higher negative recall (79%) but 58% overall accuracy makes it unreliable — too many false alarms.

**Key PM insight:** The best model is not always the most accurate one — it's the one that best serves the business goal with acceptable tradeoffs.

---

## Final Model Configuration

```python
# TF-IDF — same as SVM project
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

# KNN
knn_model = KNeighborsClassifier(
    n_neighbors=20,
    metric='cosine'
)

# Complement Naive Bayes
cnb_model = ComplementNB()
```

---

## Tools & Libraries

```python
datasets                               # Hugging Face dataset loading
pandas                                 # Data manipulation
numpy                                  # Numerical operations
scikit-learn                           # Model training and evaluation
sklearn.neighbors.KNeighborsClassifier # KNN
sklearn.naive_bayes.MultinomialNB      # Multinomial Naive Bayes
sklearn.naive_bayes.ComplementNB       # Complement Naive Bayes
sklearn.feature_extraction.TfidfVectorizer  # Text vectorization
matplotlib                             # Visualisation
python-dotenv                          # Environment variable management
huggingface_hub                        # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/shambhavichaugule/knn-naive-bayes.git
cd knn-naive-bayes

# Activate virtual environment
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Run the models
python knn_naive_bayes.py
```

---

## PM Perspective

This project simulates a real product decision: **which algorithm should power our financial sentiment feature?**

**Algorithm selection is a product decision:**
- KNN is intuitive and easy to explain to stakeholders but fails on minority classes
- Naive Bayes is lightning fast in production — ideal for real time tweet classification
- SVM is slower but more reliable — better for batch processing

**Production considerations:**
- KNN stores entire training dataset in memory — expensive at scale with millions of tweets
- Naive Bayes is extremely fast — can classify thousands of tweets per second
- ComplementNB requires no hyperparameter tuning — lower maintenance cost

**What I would do as a PM:**
1. Define minimum negative recall threshold before choosing a model — e.g. must catch 60% of negative tweets
2. KNN fails this threshold at 30% — eliminate it
3. MultinomialNB passes recall but fails accuracy — eliminate it
4. Choose between ComplementNB (53% neg recall) and SVM tuned (69% neg recall)
5. Ship SVM tuned — best negative recall among production viable models

---

## Classical ML Journey — Complete

This project completes the classical supervised learning journey:

| Project | Algorithm | Dataset | Best Metric |
|---|---|---|---|
| [Linear Regression](https://github.com/shambhavichaugule/linear-regression-project) | Linear Regression | Car prices | R²=0.09 |
| [Logistic Regression](https://github.com/shambhavichaugule/logistic-regression-project) | Logistic Regression | Shipment delays | 50% accuracy |
| [Decision Tree & Random Forest](https://github.com/shambhavichaugule/decision-tree-random-forest) | DT + RF | Shipment delays | 56% accuracy |
| [SVM](https://github.com/shambhavichaugule/svm) | SVM + TF-IDF | Twitter sentiment | 80% accuracy |
| This project | KNN + Naive Bayes | Twitter sentiment | 80% accuracy |





---

*Built as part of a learning journey from Senior PM to AI PM — March 2026*
