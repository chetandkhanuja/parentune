"""
Parentune Conversational Intelligence — Model Training Script
IIM Amritsar — NLP Term IV

Trains the Module 1 topic classifier (TF-IDF + Logistic Regression) on
data/training_queries.csv and saves it to models/topic_classifier.joblib.

Run this ONCE before starting the Streamlit app, and again any time you
update training_queries.csv. app.py never trains anything itself — it just
loads the file this script produces.

Usage:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "topic_classifier.joblib")


def train_and_save():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "training_queries.csv"))
    print(f"Loaded {len(train_df)} labeled examples across "
          f"{train_df['category'].nunique()} categories: "
          f"{sorted(train_df['category'].unique())}")

    # Held-out split purely to report a real accuracy number for your writeup —
    # the deployed model below is refit on all the data.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        train_df["text"], train_df["category"],
        test_size=0.2, random_state=42, stratify=train_df["category"]
    )

    eval_vectorizer = TfidfVectorizer(stop_words="english")
    X_train = eval_vectorizer.fit_transform(X_train_text)
    X_test = eval_vectorizer.transform(X_test_text)

    eval_clf = LogisticRegression(max_iter=1000)
    eval_clf.fit(X_train, y_train)
    preds = eval_clf.predict(X_test)

    print(f"\nHeld-out accuracy: {accuracy_score(y_test, preds):.2%}")
    print(classification_report(y_test, preds, zero_division=0))

    # Now refit on the FULL dataset (train+test combined) — this is the model
    # that actually gets saved and used by the app, so it benefits from every
    # labeled example you have, not just 80% of them.
    final_vectorizer = TfidfVectorizer(stop_words="english")
    X_full = final_vectorizer.fit_transform(train_df["text"])

    final_clf = LogisticRegression(max_iter=1000)
    final_clf.fit(X_full, train_df["category"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": final_vectorizer,
            "clf": final_clf,
            "categories": sorted(train_df["category"].unique().tolist()),
        },
        MODEL_PATH,
    )
    print(f"\nSaved trained model to: {MODEL_PATH}")
    print("app.py will now load this file instead of retraining.")


if __name__ == "__main__":
    train_and_save()
