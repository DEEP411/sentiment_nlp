"""Train a sentiment analysis model.

This script generates a synthetic text sentiment dataset, trains a
logistic regression classifier using TF‑IDF features, evaluates its
accuracy and F1‑score, and saves the trained model and vectorizer.

Usage:
    python train_nlp.py

Dependencies:
    numpy, pandas, scikit‑learn, nltk
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def generate_synthetic_sentiment_data() -> pd.DataFrame:
    
    positive_sentences = [
        "I love this product!",
        "Amazing service and friendly staff.",
        "Great experience overall.",
        "I enjoyed the meal and the ambiance.",
        "Absolutely fantastic!", 
        "Brilliant workmanship and quality.",
        "Superb performance and reliability.",
        "Highly recommend this to everyone.",
        "Delighted with the results.",
        "Everything exceeded my expectations."
    ]
    negative_sentences = [
        "I hate this product.",
        "Terrible experience, very disappointed.",
        "Not good at all; would not recommend.",
        "Service was poor and rude.",
        "The food was bland and tasteless.",
        "Extremely disappointed with the performance.",
        "It broke after one use, terrible quality.",
        "Worst purchase I’ve ever made.",
        "Far below my expectations.",
        "I will never buy this again."
    ]
    # Duplicate sentences to enlarge dataset
    texts = positive_sentences * 5 + negative_sentences * 5
    labels = [1] * (len(positive_sentences) * 5) + [0] * (len(negative_sentences) * 5)
    df = pd.DataFrame({'text': texts, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


def save_dataset(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Synthetic sentiment dataset saved to {out_path}")


def train_sentiment_model(df: pd.DataFrame):
    
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tf, y_train)

    y_pred = model.predict(X_test_tf)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return vectorizer, model, {'accuracy': accuracy, 'f1': f1}


def save_artifacts(vectorizer, model, model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    vec_path = os.path.join(model_dir, 'vectorizer.pkl')
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved vectorizer to {vec_path}")
    print(f"Saved model to {model_path}")


def main():
    df = generate_synthetic_sentiment_data()
    # Save dataset
    save_dataset(df, os.path.join('data', 'sentiment_data.csv'))
    # Train model
    vectorizer, model, metrics = train_sentiment_model(df)
    # Save artifacts
    save_artifacts(vectorizer, model, 'model')
    # Print metrics
    print("\nModel performance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1‑score: {metrics['f1']:.3f}")

    # Save metrics to file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('model_metrics.csv', index=False)
    print("Metrics saved to model_metrics.csv")


if __name__ == '__main__':
    main()
