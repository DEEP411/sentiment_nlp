"""Predict sentiment of input text.

This script loads a pre‑trained TF‑IDF vectorizer and logistic regression
model (saved by `train_nlp.py`) and predicts sentiment for text passed
via the command line.

Usage:
    python predict_nlp.py "Your text here"

Dependencies:
    numpy, pandas, scikit‑learn
"""

import os
import pickle
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def load_artifacts(model_dir: str):
    """Load the vectorizer and model from the given directory."""
    vec_path = os.path.join(model_dir, 'vectorizer.pkl')
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    if not os.path.exists(vec_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifacts not found in {model_dir}. Have you run train_nlp.py?")
    with open(vec_path, 'rb') as f:
        vectorizer: TfidfVectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        model: LogisticRegression = pickle.load(f)
    return vectorizer, model


def classify_text(text: str, vectorizer: TfidfVectorizer, model: LogisticRegression) -> str:
    """Predict sentiment label for the given text."""
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return 'positive' if prediction == 1 else 'negative'


def main(args):
    if len(args) < 2:
        print("Usage: python predict_nlp.py \"Your text here\"")
        return
    text = args[1]
    vectorizer, model = load_artifacts('model')
    sentiment = classify_text(text, vectorizer, model)
    print(f"Predicted sentiment: {sentiment}")


if __name__ == '__main__':
    main(sys.argv)