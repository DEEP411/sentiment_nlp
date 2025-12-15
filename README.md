# Sentiment Analysis & NLP Pipeline

This project implements an end‑to‑end pipeline for classifying text sentiment.  
It demonstrates how to preprocess raw text, extract features, train a classifier, evaluate its performance, and deploy it for inference.  

The résumé entry referenced using a **BERT** model and packaging the service with Flask and Docker.  To keep this project lightweight and self‑contained, the provided implementation uses scikit‑learn with a **logistic regression** classifier and **TF‑IDF** features.  On the synthetic dataset included here, this simple model achieves perfect accuracy.

If you wish to experiment with transformer‑based models (e.g., BERT) for higher accuracy on real‑world data, you can install the `transformers` library and modify `train_nlp.py` accordingly.  In the absence of an internet connection on this environment, that step is left to the user.

## Requirements

The project uses the following Python packages:

```
numpy
pandas
scikit‑learn
nltk
```

Install dependencies with:

```
pip install numpy pandas scikit‑learn nltk
```

## Project Structure

```
sentiment_nlp/
├── data/
│   └── sentiment_data.csv    # synthetic dataset of text and labels
├── train_nlp.py             # trains the text classification model
├── predict_nlp.py           # loads the model and predicts sentiment for new input
├── requirements.txt         # pinned dependencies
└── README.md                # this file
```

## Usage

1. **Generate data and train model**

   Run `train_nlp.py` to create a synthetic sentiment dataset, train a logistic regression classifier using TF‑IDF features, and evaluate it.  The script reports accuracy and F1‑score and saves the trained model and vectorizer to disk.

   ```bash
   cd sentiment_nlp
   python train_nlp.py
   ```

2. **Classify new text**

   Use `predict_nlp.py` to load the saved model and vectorizer and classify new input sentences provided via the command line.

   ```bash
   python predict_nlp.py "I really enjoyed the service and will come back again!"
   ```

## Notes

* The sample dataset is synthetic and intended for demonstration.  Replace `data/sentiment_data.csv` with your own labeled dataset (columns: `text`, `label`) to train on real data.
* The script uses **scikit‑learn** instead of BERT due to environment constraints.  To use a BERT model, install a library like `transformers` and modify `train_nlp.py` accordingly.
* The simple command‑line interface in `predict_nlp.py` can be extended into a REST API (e.g., using Flask or FastAPI) and containerized with a `Dockerfile` for production deployment.