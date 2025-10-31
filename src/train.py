"""Train a simple TF-IDF + LogisticRegression classifier for emotion detection.

Usage:
    python -m src.train train

This script reads `data/emotion_small.csv`, trains a classifier, and writes
models to `models/` (joblib files).
"""
import os
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.utils import preprocess_text


DATA_PATH = os.path.join('data', 'emotion_small.csv')
MODELS_DIR = 'models'


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = df.dropna(subset=['text', 'label'])
    return df['text'].tolist(), df['label'].tolist()


def train_and_save():
    texts, labels = load_data()
    texts = [preprocess_text(t) for t in texts]
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X_train_t = vectorizer.fit_transform(X_train)
    X_test_t = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_t, y_train)

    preds = clf.predict(X_test_t)
    print('Evaluation on holdout set:')
    print(classification_report(y_test, preds))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    joblib.dump(clf, os.path.join(MODELS_DIR, 'sk_model.joblib'))
    print(f'Model and vectorizer saved to {MODELS_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['train'], help='train: train and save model')
    args = parser.parse_args()
    if args.cmd == 'train':
        train_and_save()
