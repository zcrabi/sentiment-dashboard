"""
TF-IDF özellik çıkarımı + 3 model eğitimi (Logistic Regression, Naive Bayes, Linear SVM)
Metrikleri karşılaştırır, en iyi modeli + vectorizer'ı diske kaydeder.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

BASE = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "..", "data", "reviews_clean.csv")
MODELS_DIR = os.path.join(BASE, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Veri yüklendi: {len(df)} satır")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("TF-IDF vektörleştirme...")
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, C=1.0),
        "naive_bayes": MultinomialNB(),
        "svm": LinearSVC(max_iter=2000),
    }

    results = {}
    best_name, best_f1, best_model = None, -1, None

    for name, model in models.items():
        print(f"\n=== {name} eğitiliyor ===")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
        }
        print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
        print("Confusion Matrix:", cm)

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    print(f"\n>>> En iyi model: {best_name} (F1={best_f1:.4f})")

    # Kaydet
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    for name, model in models.items():
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.joblib"))

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nModeller ve metrikler '{MODELS_DIR}' içine kaydedildi.")


if __name__ == "__main__":
    main()
