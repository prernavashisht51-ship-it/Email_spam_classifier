"""Email spam classifier: TF-IDF + Logistic Regression / Naive Bayes.

Usage:
    pip install pandas scikit-learn joblib matplotlib
    python spam_classifier.py
Expects emails.csv (columns: text, label) in the same folder. label: 0 = ham, 1 = spam.
"""
import re
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " emailaddr ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=2)),
        ("clf", model),
    ])


def main():
    df = pd.read_csv(BASE_DIR / "emails.csv").dropna(subset=["text", "label"])
    df["clean"] = df["text"].apply(clean_text)
    print("Dataset shape:", df.shape)
    print(df["label"].value_counts().rename({0: "ham", 1: "spam"}), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }
    best_name, best_acc, best_pipe = None, 0, None
    for name, model in models.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"=== {name} === accuracy: {acc:.4f}")
        print(classification_report(y_test, pred, target_names=["ham", "spam"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, pred), "\n")
        if acc > best_acc:
            best_name, best_acc, best_pipe = name, acc, pipe

    print(f"Best model: {best_name} ({best_acc:.4f})")
    joblib.dump(best_pipe, BASE_DIR / "spam_model.joblib")
    print("Model saved to:", BASE_DIR / "spam_model.joblib")

    # Try it on new emails
    samples = [
        "Congratulations! You won a free prize, click http://win.example.com now to claim cash",
        "Hi team, attaching the meeting notes and the gas nomination schedule for tomorrow.",
    ]
    for s in samples:
        label = best_pipe.predict([clean_text(s)])[0]
        print(f"{'SPAM' if label == 1 else 'HAM '} <- {s[:70]}")


if __name__ == "__main__":
    main()