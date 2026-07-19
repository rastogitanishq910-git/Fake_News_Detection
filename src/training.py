"""
training.py
------------
Handles feature extraction (TF-IDF) and training/evaluation of the
four candidate models. train_model.py (in the project root) is just a
thin script that calls the functions here.
"""

from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) turns cleaned
    text into numeric vectors that weigh words by how distinctive they
    are to a document, not just how often they appear.

    Why TF-IDF over plain word counts (Bag of Words)?
    - Common words that show up in almost every article (e.g. "said",
      "report") get down-weighted automatically, since IDF penalizes
      terms that appear across many documents.
    - Words that are rare overall but frequent in a specific article
      (often the words that actually carry meaning) get boosted.
    - It's fast, memory-efficient compared to word embeddings, and
      works well with linear models like Logistic Regression / PA
      classifier, which is most of what we're using here.
    - No need for a GPU or pretrained embeddings, so it's a practical
      choice for a project that has to train quickly on a laptop.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )


def get_candidate_models():
    """
    Returns the four models we compare. Kept in a dict so training.py
    and any reporting code can loop over them by name.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(
            max_iter=1000, random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": cm,
    }


def train_and_compare(df, text_column: str = "clean_text", label_column: str = "label"):
    """
    Splits the data, fits the vectorizer, trains all candidate models,
    and returns everything needed to pick a winner and persist it.
    """
    X = df[text_column]
    y = df[label_column]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    results = {}
    fitted_models = {}

    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = model

    best_name = max(results, key=lambda name: results[name]["f1_score"])
    best_model = fitted_models[best_name]

    metadata = {
        "best_model": best_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "vocab_size": len(vectorizer.vocabulary_),
        "models_compared": list(results.keys()),
    }

    return {
        "vectorizer": vectorizer,
        "best_model_name": best_name,
        "best_model": best_model,
        "all_results": results,
        "metadata": metadata,
    }
