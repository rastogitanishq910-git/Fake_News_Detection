"""
utils.py
--------
Small shared helpers so training.py / predict.py / app.py aren't
repeating the same path-building and loading logic everywhere.
"""

import os
import json

import joblib
import pandas as pd

# Project root = one level above src/, works no matter where the
# script is invoked from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FAKE_CSV = os.path.join(DATASET_DIR, "Fake.csv")
TRUE_CSV = os.path.join(DATASET_DIR, "True.csv")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_raw_dataset() -> pd.DataFrame:
    """
    Loads Fake.csv + True.csv, tags them, merges, and shuffles.
    Label convention (kept consistent everywhere in the project):
        Fake = 0
        Real = 1
    """
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        raise FileNotFoundError(
            "Could not find dataset/Fake.csv and dataset/True.csv.\n"
            "Download the 'Fake and Real News Dataset' from Kaggle and "
            "place both CSV files inside the dataset/ folder, or run "
            "src/generate_sample_data.py to create a small demo dataset."
        )

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Most versions of this dataset have separate "title" and "text"
    # columns. Combining them gives the model more signal (fake
    # headlines are often clickbait-y in a way the body text isn't).
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "title" in df.columns and "text" not in df.columns:
        df["text"] = df["title"]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_model_artifacts(model, vectorizer, metrics: dict, metadata: dict):
    ensure_dirs()
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    save_json(metrics, METRICS_PATH)
    save_json(metadata, METADATA_PATH)


def load_model_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run 'python train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer
