"""
train_model.py
----------------
Entry point for training the fake news detector.

Usage:
    python train_model.py

What it does:
    1. Loads dataset/Fake.csv + dataset/True.csv
    2. Cleans the text (src/preprocessing.py)
    3. Fits a TF-IDF vectorizer
    4. Trains 4 models and compares them
    5. Saves the best model + vectorizer + metrics into models/

Run this before starting the Streamlit app — app.py expects the files
in models/ to already exist.
"""

import sys
import time

from src.utils import load_raw_dataset, save_model_artifacts, ensure_dirs
from src.preprocessing import clean_dataframe
from src.training import train_and_compare


def main():
    print("=" * 60)
    print("FAKE NEWS DETECTION — MODEL TRAINING")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/4] Loading dataset...")
    try:
        df = load_raw_dataset()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print(f"      Loaded {len(df)} articles.")

    print("\n[2/4] Cleaning text (this can take a minute on the full dataset)...")
    start = time.time()
    df = clean_dataframe(df, text_column="text", new_column="clean_text")
    print(f"      Done in {time.time() - start:.1f}s. {len(df)} usable rows remain.")

    print("\n[3/4] Training and comparing models...")
    start = time.time()
    result = train_and_compare(df)
    print(f"      Done in {time.time() - start:.1f}s.")

    print("\n      Model comparison (sorted by F1 score):")
    sorted_results = sorted(
        result["all_results"].items(), key=lambda x: x[1]["f1_score"], reverse=True
    )
    for name, metrics in sorted_results:
        print(
            f"        {name:<32} "
            f"Acc: {metrics['accuracy']:.3f}  "
            f"Prec: {metrics['precision']:.3f}  "
            f"Rec: {metrics['recall']:.3f}  "
            f"F1: {metrics['f1_score']:.3f}"
        )

    print(f"\n      Best model: {result['best_model_name']}")

    print("\n[4/4] Saving model artifacts to models/...")
    save_model_artifacts(
        model=result["best_model"],
        vectorizer=result["vectorizer"],
        metrics=result["all_results"],
        metadata=result["metadata"],
    )
    print("      Saved: models/best_model.pkl, models/tfidf_vectorizer.pkl,")
    print("             models/metrics.json, models/model_metadata.json")

    print("\n" + "=" * 60)
    print("Training complete. Run 'streamlit run app.py' to launch the app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
