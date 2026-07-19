"""
predict.py
-----------
Wraps the saved model + vectorizer so app.py can just call
predict_news(text) without worrying about loading artifacts or
repeating the cleaning pipeline.
"""

from src.preprocessing import clean_text
from src.utils import load_model_artifacts

_model = None
_vectorizer = None


def _get_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model, _vectorizer = load_model_artifacts()
    return _model, _vectorizer


def predict_news(raw_text: str) -> dict:
    """
    Takes a raw (uncleaned) article/headline and returns a prediction
    dict with label, confidence, and probability breakdown where the
    model supports it.
    """
    model, vectorizer = _get_artifacts()

    cleaned = clean_text(raw_text)
    if not cleaned.strip():
        raise ValueError(
            "The text became empty after cleaning — try pasting a "
            "longer, more descriptive piece of text."
        )

    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    label = "Real" if prediction == 1 else "Fake"

    result = {
        "label": label,
        "raw_prediction": int(prediction),
        "cleaned_text": cleaned,
    }

    # Not every model exposes predict_proba (PassiveAggressiveClassifier
    # doesn't), so we fall back gracefully.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        result["prob_fake"] = round(float(proba[0]), 4)
        result["prob_real"] = round(float(proba[1]), 4)
        result["confidence"] = round(float(max(proba)), 4)
    elif hasattr(model, "decision_function"):
        # For PassiveAggressiveClassifier we use the distance from the
        # decision boundary as a rough confidence proxy, squashed into
        # a 0-1 range with a sigmoid.
        import math
        score = model.decision_function(vector)[0]
        confidence = 1 / (1 + math.exp(-abs(score)))
        result["confidence"] = round(confidence, 4)
        result["prob_fake"] = None
        result["prob_real"] = None
    else:
        result["confidence"] = None
        result["prob_fake"] = None
        result["prob_real"] = None

    return result
