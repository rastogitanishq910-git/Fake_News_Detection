"""
preprocessing.py
-----------------
Text cleaning utilities for the Fake News Detection project.

Everything the model sees goes through clean_text() first, so this file
is basically the "gatekeeper" for data quality. Keeping it separate from
training.py makes it easy to reuse the exact same cleaning logic at
prediction time (predict.py imports from here too).
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# NLTK needs its stopword/punkt data downloaded once. We try silently and
# only hit the network if the data isn't already cached locally.
def _ensure_nltk_data():
    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                # If we're offline this will fail silently and the
                # fallback tokenizer/stopword list below will kick in.
                pass


_ensure_nltk_data()
stemmer = PorterStemmer()

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    # Small fallback list so the pipeline never hard-crashes if the
    # NLTK corpus couldn't be downloaded (e.g. no internet access).
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
        "in", "on", "at", "to", "of", "for", "with", "as", "by", "that",
        "this", "it", "from", "be", "has", "have", "had", "not", "will",
    }

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NUMBER_PATTERN = re.compile(r"\d+")
EXTRA_SPACE_PATTERN = re.compile(r"\s+")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_html_tags(text: str) -> str:
    return HTML_PATTERN.sub(" ", text)


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    return NUMBER_PATTERN.sub(" ", text)


def remove_special_characters(text: str) -> str:
    # After punctuation/number removal this mostly mops up leftovers
    # like curly quotes, em-dashes, emojis, etc.
    return NON_ALPHA_PATTERN.sub(" ", text)


def remove_extra_spaces(text: str) -> str:
    return EXTRA_SPACE_PATTERN.sub(" ", text).strip()


def remove_stopwords(text: str) -> str:
    words = text.split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered)


def stem_text(text: str) -> str:
    words = text.split()
    stemmed = [stemmer.stem(w) for w in words]
    return " ".join(stemmed)


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline applied to a single piece of text.
    Order matters here — URLs/HTML need to go before we strip
    punctuation, otherwise "https://" and "<br/>" turn into noise
    that's harder to catch.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_spaces(text)
    text = remove_stopwords(text)
    text = stem_text(text)

    return text


def clean_dataframe(df, text_column: str = "text", new_column: str = "clean_text"):
    """
    Applies clean_text() to an entire dataframe column.
    Drops rows where the source text is null/empty before processing
    so we don't waste time cleaning nothing.
    """
    df = df.dropna(subset=[text_column]).copy()
    df = df[df[text_column].str.strip() != ""]
    df[new_column] = df[text_column].apply(clean_text)
    # A handful of articles become empty strings after cleaning
    # (e.g. text that was 100% numbers/links) — drop those too.
    df = df[df[new_column].str.strip() != ""]
    return df.reset_index(drop=True)
