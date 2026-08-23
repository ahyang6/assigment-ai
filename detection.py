"""Core detection logic for Message Guard: text preprocessing, the risk-
scoring heuristics, the ML training pipeline (all 4 candidate algorithms),
and the SpamDetector class used to classify a message with a chosen
algorithm. Kept separate from app.py (UI/routing) and gmail_integration.py
(Gmail OAuth + scanning) so it can be imported by both without either of
them needing to import the other - gmail_integration.py needs `detector()`
from here, and app.py needs the `dashboard` page from gmail_integration.py;
putting the shared ML logic in its own module avoids that import cycle."""
import csv
import json
import logging
import re
import string
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "spam.csv"
MODEL_DIR = BASE_DIR / "model"
MODELS_PATH = MODEL_DIR / "spam_models.pkl"
CATEGORY_MODELS_PATH = MODEL_DIR / "category_models.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"
RANDOM_STATE = 42
LABELS = ["low", "medium", "high"]


KEYWORDS = {
    "urgency": ["urgent", "immediately", "now", "act fast", "limited", "hurry", "today"],
    "promotional": ["free", "prize", "winner", "won", "offer", "cash", "gift"],
    "security": ["verify", "password", "account suspended", "login", "confirm", "security alert"],
    "action": ["click", "claim", "reply", "call", "subscribe", "open link"],
}
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s()\-]{6,}\d)(?!\w)")
SUSPICIOUS_URL_PATTERN = re.compile(
    r"(?:\d{1,3}\.){3}\d{1,3}|bit\.ly|tinyurl|login|verify|secure|account|password|update",
    re.IGNORECASE,
)


def find_indicators(message: str) -> dict[str, Any]:
    """Locate linguistic and structural warning signals in a message."""
    lowered = message.lower()
    found = {category: [term for term in terms if term in lowered] for category, terms in KEYWORDS.items()}
    found = {key: value for key, value in found.items() if value}
    urls = URL_PATTERN.findall(message)
    return {
        "keywords": found,
        "urls": urls,
        "suspicious_urls": [url for url in urls if SUSPICIOUS_URL_PATTERN.search(url)],
        "emails": EMAIL_PATTERN.findall(message),
        "phones": PHONE_PATTERN.findall(message),
        "caps": len(re.findall(r"\b[A-Z]{3,}\b", message)),
        "repeated_punctuation": bool(re.search(r"[!?]{2,}", message)),
    }


def calculate_risk(probabilities: dict[str, float], indicators: dict[str, Any]) -> tuple[int, str]:
    """Calculate transparent 0–100 risk score from model and observed indicators."""
    # "medium" contributes at half the weight of "high" - treating them
    # equally meant a message the model confidently called "medium" alone
    # produced a base score near 100 (since medium+high probabilities
    # summed to ~1), which always landed in the "High Risk" bucket and then
    # got escalated over the model's own "medium" prediction. Extra points
    # from genuinely risky signals below can still legitimately push a
    # medium-leaning message into High Risk - the probability alone just
    # no longer forces that outcome by itself.
    base = 100 * (0.5 * probabilities.get("medium", 0) + probabilities.get("high", 0))
    keyword_count = sum(len(items) for items in indicators["keywords"].values())
    score = base + min(keyword_count * 4, 16) + (12 if indicators["urls"] else 0)
    score += min(len(indicators.get("suspicious_urls", [])) * 6, 18)
    score += 5 if indicators["emails"] else 0
    score += 5 if indicators["phones"] else 0
    score += min(indicators["caps"] * 2, 8) + (5 if indicators["repeated_punctuation"] else 0)
    score = min(100, round(score))
    level = "Low Risk" if score <= 30 else "Medium Risk" if score <= 70 else "High Risk"
    return score, level


SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}


def reconcile_severity(prediction: str, risk_level: str) -> str:
    """The model's raw class prediction and the separately-computed 0-100
    risk score can genuinely disagree - the score also factors in keyword,
    URL, and formatting signals that the classifier's probabilities alone
    don't fully capture. Rather than show a headline that undersells what
    the risk score underneath it says (e.g. "MEDIUM" next to "High Risk"),
    always surface the more severe of the two."""
    risk_key = risk_level.split()[0].lower()  # "High Risk" -> "high"
    return prediction if SEVERITY_RANK[prediction] >= SEVERITY_RANK.get(risk_key, 0) else risk_key


def explanation(prediction: str, indicators: dict[str, Any]) -> str:
    """Turn detected signals into a concise human-readable reason."""
    parts = []
    words = [word.upper() for group in indicators["keywords"].values() for word in group]
    if words:
        parts.append("suspicious language: " + ", ".join(words))
    if indicators.get("suspicious_urls"):
        parts.append("a suspicious URL")
    elif indicators["urls"]:
        parts.append("a URL")
    if indicators["repeated_punctuation"]:
        parts.append("repeated punctuation")
    if prediction == "low" and not parts:
        return "No strong spam or phishing signals were detected."
    return "The message was flagged because it contains " + (", ".join(parts) or "patterns associated with unsafe messages") + "."


def categorize_message(prediction: str, indicators: dict[str, Any]) -> str:
    """Derive a human-readable message category (Phishing / Spam / Scam / etc.)
    from the risk level and the same rule-based indicators used elsewhere,
    without requiring a separately trained category model."""
    if prediction == "low":
        return "Legitimate Message"

    groups = set(indicators["keywords"].keys())
    has_suspicious_url = bool(indicators.get("suspicious_urls"))
    has_url = bool(indicators["urls"])

    if "security" in groups and (has_suspicious_url or has_url):
        return "Phishing Attempt"
    if "promotional" in groups:
        return "Spam / Promotional"
    # Require at least 2 total urgency+action keyword hits, not just one of
    # each - a single common word like "now" or "call" shows up constantly
    # in ordinary everyday requests ("can we call now?") and isn't on its
    # own meaningful evidence of a scam pattern.
    urgency_action_hits = len(indicators["keywords"].get("urgency", [])) + len(indicators["keywords"].get("action", []))
    if "urgency" in groups and "action" in groups and urgency_action_hits >= 3:
        return "Urgent Action Scam"
    return "Suspicious Message"


HISTORY_COLUMNS = ["Date", "Message", "Prediction", "Confidence", "Risk Score", "Category"]


def _ensure_history_schema() -> None:
    """Self-heal the history file if it was created under an older column
    layout (e.g. before Category existed). Appending new-format rows to an
    old-format file produces a ragged CSV with a different field count per
    row, which pandas' parser can't read at all - so this rewrites the whole
    file to the current schema, padding any missing trailing fields, before
    any other read or write touches it."""
    if not HISTORY_PATH.exists():
        return
    with HISTORY_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return
    header, data_rows = rows[0], rows[1:]
    if header == HISTORY_COLUMNS and all(len(row) == len(HISTORY_COLUMNS) for row in data_rows):
        return  # already consistent, nothing to migrate

    fixed_rows = []
    for row in data_rows:
        row = list(row[:len(HISTORY_COLUMNS)])  # trim any unexpected extra fields
        while len(row) < len(HISTORY_COLUMNS):
            row.append("—" if HISTORY_COLUMNS[len(row)] == "Category" else "")
        fixed_rows.append(row)

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HISTORY_COLUMNS)
        writer.writerows(fixed_rows)


def append_history(message: str, prediction: str, confidence: float, risk: int, category: str) -> None:
    """Persist one analysis result to the local CSV history."""
    _ensure_history_schema()
    new_file = not HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(HISTORY_COLUMNS)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), message, prediction, round(confidence, 4), risk, category])


def load_history() -> pd.DataFrame:
    """Return saved predictions, or an empty table with the expected columns."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    _ensure_history_schema()
    return pd.read_csv(HISTORY_PATH)


def clear_history() -> None:
    """Remove all saved prediction history."""
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def delete_history_rows(row_positions: list[int]) -> None:
    """Remove specific rows (by their position in the saved file) from history."""
    if not HISTORY_PATH.exists():
        return
    _ensure_history_schema()
    history = pd.read_csv(HISTORY_PATH)
    history = history.drop(index=row_positions, errors="ignore").reset_index(drop=True)
    history.to_csv(HISTORY_PATH, index=False)

# =========================================================================
# preprocess.py
# =========================================================================
import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s()\-]{6,}\d)(?!\w)")


@lru_cache(maxsize=1)
def _resources() -> tuple[set[str], WordNetLemmatizer]:
    """Ensure NLTK data exists and return preprocessing resources."""
    for package, location in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(location)
        except LookupError:
            nltk.download(package, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


def _replace_signals(text: str) -> str:
    """Replace URLs, emails, and phone numbers with stable tokens for TF-IDF."""
    text = URL_PATTERN.sub(" urltoken ", text)
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    text = PHONE_PATTERN.sub(" phonetoken ", text)
    return text


def preprocess_text(text: object) -> str:
    """Normalize text, preserve structural tokens, remove noise, then lemmatize."""
    if not isinstance(text, str):
        return ""
    stops, lemmatizer = _resources()
    cleaned = _replace_signals(text.lower())
    cleaned = re.sub(r"\d+", " ", cleaned)
    cleaned = cleaned.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = word_tokenize(cleaned)
    preserved = {"urltoken", "emailtoken", "phonetoken"}
    return " ".join(
        token if token in preserved else lemmatizer.lemmatize(token)
        for token in tokens
        if token in preserved or (token not in stops and len(token) > 1)
    )

# =========================================================================
# evaluation.py
# =========================================================================
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(model: Any, x_test: Any, y_test: Any, labels: list[str]) -> dict[str, Any]:
    """Return standard and per-class classification metrics for a fitted model."""
    predicted = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predicted, labels=labels, average="weighted", zero_division=0
    )
    per_class = classification_report(
        y_test, predicted, labels=labels, output_dict=True, zero_division=0
    )
    per_class_metrics = {
        label: {
            "precision": round(float(per_class[label]["precision"]), 4),
            "recall": round(float(per_class[label]["recall"]), 4),
            "f1": round(float(per_class[label]["f1-score"]), 4),
            "support": int(per_class[label]["support"]),
        }
        for label in labels
        if label in per_class
    }
    return {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "per_class": per_class_metrics,
        "labels": labels,
    }


def cross_validate_scores(model: Any, x_data: Any, y_data: Any, cv: Any) -> dict[str, float]:
    """Return mean accuracy and weighted F1 from stratified cross-validation."""
    from sklearn.model_selection import cross_validate

    scores = cross_validate(
        model,
        x_data,
        y_data,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1": "f1_weighted"},
        n_jobs=1,
    )
    return {
        "cv_accuracy": round(float(np.mean(scores["test_accuracy"])), 4),
        "cv_f1": round(float(np.mean(scores["test_f1"])), 4),
        "cv_f1_std": round(float(np.std(scores["test_f1"])), 4),
    }

# =========================================================================
# predict.py
# =========================================================================
from typing import Any
import joblib


class SpamDetector:
    """Load saved artifacts and produce explainable message classifications
    using a specific trained algorithm (or the best one, by default)."""
    def __init__(self, model_name: str | None = None) -> None:
        if not MODELS_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError("No model artifacts found. Run: python train_model.py")
        all_models: dict[str, Any] = joblib.load(MODELS_PATH)
        if not model_name:
            info = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
            model_name = info.get("best_model") or next(iter(all_models))
        if model_name not in all_models:
            raise ValueError(f"Unknown algorithm: {model_name!r}. Available: {sorted(all_models)}")
        self.model_name = model_name
        self.model = all_models[model_name]
        self.vectorizer = joblib.load(VECTORIZER_PATH)

        # Category prediction is AI-trained (same algorithm, same TF-IDF
        # features, target=category) rather than the old purely rule-based
        # keyword lookup - falls back to that rule-based categorize_message()
        # if no category model was trained (e.g. an older dataset without
        # a "category" column).
        self.category_model = None
        if CATEGORY_MODELS_PATH.exists():
            category_models: dict[str, Any] = joblib.load(CATEGORY_MODELS_PATH)
            self.category_model = category_models.get(model_name)

    def analyze(self, message: str) -> dict[str, Any]:
        """Classify a non-empty message and calculate its risk explanation."""
        if not message or not message.strip():
            raise ValueError("Please enter a message to analyse.")
        processed = preprocess_text(message)
        features = self.vectorizer.transform([processed])
        indicators = find_indicators(message)

        # If almost none of the message's words are in the vectorizer's
        # learned vocabulary (e.g. it's just a couple of rare proper nouns
        # never seen in training), the model has essentially no real
        # signal to work with. Different algorithms handle a near-empty
        # feature vector inconsistently - SVM/Logistic Regression fall back
        # toward "low" via their decision boundary, while Naive Bayes'
        # multiplicative likelihood math can swing toward other classes
        # purely from smoothing artifacts, not genuine evidence. Rather
        # than presenting that as a confident, algorithm-dependent verdict,
        # treat "too little recognizable content" the same way regardless
        # of which algorithm is selected - unless there are still concrete
        # rule-based warning signs (a URL, phone number, etc.) worth flagging.
        has_matching_vocabulary = features.nnz > 0
        has_other_signals = bool(
            indicators["keywords"] or indicators["urls"] or indicators["emails"] or indicators["phones"]
        )
        if not has_matching_vocabulary and not has_other_signals:
            return {
                "prediction": "low", "confidence": 1.0,
                "probabilities": {"low": 1.0, "medium": 0.0, "high": 0.0},
                "risk_score": 0, "risk_level": "Low Risk", "indicators": indicators,
                "algorithm": self.model_name,
                "explanation": "This message is too short or doesn't contain enough recognizable text for a reliable analysis.",
                "category": "Legitimate Message",
                "category_confidence": None,
            }

        probabilities = dict(zip(self.model.classes_, self.model.predict_proba(features)[0]))
        raw_prediction = max(probabilities, key=probabilities.get)
        risk, risk_level = calculate_risk(probabilities, indicators)
        prediction = reconcile_severity(raw_prediction, risk_level)

        if self.category_model is not None:
            category_probs = dict(zip(self.category_model.classes_, self.category_model.predict_proba(features)[0]))
            category = max(category_probs, key=category_probs.get)
            category_confidence = float(category_probs[category])
        else:
            category = categorize_message(prediction, indicators)
            category_confidence = None

        return {"prediction": prediction, "confidence": float(probabilities[prediction]), "probabilities": probabilities,
                "risk_score": risk, "risk_level": risk_level, "indicators": indicators, "algorithm": self.model_name,
                "explanation": explanation(prediction, indicators), "category": category,
                "category_confidence": category_confidence}

# =========================================================================
# train_model.py
# =========================================================================
import json
import logging
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


CATEGORY_LABELS = [
    "Legitimate Message", "Phishing Attempt", "Spam / Promotional",
    "Urgent Action Scam", "Suspicious Message",
]


def load_dataset() -> pd.DataFrame:
    """Load, validate, de-duplicate, and clean the configured dataset."""
    data = pd.read_csv(DATASET_PATH)
    if not {"message", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain 'message' and 'label' columns.")
    keep_cols = ["message", "label"] + (["category"] if "category" in data.columns else [])
    data = data[keep_cols].dropna(subset=["message", "label"]).drop_duplicates(subset=["message"])
    data["label"] = data["label"].str.lower().str.strip()
    data = data[data["label"].isin(LABELS)]
    if "category" in data.columns:
        data = data[data["category"].isin(CATEGORY_LABELS)]
    if data.empty or data["label"].nunique() < 2:
        raise ValueError("Dataset needs at least two valid label classes.")
    return data


# Kept in sync with the keys of the `candidates` dict inside train() below.
# detector() compares this against what's saved in metrics.json to detect
# when a newly added/removed algorithm means the saved model is stale and
# needs retraining - not just when metrics.json is completely missing.
EXPECTED_CANDIDATE_NAMES = sorted([
    "Support Vector Machine", "Logistic Regression", "Multinomial Naive Bayes",
])


def train() -> dict:
    """Train candidates with cross-validation, select highest CV F1, and save artifacts."""
    data = load_dataset()
    data["processed"] = data["message"].map(preprocess_text)
    labels = sorted(data["label"].unique())
    folds = min(5, data["label"].value_counts().min())
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    x_train, x_test, y_train, y_test = train_test_split(
        data["processed"],
        data["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=3000,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    x_all_vec = vectorizer.transform(data["processed"])

    candidates = {
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            ensemble=False,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
    }

    results: dict[str, dict] = {}
    best_name, best_score = "", -1.0
    for name, model in candidates.items():
        cv_metrics = cross_validate_scores(model, x_all_vec, data["label"], cv)

        train_start = time.perf_counter()
        model.fit(x_train_vec, y_train)
        training_time_ms = (time.perf_counter() - train_start) * 1000

        # Time just the predict() call itself, separate from downstream
        # metric computation (precision/recall/confusion matrix etc.), for
        # a clean measure of per-message inference speed.
        predict_start = time.perf_counter()
        model.predict(x_test_vec)
        prediction_time_ms = (time.perf_counter() - predict_start) * 1000
        prediction_time_per_message_ms = prediction_time_ms / max(1, x_test_vec.shape[0])

        holdout = evaluate_model(model, x_test_vec, y_test, labels)
        results[name] = {
            **cv_metrics,
            "accuracy": holdout["accuracy"],
            "precision": holdout["precision"],
            "recall": holdout["recall"],
            "f1": holdout["f1"],
            "training_time_ms": round(training_time_ms, 2),
            "prediction_time_per_message_ms": round(prediction_time_per_message_ms, 4),
            "confusion_matrix": holdout["confusion_matrix"],
            "per_class": holdout["per_class"],
            "labels": labels,
        }
        if cv_metrics["cv_f1"] > best_score:
            best_name, best_score = name, cv_metrics["cv_f1"]

    # Refit every candidate (not just the winner) on the full dataset and
    # save all of them, so the user can pick any algorithm for live
    # detection on the Analyze Message page - not only the best-scoring one.
    fitted_models: dict[str, Any] = {}
    for name, model in candidates.items():
        model.fit(x_all_vec, data["label"])
        fitted_models[name] = model

    # Also train a category classifier (same algorithm types, same TF-IDF
    # features) predicting the message's category - a genuinely AI-learned
    # prediction rather than the old purely rule-based keyword lookup.
    # Category labels come from the dataset's own "category" column when
    # present; datasets that predate this column simply skip it, so
    # SpamDetector falls back to the rule-based categorize_message().
    MODEL_DIR.mkdir(exist_ok=True)
    category_models: dict[str, Any] = {}
    if "category" in data.columns:
        category_candidates = {
            "Support Vector Machine": CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
                ensemble=False,
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "Multinomial Naive Bayes": MultinomialNB(),
        }
        for name, model in category_candidates.items():
            model.fit(x_all_vec, data["category"])
            category_models[name] = model
        joblib.dump(category_models, CATEGORY_MODELS_PATH)

    joblib.dump(fitted_models, MODELS_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    payload = {
        "best_model": best_name,
        "dataset_rows": len(data),
        "cv_folds": folds,
        "class_distribution": data["label"].value_counts().to_dict(),
        "models": results,
        "holdout": results[best_name],
        "candidate_names": sorted(candidates.keys()),
        "dataset_fingerprint": _dataset_fingerprint(),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info(
        "Saved %d algorithms, best=%s (CV weighted F1: %.4f, holdout F1: %.4f)",
        len(fitted_models),
        best_name,
        best_score,
        results[best_name]["f1"],
    )
    return payload


def _dataset_fingerprint() -> str:
    """Content hash of the dataset file, used to detect when spam.csv has
    changed (rows added/edited/removed) so the cached model can be
    automatically retrained - comparing only the algorithm name list
    wasn't enough, since editing the dataset without changing which
    algorithms are used would otherwise keep silently serving the old
    model trained on the old data."""
    import hashlib
    if not DATASET_PATH.exists():
        return ""
    return hashlib.md5(DATASET_PATH.read_bytes()).hexdigest()


@st.cache_resource
def detector(model_name: str | None = None) -> SpamDetector:
    """Load an existing model (or train from scratch on first deployment,
    or retrain if the saved model's algorithm set or the underlying
    dataset is stale - e.g. a new candidate algorithm was added, or
    spam.csv was edited)."""
    needs_training = (
        not METRICS_PATH.exists() or not MODELS_PATH.exists() or not VECTORIZER_PATH.exists()
        or ("category" in load_dataset().columns and not CATEGORY_MODELS_PATH.exists())
    )
    if not needs_training:
        try:
            saved_info = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if sorted(saved_info.get("candidate_names", [])) != EXPECTED_CANDIDATE_NAMES:
                needs_training = True
            elif saved_info.get("dataset_fingerprint") != _dataset_fingerprint():
                needs_training = True
        except Exception:
            needs_training = True
    if needs_training:
        with st.spinner("Preparing the AI model for first use…"):
            train()
        metrics.clear()  # invalidate cached metrics() so the new candidate_names show up immediately
    return SpamDetector(model_name)

def ensure_trained() -> None:
    """Make sure a trained model (all candidate algorithms) exists before
    any page needs metrics() or a SpamDetector - so a first-time visitor
    doesn't have to run an analysis first just to see the algorithm
    dropdown or the comparison table populated."""
    detector()  # st.cache_resource-cached: only actually trains once


@st.cache_data
def metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}