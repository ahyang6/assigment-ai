"""Streamlit interface for the AI-powered spam and phishing detector."""
import base64
import csv
import json
import logging
import re
import string
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import docx
import joblib
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_oauthlib.flow import Flow as GoogleOAuthFlow
from googleapiclient.discovery import build as build_google_service
from googleapiclient.errors import HttpError as GoogleApiError
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

# =========================================================================
# config.py
# =========================================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "spam.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"
GMAIL_TOKEN_PATH = BASE_DIR / "gmail_token.json"
GMAIL_PKCE_PATH = BASE_DIR / "gmail_pkce_verifier.txt"
RANDOM_STATE = 42
LABELS = ["low", "medium", "high"]

# =========================================================================
# helpers.py / utils/helpers.py
# =========================================================================
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd


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
    base = 100 * (probabilities.get("medium", 0) + probabilities.get("high", 0))
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
    if "urgency" in groups and "action" in groups:
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
    """Load saved artifacts and produce explainable message classifications."""
    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError("No model artifacts found. Run: python train_model.py")
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def analyze(self, message: str) -> dict[str, Any]:
        """Classify a non-empty message and calculate its risk explanation."""
        if not message or not message.strip():
            raise ValueError("Please enter a message to analyse.")
        features = self.vectorizer.transform([preprocess_text(message)])
        probabilities = dict(zip(self.model.classes_, self.model.predict_proba(features)[0]))
        raw_prediction = max(probabilities, key=probabilities.get)
        indicators = find_indicators(message)
        risk, risk_level = calculate_risk(probabilities, indicators)
        prediction = reconcile_severity(raw_prediction, risk_level)
        category = categorize_message(prediction, indicators)
        return {"prediction": prediction, "confidence": float(probabilities[prediction]), "probabilities": probabilities,
                "risk_score": risk, "risk_level": risk_level, "indicators": indicators,
                "explanation": explanation(prediction, indicators), "category": category}

# =========================================================================
# train_model.py
# =========================================================================
import json
import logging

import joblib
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class KMeansRiskClassifier(BaseEstimator, ClassifierMixin):
    """Wrap unsupervised KMeans so it fits the same fit/predict/predict_proba
    interface as the other candidates. Clusters are labelled after fitting by
    the majority class of the training rows that land in them; predict_proba
    is derived from a softmax-style normalisation of distances to each
    cluster center."""

    def __init__(self, n_clusters: int = 3, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, x, y):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans_.fit_predict(x)
        y = pd.Series(y).reset_index(drop=True)
        self.classes_ = np.array(sorted(y.unique()))
        self.cluster_to_label_ = {}
        for cluster_id in range(self.n_clusters):
            in_cluster = y[clusters == cluster_id]
            self.cluster_to_label_[cluster_id] = (
                in_cluster.value_counts().idxmax() if not in_cluster.empty else y.value_counts().idxmax()
            )
        return self

    def predict(self, x):
        clusters = self.kmeans_.predict(x)
        return np.array([self.cluster_to_label_[cluster_id] for cluster_id in clusters])

    def predict_proba(self, x):
        distances = self.kmeans_.transform(x)
        similarities = 1 / (1 + distances)
        cluster_probabilities = similarities / similarities.sum(axis=1, keepdims=True)
        class_index = {label: index for index, label in enumerate(self.classes_)}
        class_probabilities = np.zeros((x.shape[0], len(self.classes_)))
        for cluster_id, label in self.cluster_to_label_.items():
            class_probabilities[:, class_index[label]] += cluster_probabilities[:, cluster_id]
        return class_probabilities


class AgglomerativeRiskClassifier(BaseEstimator, ClassifierMixin):
    """Wrap unsupervised AgglomerativeClustering so it fits the same
    fit/predict/predict_proba interface as the other candidates.
    AgglomerativeClustering has no native out-of-sample predict, so cluster
    centroids are computed after fitting and new samples are assigned to the
    nearest one; clusters are labelled the same majority-vote way as
    KMeansRiskClassifier."""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters

    def fit(self, x, y):
        x_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        self.clustering_ = AgglomerativeClustering(n_clusters=self.n_clusters)
        clusters = self.clustering_.fit_predict(x_dense)
        y = pd.Series(y).reset_index(drop=True)
        self.classes_ = np.array(sorted(y.unique()))
        self.cluster_centers_ = np.array([
            x_dense[clusters == cluster_id].mean(axis=0)
            for cluster_id in range(self.n_clusters)
        ])
        self.cluster_to_label_ = {}
        for cluster_id in range(self.n_clusters):
            in_cluster = y[clusters == cluster_id]
            self.cluster_to_label_[cluster_id] = (
                in_cluster.value_counts().idxmax() if not in_cluster.empty else y.value_counts().idxmax()
            )
        return self

    def _cluster_distances(self, x):
        x_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        return cdist(x_dense, self.cluster_centers_)

    def predict(self, x):
        nearest = self._cluster_distances(x).argmin(axis=1)
        return np.array([self.cluster_to_label_[cluster_id] for cluster_id in nearest])

    def predict_proba(self, x):
        distances = self._cluster_distances(x)
        similarities = 1 / (1 + distances)
        cluster_probabilities = similarities / similarities.sum(axis=1, keepdims=True)
        class_index = {label: index for index, label in enumerate(self.classes_)}
        class_probabilities = np.zeros((distances.shape[0], len(self.classes_)))
        for cluster_id, label in self.cluster_to_label_.items():
            class_probabilities[:, class_index[label]] += cluster_probabilities[:, cluster_id]
        return class_probabilities


class BirchRiskClassifier(BaseEstimator, ClassifierMixin):
    """Wrap unsupervised Birch so it fits the same fit/predict/predict_proba
    interface as the other candidates. Birch's own predict() correctly
    assigns new points to one of the final n_clusters, but its transform()
    returns distances to internal CF-tree subclusters rather than the final
    clusters, so predict_proba instead uses our own centroids of the final
    clusters — the same approach used for AgglomerativeRiskClassifier."""

    def __init__(self, n_clusters: int = 3, threshold: float = 0.5):
        self.n_clusters = n_clusters
        self.threshold = threshold

    def fit(self, x, y):
        x_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        self.birch_ = Birch(n_clusters=self.n_clusters, threshold=self.threshold)
        clusters = self.birch_.fit_predict(x_dense)
        y = pd.Series(y).reset_index(drop=True)
        self.classes_ = np.array(sorted(y.unique()))
        self.cluster_centers_ = np.array([
            x_dense[clusters == cluster_id].mean(axis=0)
            for cluster_id in range(self.n_clusters)
        ])
        self.cluster_to_label_ = {}
        for cluster_id in range(self.n_clusters):
            in_cluster = y[clusters == cluster_id]
            self.cluster_to_label_[cluster_id] = (
                in_cluster.value_counts().idxmax() if not in_cluster.empty else y.value_counts().idxmax()
            )
        return self

    def predict(self, x):
        x_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        clusters = self.birch_.predict(x_dense)
        return np.array([self.cluster_to_label_[cluster_id] for cluster_id in clusters])

    def predict_proba(self, x):
        x_dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        distances = cdist(x_dense, self.cluster_centers_)
        similarities = 1 / (1 + distances)
        cluster_probabilities = similarities / similarities.sum(axis=1, keepdims=True)
        class_index = {label: index for index, label in enumerate(self.classes_)}
        class_probabilities = np.zeros((x_dense.shape[0], len(self.classes_)))
        for cluster_id, label in self.cluster_to_label_.items():
            class_probabilities[:, class_index[label]] += cluster_probabilities[:, cluster_id]
        return class_probabilities


def load_dataset() -> pd.DataFrame:
    """Load, validate, de-duplicate, and clean the configured dataset."""
    data = pd.read_csv(DATASET_PATH)
    if not {"message", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain 'message' and 'label' columns.")
    data = data[["message", "label"]].dropna().drop_duplicates()
    data["label"] = data["label"].str.lower().str.strip()
    data = data[data["label"].isin(LABELS)]
    if data.empty or data["label"].nunique() < 2:
        raise ValueError("Dataset needs at least two valid label classes.")
    return data


# Kept in sync with the keys of the `candidates` dict inside train() below.
# detector() compares this against what's saved in metrics.json to detect
# when a newly added/removed algorithm means the saved model is stale and
# needs retraining - not just when metrics.json is completely missing.
EXPECTED_CANDIDATE_NAMES = sorted([
    "Support Vector Machine", "K-Means Clustering",
    "Agglomerative Clustering", "BIRCH Clustering",
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
        "K-Means Clustering": KMeansRiskClassifier(n_clusters=len(LABELS), random_state=RANDOM_STATE),
        "Agglomerative Clustering": AgglomerativeRiskClassifier(n_clusters=len(LABELS)),
        "BIRCH Clustering": BirchRiskClassifier(n_clusters=len(LABELS)),
    }

    results: dict[str, dict] = {}
    best_name, best_model, best_score = "", None, -1.0
    for name, model in candidates.items():
        cv_metrics = cross_validate_scores(model, x_all_vec, data["label"], cv)
        model.fit(x_train_vec, y_train)
        holdout = evaluate_model(model, x_test_vec, y_test, labels)
        results[name] = {
            **cv_metrics,
            "accuracy": holdout["accuracy"],
            "precision": holdout["precision"],
            "recall": holdout["recall"],
            "f1": holdout["f1"],
            "confusion_matrix": holdout["confusion_matrix"],
            "per_class": holdout["per_class"],
            "labels": labels,
        }
        if cv_metrics["cv_f1"] > best_score:
            best_name, best_model, best_score = name, model, cv_metrics["cv_f1"]

    best_model.fit(x_all_vec, data["label"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    payload = {
        "best_model": best_name,
        "dataset_rows": len(data),
        "cv_folds": folds,
        "class_distribution": data["label"].value_counts().to_dict(),
        "models": results,
        "holdout": results[best_name],
        "candidate_names": sorted(candidates.keys()),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.info(
        "Saved %s (CV weighted F1: %.4f, holdout F1: %.4f)",
        best_name,
        best_score,
        results[best_name]["f1"],
    )
    return payload


# =========================================================================
# app.py
# =========================================================================
import json
from io import BytesIO
import docx
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from design import (
    CATEGORY_ICONS, CATEGORY_COLORS, GLOBAL_CSS, RISK_COLORS, RISK_ICONS,
    apply_theme, page_header, render_hero, stat_card, style_fig,
)
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_oauthlib.flow import Flow as GoogleOAuthFlow
from googleapiclient.discovery import build as build_google_service
from googleapiclient.errors import HttpError as GoogleApiError
from pypdf import PdfReader


if "started" not in st.session_state:
    st.session_state.started = False
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

# The "Get Started" button lives inside the hero's HTML component (so it's
# visually one piece with the background) rather than as a normal Streamlit
# button. A component iframe can't call Streamlit callbacks directly, so the
# button is a plain link that navigates the *parent* page to `?start=1`;
# we catch that here and translate it into normal session-state navigation.
if st.query_params.get("start") == "1":
    st.session_state.started = True
    st.session_state.nav_page = "Analyze Message"
    st.query_params.clear()
elif st.query_params.get("code"):
    # Returning from the Google OAuth redirect (Dashboard's Gmail connect
    # flow). This is a fresh page load, so without this check the user
    # would land back on the Home hero instead - and clicking "Get Started"
    # from there would wipe the ?code= param (it navigates to a bare
    # "?start=1") before Dashboard ever gets a chance to exchange it for a
    # token. Skip straight to Dashboard and leave the param in place; its
    # own OAuth handling reads and clears it after a successful exchange.
    st.session_state.started = True
    st.session_state.nav_page = "Dashboard"

st.set_page_config(
    page_title="Message Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Home", "Analyze Message", "Dashboard", "History", "File Translation"]


@st.cache_resource
def detector() -> SpamDetector:
    """Load an existing model, or train one automatically on first deployment."""
    if not METRICS_PATH.exists():
        with st.spinner("Preparing the AI model for first use…"):
            train()
    return SpamDetector()


@st.cache_data
def metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}


def result_pdf(message: str, result: dict) -> bytes:
    """Make a small downloadable report for one result."""
    buffer = BytesIO(); pdf = canvas.Canvas(buffer, pagesize=letter)
    text = pdf.beginText(48, 740); text.setFont("Helvetica", 11)
    category = result.get("category") or categorize_message(result["prediction"], result["indicators"])
    lines = ["Message Guard - Analysis Report", "", f"Prediction: {result['prediction'].title()}",
             f"Category: {category}",
             f"Confidence: {result['confidence']:.1%}", f"Risk: {result['risk_score']}/100 ({result['risk_level']})", "",
             "Explanation:", result['explanation'], "",
             "Probability Distribution:"]
    lines += [f"  {cls.title()}: {prob:.1%}" for cls, prob in result["probabilities"].items()]
    words = [w.upper() for values in result["indicators"]["keywords"].values() for w in values]
    lines += ["", "Detected Keywords: " + (", ".join(words) if words else "None")]
    if result["indicators"]["urls"]:
        lines.append("Suspicious URLs: " + ", ".join(result["indicators"]["urls"]))
    lines += ["", "Message:"]
    for line in lines + [message[i:i+90] for i in range(0, len(message), 90)]:
        text.textLine(line)
    pdf.drawText(text); pdf.save(); return buffer.getvalue()


TEXT_DECODABLE_EXTENSIONS = {".txt", ".csv", ".json", ".html", ".htm", ".md", ".log", ".xml", ".rtf"}
SUPPORTED_UPLOAD_FORMATS_MESSAGE = "Supported formats: .txt, .csv, .json, .html, .md, .log, .xml, .eml, .pdf, .docx"


def extract_text_from_upload(filename: str, raw_bytes: bytes) -> str:
    """Extract plain text from an uploaded file of (almost) any common
    format, for use either as an analyzable message or as input to the
    File Translation converter. Raises ValueError with a clear message for
    formats that don't contain extractable text (images, old .doc, other
    unrecognised binaries, scanned/image-only PDFs, etc.)."""
    suffix = Path(filename).suffix.lower()

    if suffix in TEXT_DECODABLE_EXTENSIONS:
        return raw_bytes.decode("utf-8", errors="ignore")

    if suffix == ".eml":
        email_message = BytesParser(policy=policy.default).parse(BytesIO(raw_bytes))
        if email_message.is_multipart():
            body = email_message.get_body(preferencelist=("plain",))
            return body.get_content() if body else ""
        return email_message.get_content()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(raw_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("This PDF has no extractable text (it may be a scanned/image-only PDF).")
        return text

    if suffix == ".docx":
        document = docx.Document(BytesIO(raw_bytes))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        if not text.strip():
            raise ValueError("This Word document appears to be empty.")
        return text

    if suffix == ".doc":
        raise ValueError("Old-format .doc files aren't supported - only modern .docx.")

    raise ValueError(f"Unsupported file format: {suffix or '(no extension)'}. {SUPPORTED_UPLOAD_FORMATS_MESSAGE}")


def build_eml_bytes(subject: str, body_text: str) -> bytes:
    """Wrap plain text into a minimal, valid .eml file (RFC822 email)."""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = "converted@message-guard.local"
    message["To"] = "recipient@message-guard.local"
    message.set_content(body_text)
    return message.as_bytes()


@st.dialog("Unsupported File Format")
def unsupported_format_dialog(filename: str) -> None:
    """Shown when Analyze Message receives a file that isn't .txt/.eml."""
    suffix = Path(filename).suffix or "(no extension)"
    st.error(f"**{filename}** — the file type `{suffix}` isn't supported here.")
    st.write("Analyze Message only accepts **.txt** and **.eml** files directly.")
    st.write("Use **File Translation** in the sidebar to convert this file to .txt or .eml first, then upload the converted file here.")
    if st.button("Got it", type="primary", use_container_width=True):
        st.rerun()


def go_to(page_name: str) -> None:
    """Central helper: change page + rerun (avoids duplicated rerun logic)."""
    st.session_state.nav_page = page_name
    st.rerun()


def home() -> None:
    # Full-bleed home page: strip the block-container's padding/max-width
    # just for this render so the hero can fill the entire browser viewport
    # edge-to-edge instead of sitting inside a small centered card. The
    # sidebar itself is always "expanded" per set_page_config (Streamlit's
    # initial_sidebar_state only applies once at true first load and can't
    # be reliably re-toggled dynamically across reruns), so it's hidden here
    # via plain CSS instead — fully within our own control.
    st.html(
        """
        <style>
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            max-width: 100% !important;
        }
        section[data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """
    )

    render_hero("EMAIL DETECTION", dark=True)


def analyze() -> None:
    page_header(
        "🔍", "Analyze Message",
        "root@messageguard:~$ paste a message or email to classify its risk",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )

    sample = "URGENT! Verify your account now at https://secure-check.example or it will be suspended!!"

    uploaded_message = ""
    with st.expander("Or drag and drop a file to analyze (.txt / .eml — other formats can be converted first via File Translation)"):
        uploaded_file = st.file_uploader("Upload file", type=None, label_visibility="collapsed")

        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix not in (".txt", ".eml"):
                if st.session_state.get("_last_invalid_upload") != (uploaded_file.name, uploaded_file.size):
                    st.session_state._last_invalid_upload = (uploaded_file.name, uploaded_file.size)
                    unsupported_format_dialog(uploaded_file.name)
            else:
                try:
                    uploaded_message = extract_text_from_upload(uploaded_file.name, uploaded_file.read())
                except Exception as e:
                    st.error(f"Unable to read file: {e}")
                    return

    if "message_input" not in st.session_state:
        st.session_state.message_input = ""
    if uploaded_message and st.session_state.get("_last_upload_id") != (uploaded_file.name, uploaded_file.size):
        st.session_state.message_input = uploaded_message
        st.session_state._last_upload_id = (uploaded_file.name, uploaded_file.size)

    # Text area
    message = st.text_area(
        "Paste a text message or email",
        key="message_input",
        height=120,
        placeholder=sample
    )

    if st.button("Analyze Message", type="primary", use_container_width=True):
        try:
            with st.spinner("Checking language patterns and risk indicators..."):
                result = detector().analyze(message)

            category = result.get("category") or categorize_message(result["prediction"], result["indicators"])
            append_history(
                message,
                result["prediction"],
                result["confidence"],
                result["risk_score"],
                category
            )

            st.session_state.result = result
            st.session_state.message = message

        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            return

    result = st.session_state.get("result")

    if not result:
        return

    risk_color = RISK_COLORS[result["prediction"]]
    icon = RISK_ICONS[result["prediction"]]

    row1 = st.columns([1.4, 1])

    # --- panel 1: verdict + gauge -------------------------------------------------
    category = result.get("category") or categorize_message(result["prediction"], result["indicators"])
    category_icon = CATEGORY_ICONS.get(category, "⚠️")
    with row1[0]:
        st.html(
            f"""
            <div class="mg-terminal-card cf-fade" style="border-color:{risk_color}55; box-shadow:0 0 26px {risk_color}22; height:230px;">
                <div class="mg-terminal-card-bar">
                    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
                    <span class="label">verdict.log</span>
                </div>
                <div class="mg-terminal-card-body" style="display:flex; align-items:center; gap:1rem;">
                    <div class="mg-gauge" style="--gauge-pct:{result['risk_score']}; --gauge-color:{risk_color}; --gauge-glow:{risk_color}33;">
                        <div class="mg-gauge-inner">
                            <div class="mg-gauge-value">{result['risk_score']}</div>
                            <div class="mg-gauge-label">/ 100</div>
                        </div>
                    </div>
                    <div>
                        <div style="color:{risk_color}; font-size:1.35rem; font-weight:800; letter-spacing:0.02em;">
                            {icon}&nbsp;{result['prediction'].upper()}
                        </div>
                        <div style="color:var(--mg-text-dim); font-size:0.78rem; margin-top:0.3rem;">
                            RISK: {result['risk_level']}<br>CONF: {result['confidence']:.1%}
                        </div>
                        <div style="margin-top:0.5rem;">
                            <span class="mg-badge">{category_icon}&nbsp;{category}</span>
                        </div>
                    </div>
                </div>
            </div>
            """
        )

    # --- panel 2: link to the full breakdown (probability + indicators) -----------
    with row1[1]:
        st.html(
            """
            <div class="mg-terminal-card cf-fade" style="height:230px; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; gap:0.6rem;">
                <div style="color:var(--mg-text-dim); font-size:0.85rem;">
                    Full probability breakdown and<br>detected indicators are on a<br>dedicated details page.
                </div>
            </div>
            """
        )
        if st.button("🔗 View Full Analysis Details", use_container_width=True, key="view_details_btn"):
            go_to("Analysis Details")

    row2 = st.columns([2, 1])

    # --- panel 4: explanation ------------------------------------------------------
    with row2[0]:
        st.html(
            f"""
            <div class="mg-terminal-card cf-fade" style="height:120px;">
                <div class="mg-terminal-card-bar">
                    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
                    <span class="label">explanation.log</span>
                </div>
                <div class="mg-terminal-card-body" style="color:var(--mg-text); font-size:0.88rem;">
                    {result['explanation']}
                </div>
            </div>
            """
        )

    # --- panel 5: export -------------------------------------------------------------
    with row2[1]:
        st.download_button(
            "Download Result as PDF",
            result_pdf(st.session_state.message, result),
            "message-analysis.pdf",
            "application/pdf",
            use_container_width=True,
        )


def analysis_details() -> None:
    page_header(
        "📄", "Analysis Details",
        "root@messageguard:~$ cat probability_distribution.log indicators.log",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )

    if st.button("← Back to Analyze Message"):
        go_to("Analyze Message")

    result = st.session_state.get("result")
    if not result:
        st.info("Run an analysis first from the Analyze Message page.")
        return

    left, right = st.columns(2)

    # --- probability distribution donut --------------------------------------------
    with left:
        chart = pd.DataFrame({
            "Class": list(result["probabilities"]),
            "Probability": list(result["probabilities"].values())
        })
        fig = px.pie(chart, names="Class", values="Probability", hole=0.55, color="Class", color_discrete_map=RISK_COLORS)
        fig = style_fig(fig)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15))
        fig.update_traces(textinfo="percent")
        st.html('<div class="mg-panel-title">Probability Distribution</div>')
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # --- detected indicators --------------------------------------------------------
    with right:
        words = [
            word.upper()
            for values in result["indicators"]["keywords"].values()
            for word in values
        ]
        badges = "".join(f'<span class="mg-badge">{w}</span>' for w in words) or '<span class="mg-panel-title">None detected</span>'
        url_badges = "".join(f'<span class="mg-badge danger">{u}</span>' for u in result["indicators"]["urls"])
        st.html(
            f"""
            <div class="mg-terminal-card cf-fade">
                <div class="mg-terminal-card-bar">
                    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
                    <span class="label">indicators.log</span>
                </div>
                <div class="mg-terminal-card-body">
                    <div class="mg-panel-title">Keywords</div>
                    <div>{badges}</div>
                    {"<div class='mg-panel-title' style='margin-top:0.7rem;'>Suspicious URLs</div><div>" + url_badges + "</div>" if url_badges else ""}
                </div>
            </div>
            """
        )

    st.html("<div style='margin-top:1rem;'></div>")
    st.download_button(
        "Download Result as PDF",
        result_pdf(st.session_state.message, result),
        "message-analysis.pdf",
        "application/pdf",
        use_container_width=True,
    )


GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_SETUP_HELP = (
    "Add `google_client_id`, `google_client_secret`, and `google_redirect_uri` "
    "to this app's Streamlit secrets to enable Gmail scanning."
)


def gmail_oauth_configured() -> bool:
    """Whether the required Google OAuth secrets have been set up."""
    return all(key in st.secrets for key in ("google_client_id", "google_client_secret", "google_redirect_uri"))


def build_gmail_oauth_flow() -> GoogleOAuthFlow:
    """Build the OAuth flow using credentials from Streamlit secrets."""
    client_config = {
        "web": {
            "client_id": st.secrets["google_client_id"],
            "client_secret": st.secrets["google_client_secret"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [st.secrets["google_redirect_uri"]],
        }
    }
    flow = GoogleOAuthFlow.from_client_config(client_config, scopes=GMAIL_SCOPES)
    flow.redirect_uri = st.secrets["google_redirect_uri"]
    return flow


def save_gmail_credentials(credentials: GoogleCredentials) -> None:
    """Persist credentials to a shared file (not just session_state), so a
    connection completed in one browser tab is picked up by other tabs too
    - each tab is its own independent Streamlit session and can't see
    another tab's session_state directly."""
    GMAIL_TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")


def load_gmail_credentials() -> GoogleCredentials | None:
    """Load previously-saved credentials from the shared file, if any."""
    if not GMAIL_TOKEN_PATH.exists():
        return None
    try:
        return GoogleCredentials.from_authorized_user_info(
            json.loads(GMAIL_TOKEN_PATH.read_text(encoding="utf-8")), scopes=GMAIL_SCOPES
        )
    except Exception:
        return None


def clear_gmail_credentials() -> None:
    """Remove the saved credentials file (used on disconnect)."""
    if GMAIL_TOKEN_PATH.exists():
        GMAIL_TOKEN_PATH.unlink()


def _extract_gmail_body(payload: dict) -> str:
    """Recursively find and decode the plain-text (or HTML, as fallback)
    body from a Gmail API message payload."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")

    html_fallback = ""
    for part in payload.get("parts", []) or []:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
        if part.get("mimeType") == "text/html" and part.get("body", {}).get("data") and not html_fallback:
            html_fallback = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
        if part.get("parts"):
            nested = _extract_gmail_body(part)
            if nested:
                return nested

    if html_fallback:
        return re.sub(r"<[^>]+>", " ", html_fallback)  # strip HTML tags as a simple fallback
    return ""


def fetch_gmail_messages(credentials, max_results, progress_callback=None) -> list[str]:
    """Fetch and decode the body text of messages in the user's inbox
    (paginated through the full inbox), up to max_results if given
    (None/0 = no limit, scan everything)."""
    service = build_google_service("gmail", "v1", credentials=credentials)
    message_refs = []
    page_token = None
    while True:
        response = service.users().messages().list(
            userId="me", labelIds=["INBOX"], maxResults=500, pageToken=page_token
        ).execute()
        message_refs.extend(response.get("messages", []))
        page_token = response.get("nextPageToken")
        if not page_token or (max_results and len(message_refs) >= max_results):
            break
    if max_results:
        message_refs = message_refs[:max_results]

    bodies = []
    total = len(message_refs)
    for i, msg_ref in enumerate(message_refs):
        full_message = service.users().messages().get(userId="me", id=msg_ref["id"], format="full").execute()
        body = _extract_gmail_body(full_message.get("payload", {}))
        if body.strip():
            bodies.append(body)
        if progress_callback:
            progress_callback(i + 1, total)
    return bodies


def dashboard() -> None:
    page_header(
        "📊", "Statistics Dashboard", "root@messageguard:~$ connect a gmail inbox to scan its risk",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )

    if not gmail_oauth_configured():
        st.warning(f"Gmail integration isn't configured yet. {GMAIL_SETUP_HELP}")
        return

    if "gmail_credentials" not in st.session_state:
        loaded = load_gmail_credentials()
        if loaded:
            st.session_state.gmail_credentials = loaded

    if "gmail_credentials" not in st.session_state:
        auth_code = st.query_params.get("code")
        if auth_code:
            try:
                flow = build_gmail_oauth_flow()
                # Navigating to Google and back is a fresh page load even
                # within the same tab, so the PKCE code_verifier generated
                # when the auth URL was built has to be recovered from the
                # shared file rather than regenerated here.
                if GMAIL_PKCE_PATH.exists():
                    flow.code_verifier = GMAIL_PKCE_PATH.read_text(encoding="utf-8").strip()
                    GMAIL_PKCE_PATH.unlink()  # one-time use
                flow.fetch_token(code=auth_code)
                st.session_state.gmail_credentials = flow.credentials
                save_gmail_credentials(flow.credentials)
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Gmail authorization failed: {e}")
        else:
            st.write("Connect your Gmail account to scan your inbox and see how many emails are Low, Medium, or High risk.")
            flow = build_gmail_oauth_flow()
            flow.autogenerate_code_verifier = True
            auth_url, _ = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
            GMAIL_PKCE_PATH.write_text(flow.code_verifier, encoding="utf-8")
            # A plain <a> with target="_top" navigates *this* tab (not a
            # new one) - st.link_button always opens a new tab with no way
            # to turn that off, which is why it isn't used here.
            st.html(
                f"""
                <a href="{auth_url}" target="_top" style="
                    display:block; text-align:center; text-decoration:none;
                    background: linear-gradient(135deg, #f6821f, #ff9d3d);
                    border: 1px solid rgba(255, 157, 61, 0.6);
                    border-radius: 2px; color: #ffffff; font-weight: 700;
                    font-size: 1.0rem; letter-spacing: 0.03em;
                    padding: 0.75rem 1.6rem; box-shadow: 0 0 16px rgba(246, 130, 31, 0.35);
                ">🔗 Connect Gmail Account</a>
                """
            )
        return

    credentials = st.session_state.gmail_credentials
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(GoogleAuthRequest())
        st.session_state.gmail_credentials = credentials
        save_gmail_credentials(credentials)

    status_col, disconnect_col = st.columns([3, 1])
    with status_col:
        st.success("✅ Gmail account connected.")
    with disconnect_col:
        if st.button("Disconnect Gmail", use_container_width=True):
            del st.session_state.gmail_credentials
            st.session_state.pop("gmail_scan_results", None)
            clear_gmail_credentials()
            st.rerun()

    limit_input = st.number_input(
        "Max emails to scan (0 = entire inbox)", min_value=0, value=0, step=50,
    )

    if st.button("🔍 Scan Inbox", type="primary", use_container_width=True):
        try:
            progress_bar = st.progress(0.0, text="Fetching inbox...")

            def update_progress(done, total):
                progress_bar.progress(done / total if total else 0.0, text=f"Analysing email {done}/{total}...")

            bodies = fetch_gmail_messages(credentials, max_results=limit_input or None, progress_callback=update_progress)
            results = []
            for body in bodies:
                try:
                    result = detector().analyze(body)
                    results.append(result["prediction"])
                except ValueError:
                    continue  # empty/unanalyzable message body
            progress_bar.empty()
            st.session_state.gmail_scan_results = results
            st.success(f"Scanned {len(bodies)} emails.")
        except GoogleApiError as e:
            st.error(f"Gmail API error: {e}")
        except Exception as e:
            st.error(f"Scan failed: {e}")

    results = st.session_state.get("gmail_scan_results")
    if results:
        counts = pd.Series(results).value_counts()
        chart_df = pd.DataFrame({"Risk": counts.index, "Count": counts.values})

        st.html("<div style='margin-top:1rem;'></div>")
        kpi = st.columns(4)
        with kpi[0]:
            stat_card(len(results), "Emails Scanned", color="#00e5ff")
        with kpi[1]:
            stat_card(int(counts.get("low", 0)), "Low Risk", color=RISK_COLORS["low"])
        with kpi[2]:
            stat_card(int(counts.get("medium", 0)), "Medium Risk", color=RISK_COLORS["medium"])
        with kpi[3]:
            stat_card(int(counts.get("high", 0)), "High Risk", color=RISK_COLORS["high"])

        st.html("<div style='margin-top:1rem;'></div>")
        st.html('<div class="mg-panel-title">Inbox Risk Breakdown</div>')
        st.plotly_chart(
            style_fig(px.pie(chart_df, names="Risk", values="Count", color="Risk", color_discrete_map=RISK_COLORS)),
            use_container_width=True,
        )


def history_page() -> None:
    page_header(
        "🕘", "Prediction History", "root@messageguard:~$ cat prediction_history.csv",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )
    history = load_history()

    counts = history["Prediction"].str.lower().value_counts() if not history.empty else pd.Series(dtype=int)
    kpi = st.columns(4)
    with kpi[0]:
        stat_card(len(history), "Total Analyses", color="#00e5ff")
    with kpi[1]:
        stat_card(int(counts.get("low", 0)), "Low Risk", color=RISK_COLORS["low"])
    with kpi[2]:
        stat_card(int(counts.get("medium", 0)), "Medium Risk", color=RISK_COLORS["medium"])
    with kpi[3]:
        stat_card(int(counts.get("high", 0)), "High Risk", color=RISK_COLORS["high"])

    st.html("<div style='margin-top:1rem;'></div>")

    # Track each row's position in the underlying saved file (via a hidden
    # column) so deletion still targets the right rows even when a search
    # filter has changed which rows are currently displayed.
    history = history.reset_index(drop=True)
    history["_orig_idx"] = history.index

    search = st.text_input("Search messages or predictions")
    if search:
        history = history[history.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)]

    display_history = history.copy()
    display_history.insert(0, "Select", False)

    if st.session_state.get("_select_all_history"):
        display_history["Select"] = True
        st.session_state.pop("_select_all_history", None)
        st.session_state.pop("history_editor", None)  # force the editor to reinit with all rows checked

    log_label_col, select_all_col = st.columns([4, 1])
    with log_label_col:
        st.html('<div class="mg-panel-title" style="margin-top:0.4rem;">Prediction Log</div>')
    with select_all_col:
        if st.button("Select All", use_container_width=True, disabled=display_history.empty):
            st.session_state._select_all_history = True
            st.rerun()

    edited = st.data_editor(
        display_history,
        key="history_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["Date", "Message", "Prediction", "Category", "Confidence", "Risk Score"],
        column_order=["Select", "Date", "Message", "Prediction", "Category", "Confidence", "Risk Score"],
        column_config={"Select": st.column_config.CheckboxColumn("", width="small")},
    )
    selected_rows = edited[edited["Select"]]

    export_col, delete_selected_col, delete_all_col = st.columns(3)
    with export_col:
        st.download_button(
            "Export history as CSV", history.drop(columns=["_orig_idx"]).to_csv(index=False).encode(),
            "prediction-history.csv", "text/csv", use_container_width=True,
        )
    with delete_selected_col:
        if st.button(
            f"Delete Selected ({len(selected_rows)})",
            use_container_width=True, disabled=selected_rows.empty,
        ):
            confirm_delete_dialog(selected_rows["_orig_idx"].tolist(), f"{len(selected_rows)} selected record(s)")
    with delete_all_col:
        if st.button("Delete All History", use_container_width=True, disabled=load_history().empty):
            confirm_delete_dialog(None, f"all {len(load_history())} record(s)")


@st.dialog("Confirm Deletion")
def confirm_delete_dialog(row_positions: list[int] | None, description: str) -> None:
    """Modal confirmation shown before any history deletion — row_positions=None means delete everything."""
    st.warning(f"This will permanently delete {description}. This action cannot be undone.")
    cancel_col, confirm_col = st.columns(2)
    with cancel_col:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with confirm_col:
        if st.button("Yes, Delete", type="primary", use_container_width=True):
            if row_positions is None:
                clear_history()
            else:
                delete_history_rows(row_positions)
            st.session_state.pop("history_editor", None)
            st.rerun()


def file_translation() -> None:
    page_header(
        "🔄", "File Translation",
        "root@messageguard:~$ convert any file into .txt or .eml",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )

    st.write(
        "Analyze Message only accepts .txt and .eml files directly. Upload a file in any "
        "supported format below, pick a target format, and convert it — you'll get a "
        "download button for the converted file, which you can then upload to Analyze Message."
    )
    st.caption(SUPPORTED_UPLOAD_FORMATS_MESSAGE)

    uploaded_file = st.file_uploader("Upload a file to convert", type=None)
    target_format = st.radio("Convert to", [".txt", ".eml"], horizontal=True)

    if st.button("Convert", type="primary", use_container_width=True, disabled=uploaded_file is None):
        try:
            with st.spinner("Extracting text and converting..."):
                text = extract_text_from_upload(uploaded_file.name, uploaded_file.read())
            base_name = Path(uploaded_file.name).stem
            if target_format == ".txt":
                converted_bytes = text.encode("utf-8")
                out_name = f"{base_name}.txt"
                mime = "text/plain"
            else:
                converted_bytes = build_eml_bytes(subject=base_name, body_text=text)
                out_name = f"{base_name}.eml"
                mime = "message/rfc822"
            st.session_state.converted_file = {"bytes": converted_bytes, "name": out_name, "mime": mime}
            st.success(f"Converted successfully — {out_name} is ready to download below.")
        except Exception as e:
            st.error(f"Conversion failed: {e}")
            st.session_state.pop("converted_file", None)

    converted = st.session_state.get("converted_file")
    if converted:
        st.download_button(
            f"Download {converted['name']}",
            converted["bytes"],
            converted["name"],
            converted["mime"],
            use_container_width=True,
        )


apply_theme()

pages = {
    "Home": home,
    "Analyze Message": analyze,
    "Analysis Details": analysis_details,
    "Dashboard": dashboard,
    "History": history_page,
    "File Translation": file_translation,
}

NAV_ICONS = {
    "Analyze Message": "🔍",
    "Dashboard": "📊",
    "History": "🕘",
    "File Translation": "🔄",
}
NAV_ITEMS = ["Analyze Message", "Dashboard", "History", "File Translation"]

if st.session_state.started:
    if not st.session_state.sidebar_visible:
        # Force-hide Streamlit's native sidebar via our own CSS (not relying
        # on Streamlit's collapse mechanism, which only reliably applies once
        # at first load) and give a normal, always-reachable button in the
        # main content area to bring it back.
        st.html("<style>section[data-testid='stSidebar']{display:none !important;}</style>")
        if st.button("☰  Show Sidebar", type="primary"):
            st.session_state.sidebar_visible = True
            st.rerun()
    else:
        st.sidebar.html(
            """
            <div class="mg-sidebar-titlebar">
                <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
                <span class="brand">MESSAGE_GUARD</span>
            </div>
            """
        )

        if st.sidebar.button("✕ Hide Sidebar", use_container_width=True):
            st.session_state.sidebar_visible = False
            st.rerun()

        if st.sidebar.button("← Back to Home", use_container_width=True):
            st.session_state.started = False
            go_to("Home")

        st.sidebar.html('<div class="mg-sidebar-label">&gt; Navigate</div>')
        if st.session_state.nav_page not in pages:
            st.session_state.nav_page = NAV_ITEMS[0]

        for item in NAV_ITEMS:
            label = f"{NAV_ICONS.get(item, '')}  {item}"
            is_active = item == st.session_state.nav_page
            if st.sidebar.button(
                label, key=f"nav_btn_{item}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ) and not is_active:
                st.session_state.nav_page = item
                st.rerun()

        best_model = metrics().get("best_model", "Not trained yet")
        st.sidebar.html('<div class="mg-sidebar-spacer"></div>')
        st.sidebar.html(
            f"""
            <div class="mg-sidebar-footer">
                <span class="pulse"></span>
                <div class="meta">
                    <div class="model">{best_model}</div>
                    <div class="status">MODEL ONLINE</div>
                </div>
            </div>
            """
        )
# else: nothing rendered in the sidebar on Home — it stays hidden

pages[st.session_state.nav_page]()