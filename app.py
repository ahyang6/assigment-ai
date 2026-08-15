"""Streamlit interface for the AI-powered spam and phishing detector."""
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

import joblib
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from email import policy
from email.parser import BytesParser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
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


def append_history(message: str, prediction: str, confidence: float, risk: int) -> None:
    """Persist one analysis result to the local CSV history."""
    new_file = not HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["Date", "Message", "Prediction", "Confidence", "Risk Score"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), message, prediction, round(confidence, 4), risk])


def load_history() -> pd.DataFrame:
    """Return saved predictions, or an empty table with the expected columns."""
    columns = ["Date", "Message", "Prediction", "Confidence", "Risk Score"]
    return pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame(columns=columns)


def clear_history() -> None:
    """Remove all saved prediction history."""
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()


def delete_history_rows(row_positions: list[int]) -> None:
    """Remove specific rows (by their position in the saved file) from history."""
    if not HISTORY_PATH.exists():
        return
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
        n_jobs=-1,
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
        prediction = max(probabilities, key=probabilities.get)
        indicators = find_indicators(message)
        risk, risk_level = calculate_risk(probabilities, indicators)
        return {"prediction": prediction, "confidence": float(probabilities[prediction]), "probabilities": probabilities,
                "risk_score": risk, "risk_level": risk_level, "indicators": indicators,
                "explanation": explanation(prediction, indicators)}

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
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
        max_features=8000,
        sublinear_tf=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    x_all_vec = vectorizer.transform(data["processed"])

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
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
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from email import policy
from email.parser import BytesParser


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

st.set_page_config(
    page_title="Message Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Home", "Analyze Message", "Dashboard", "History", "About"]


# ---------------------------------------------------------------------------
# Global styling (applies outside the hero iframe: buttons, cards, layout)
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&display=swap');

    :root {
        --mg-bg: #05070a;
        --mg-surface: #0c1016;
        --mg-border: rgba(0, 229, 255, 0.18);
        --mg-cyan: #00e5ff;
        --mg-orange: #f6821f;
        --mg-text: #d7dee8;
        --mg-text-dim: #7d8b9c;
        --mg-low: #39ff88;
        --mg-medium: #ffb020;
        --mg-high: #ff3b5c;
    }

    html, body, [class*="css"] {
        font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
    }

    /* themed scrollbar so the browser's default (light) scrollbar doesn't
       clash against the dark terminal background */
    html {
        scrollbar-width: thin;
        scrollbar-color: rgba(0, 229, 255, 0.45) #05070a;
    }
    html::-webkit-scrollbar,
    body::-webkit-scrollbar,
    *::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    html::-webkit-scrollbar-track,
    body::-webkit-scrollbar-track,
    *::-webkit-scrollbar-track {
        background: #0a0e14;
    }
    html::-webkit-scrollbar-thumb,
    body::-webkit-scrollbar-thumb,
    *::-webkit-scrollbar-thumb {
        background: rgba(0, 229, 255, 0.45);
        border-radius: 2px;
        border: 2px solid #0a0e14;
    }
    html::-webkit-scrollbar-thumb:hover,
    body::-webkit-scrollbar-thumb:hover,
    *::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 229, 255, 0.7);
    }

    .stApp {
        background-color: var(--mg-bg);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    hr {
        border: none;
        border-top: 1px solid var(--mg-border);
        margin: 2rem 0;
    }

    /* fade-in-up animation applied to page sections as they render */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .cf-fade {
        animation: fadeInUp 0.5s ease-out both;
    }

    /* Consistent terminal-style page header used across every inner page */
    .mg-page-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.2rem;
    }
    .mg-page-header .mg-prompt {
        color: var(--mg-cyan);
        font-weight: 700;
        font-size: 1.6rem;
    }
    .mg-page-header .mg-title {
        color: var(--mg-text);
        font-weight: 800;
        font-size: 1.6rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .mg-page-subtitle {
        color: var(--mg-text-dim);
        font-size: 0.9rem;
        margin: 0.15rem 0 1.6rem 0;
        border-left: 2px solid var(--mg-border);
        padding-left: 0.6rem;
    }

    /* Get Started / primary buttons: sharp, neon-edged, animated on hover */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f6821f, #ff9d3d);
        border: 1px solid rgba(255, 157, 61, 0.6);
        border-radius: 2px;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.0rem;
        letter-spacing: 0.03em;
        padding: 0.75rem 1.6rem;
        box-shadow: 0 0 16px rgba(246, 130, 31, 0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 26px rgba(246, 130, 31, 0.55);
        background: linear-gradient(135deg, #e0740f, #f6821f);
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px);
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: 2px;
        border: 1px solid var(--mg-border);
        background: var(--mg-surface);
        color: var(--mg-text);
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--mg-cyan);
        color: var(--mg-cyan);
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.25);
    }

    div[data-testid="stMetric"] {
        background: var(--mg-surface);
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        padding: 0.9rem 1rem;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--mg-cyan);
        box-shadow: 0 0 16px rgba(0, 229, 255, 0.18);
    }
    div[data-testid="stMetricValue"] {
        font-family: "JetBrains Mono", monospace;
    }

    /* Generic terminal-panel card, used for the analysis verdict etc. */
    .mg-terminal-card {
        background: var(--mg-surface);
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        overflow: hidden;
        margin-bottom: 1.2rem;
    }
    .mg-terminal-card-bar {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 0.5rem 0.8rem;
        background: rgba(255,255,255,0.02);
        border-bottom: 1px solid var(--mg-border);
    }
    .mg-terminal-card-bar span.dot {
        width: 9px; height: 9px; border-radius: 50%;
        display: inline-block;
    }
    .mg-terminal-card-bar .dot.r { background: #ff5f56; }
    .mg-terminal-card-bar .dot.y { background: #ffbd2e; }
    .mg-terminal-card-bar .dot.g { background: #27c93f; }
    .mg-terminal-card-bar .label {
        color: var(--mg-text-dim);
        font-size: 0.78rem;
        margin-left: 0.4rem;
        letter-spacing: 0.03em;
    }
    .mg-terminal-card-body {
        padding: 1.4rem 1.5rem;
    }

    /* Circular risk-score gauge (pure CSS conic-gradient ring) */
    .mg-gauge {
        width: 108px;
        height: 108px;
        border-radius: 50%;
        background: conic-gradient(var(--gauge-color) calc(var(--gauge-pct) * 1%), rgba(255,255,255,0.08) 0);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 0 18px var(--gauge-glow);
    }
    .mg-gauge-inner {
        width: 82px;
        height: 82px;
        border-radius: 50%;
        background: var(--mg-surface);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .mg-gauge-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f1f4f8;
        line-height: 1;
    }
    .mg-gauge-label {
        font-size: 0.68rem;
        color: var(--mg-text-dim);
        margin-top: 2px;
    }

    /* small inline tags used for detected keywords / URLs / signals */
    .mg-badge {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px 5px 2px 0;
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        font-size: 0.72rem;
        color: var(--mg-cyan);
        background: rgba(0, 229, 255, 0.06);
    }
    .mg-badge.danger {
        color: var(--mg-high);
        border-color: rgba(255, 59, 92, 0.4);
        background: rgba(255, 59, 92, 0.08);
    }

    /* compact panel used in the dashboard-style Analyze grid */
    .mg-panel-title {
        color: var(--mg-text-dim);
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* give native Plotly / dataframe widgets the same bordered card look
       as the pure-HTML terminal cards, so charts and tables match the
       Analyze page's visual language */
    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        background: var(--mg-surface);
        padding: 0.5rem;
    }

    /* compact KPI stat card (value + label, no dot-bar) */
    .mg-stat-card {
        border: 1px solid var(--mg-border);
        border-radius: 2px;
        background: var(--mg-surface);
        padding: 0.9rem 1rem;
        height: 92px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .mg-stat-card .value {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .mg-stat-card .label {
        color: var(--mg-text-dim);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.35rem;
    }
</style>
"""

# ---------------------------------------------------------------------------
# Hero section: dark cybersecurity background + mouse-triggered glitch title
# Rendered as a self-contained HTML component (own CSS + JS, no Streamlit
# rerun involved) so the scramble animation is instant and client-side only.
# The hero now fills the entire viewport (full-bleed, edge-to-edge, no
# rounded corners) instead of sitting in a small centered card.
# ---------------------------------------------------------------------------
def render_hero(title: str = "EMAIL DETECTION", dark: bool = True, component_height: int = 900) -> None:
    # Theme tokens: plain white surface in light mode, plain black in dark
    # mode (no gradient wash) so the toggle reads clearly as two states.
    if dark:
        bg = "#000000"
        grid_line = "rgba(255, 255, 255, 0.07)"
        glyph_color = "rgba(255, 255, 255, 0.16)"
        glyph_flash = "rgba(246, 130, 31, 0.55)"
        title_color = "#f5f7fa"
        title_glow = "rgba(56, 189, 248, 0.25)"
        sub_color = "#aab3c2"
        icon_glow = "rgba(56, 189, 248, 0.45)"
    else:
        bg = "#ffffff"
        grid_line = "rgba(15, 23, 42, 0.06)"
        glyph_color = "rgba(15, 23, 42, 0.10)"
        glyph_flash = "rgba(246, 130, 31, 0.65)"
        title_color = "#14181f"
        title_glow = "rgba(246, 130, 31, 0.12)"
        sub_color = "#5b6472"
        icon_glow = "rgba(246, 130, 31, 0.35)"

    html = f"""
    <div class="hero-wrap">
      <style>
        * {{ box-sizing: border-box; }}
        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
        }}
        .hero-wrap {{
            position: relative;
            width: 100%;
            height: 100vh;
            min-height: 100%;
            border-radius: 0;
            overflow: hidden;
            background: {bg};
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            animation: heroFadeIn 0.8s ease-out both;
        }}
        @keyframes heroFadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        /* faint circuit / grid overlay for the cybersecurity feel */
        .hero-grid {{
            position: absolute;
            inset: 0;
            background-image:
                linear-gradient({grid_line} 1px, transparent 1px),
                linear-gradient(90deg, {grid_line} 1px, transparent 1px);
            background-size: 32px 32px;
            mask-image: radial-gradient(circle at 50% 40%, black 0%, transparent 75%);
        }}

        /* background "garbled code" layer: a grid of monospace glyphs.
           They idle on "/" and only scramble to other characters where
           the cursor has passed, like a glitch trail, then settle back
           down to "/" again once left alone. */
        .hero-glyphs {{
            position: absolute;
            inset: 0;
            display: grid;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 12px;
            line-height: 1;
            color: {glyph_color};
            user-select: none;
            pointer-events: none;
            mask-image: radial-gradient(circle at 50% 45%, transparent 0%, transparent 28%, black 60%, black 100%);
        }}
        .hero-glyphs span {{
            transition: color 1.1s ease-out;
        }}
        .hero-glyphs span.flash {{
            color: {glyph_flash};
            transition: color 0.05s ease-out;
        }}

        .hero-inner {{
            position: relative;
            z-index: 2;
            text-align: center;
            padding: 2.5rem 1.25rem;
            max-width: 720px;
        }}

        .hero-icon {{
            font-size: 2.4rem;
            margin-bottom: 0.5rem;
            filter: drop-shadow(0 0 10px {icon_glow});
        }}

        .glitch-title {{
            font-size: clamp(2.1rem, 6vw, 3.6rem);
            font-weight: 800;
            letter-spacing: 0.06em;
            color: {title_color};
            margin: 0 0 0.9rem 0;
            cursor: default;
            text-shadow: 0 0 18px {title_glow};
            user-select: none;
        }}
        .glitch-title span.char {{
            display: inline-block;
            min-width: 0.15em;
        }}

        .hero-sub {{
            font-size: clamp(0.92rem, 2vw, 1.05rem);
            color: {sub_color};
            line-height: 1.6;
            margin: 0 auto;
        }}

        /* "Get Started" is a real link baked into the hero markup, styled
           to match the app's primary-button look, so it reads as one
           integrated piece with the background rather than a separate
           Streamlit widget sitting below it. */
        .hero-cta {{
            display: inline-block;
            margin-top: 2rem;
            background: linear-gradient(135deg, #f6821f, #ff9d3d);
            color: #ffffff;
            font-weight: 700;
            font-size: 1.05rem;
            text-decoration: none;
            padding: 0.85rem 2.3rem;
            border-radius: 8px;
            box-shadow: 0 4px 14px rgba(246, 130, 31, 0.35);
            transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
            cursor: pointer;
        }}
        .hero-cta:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 24px rgba(246, 130, 31, 0.45);
            background: linear-gradient(135deg, #e0740f, #f6821f);
        }}
        .hero-cta:active {{
            transform: translateY(0) scale(0.98);
        }}

        @media (max-width: 640px) {{
            .hero-inner {{ padding: 1.75rem 1rem; }}
        }}
      </style>

      <div class="hero-grid"></div>
      <div class="hero-glyphs" id="heroGlyphs"></div>
      <div class="hero-inner">
        <div class="hero-icon">🛡️</div>
        <h1 class="glitch-title" id="glitchTitle"></h1>
        <p class="hero-sub">
          AI-powered spam and phishing detection system that analyses emails and
          messages using NLP and Machine Learning.
        </p>
        <a href="#" class="hero-cta" onclick="goToApp(); return false;">Get Started →</a>
      </div>
    </div>

    <script>
      // Navigates the *top* Streamlit page (not this component iframe) to
      // "?start=1". Built explicitly off window.parent.location rather than
      // a plain relative href, since relative URLs inside a srcdoc iframe
      // don't reliably resolve against the parent page.
      function goToApp() {{
        try {{
          const parentLoc = window.parent.location;
          const base = parentLoc.origin + parentLoc.pathname;
          parentLoc.href = base + "?start=1";
        }} catch (e) {{
          window.location.href = "?start=1";
        }}
      }}

      (function() {{
        const target = {json.dumps(title)};
        const el = document.getElementById('glitchTitle');
        const glitchChars = "!<>-_\\\\/[]{{}}—=+*^?#$%&0123456789";
        let frame = null;
        let running = false;

        function buildSpans(text) {{
            el.innerHTML = "";
            for (const ch of text) {{
                const span = document.createElement('span');
                span.className = 'char';
                span.textContent = ch === ' ' ? '\\u00A0' : ch;
                el.appendChild(span);
            }}
        }}

        function randomChar() {{
            return glitchChars[Math.floor(Math.random() * glitchChars.length)];
        }}

        function playScramble() {{
            if (running) return;
            running = true;
            const spans = Array.from(el.querySelectorAll('.char'));
            const total = spans.length;
            const revealDelayPerChar = 55; // ms between each letter locking in
            let startTime = performance.now();

            function tick(now) {{
                const elapsed = now - startTime;
                const revealCount = Math.min(total, Math.floor(elapsed / revealDelayPerChar));

                for (let i = 0; i < total; i++) {{
                    const original = target[i] === ' ' ? '\\u00A0' : target[i];
                    if (i < revealCount) {{
                        spans[i].textContent = original;
                    }} else if (original === '\\u00A0') {{
                        spans[i].textContent = original;
                    }} else {{
                        spans[i].textContent = randomChar();
                    }}
                }}

                if (revealCount < total) {{
                    frame = requestAnimationFrame(tick);
                }} else {{
                    running = false;
                }}
            }}

            frame = requestAnimationFrame(tick);
        }}

        function resetTitle() {{
            if (frame) cancelAnimationFrame(frame);
            running = false;
            buildSpans(target);
        }}

        buildSpans(target);
        el.addEventListener('mouseenter', playScramble);
        el.addEventListener('mouseleave', resetTitle);

        // ---- background glyph noise, reacts only to the cursor ----
        // Every cell idles on "/". Moving the mouse over the hero makes
        // nearby cells flicker through random characters (a glitch
        // trail); each touched cell keeps flickering on its own for a
        // little while afterwards, gradually slowing down, before
        // settling back to "/" again — even if the mouse has already
        // moved on or left.
        (function() {{
            const wrap = document.querySelector('.hero-wrap');
            const layer = document.getElementById('heroGlyphs');
            const cell = 22; // px per glyph cell
            const idleChar = "/";
            const noiseChars = "01AXF$#%&*<>/\\\\{{}}[]=+;:";
            let cols = 0, rows = 0;

            function buildGrid() {{
                cols = Math.ceil(wrap.clientWidth / cell);
                rows = Math.ceil(wrap.clientHeight / cell);
                layer.style.gridTemplateColumns = `repeat(${{cols}}, ${{cell}}px)`;
                layer.style.gridTemplateRows = `repeat(${{rows}}, ${{cell}}px)`;
                layer.innerHTML = "";
                const total = cols * rows;
                for (let i = 0; i < total; i++) {{
                    const span = document.createElement('span');
                    span.textContent = idleChar;
                    span.style.textAlign = 'center';
                    layer.appendChild(span);
                }}
            }}

            // Kicks off (or restarts) a decaying flicker on a single glyph
            // cell: it rapidly cycles through random characters, gradually
            // slowing down, then locks back to the idle "/" character.
            function triggerGlyph(span) {{
                if (span._glyphTimer) {{
                    clearTimeout(span._glyphTimer);
                }}
                const start = performance.now();
                const duration = 900 + Math.random() * 700; // total settle time

                (function step() {{
                    const elapsed = performance.now() - start;
                    if (elapsed > duration) {{
                        span.textContent = idleChar;
                        span.classList.remove('flash');
                        span._glyphTimer = null;
                        return;
                    }}
                    span.textContent = noiseChars[Math.floor(Math.random() * noiseChars.length)];
                    span.classList.add('flash');
                    setTimeout(() => span.classList.remove('flash'), 80);

                    const progress = elapsed / duration;
                    const nextDelay = 35 + progress * 150; // flicker slows as it settles
                    span._glyphTimer = setTimeout(step, nextDelay);
                }})();
            }}

            buildGrid();
            window.addEventListener('resize', buildGrid);

            let lastMove = 0;
            wrap.addEventListener('mousemove', function(e) {{
                const now = performance.now();
                if (now - lastMove < 35) return; // light throttle
                lastMove = now;

                const rect = wrap.getBoundingClientRect();
                const col = Math.floor((e.clientX - rect.left) / cell);
                const row = Math.floor((e.clientY - rect.top) / cell);
                const radius = 2;

                for (let dr = -radius; dr <= radius; dr++) {{
                    for (let dc = -radius; dc <= radius; dc++) {{
                        const rr = row + dr, cc = col + dc;
                        if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue;
                        if (Math.sqrt(dr * dr + dc * dc) > radius) continue;
                        if (Math.random() > 0.5) continue; // keep the trail sparse
                        const span = layer.children[rr * cols + cc];
                        if (!span) continue;
                        triggerGlyph(span);
                    }}
                }}
            }});
        }})();
      }})();

      // Resize this iframe to the real browser viewport height. "100vh" CSS
      // inside an iframe only ever refers to the iframe's own fixed height
      // (set below via the `height` prop passed to components.html), not the
      // actual browser window — without this, tall screens are left with
      // empty space below a hero sized to the smaller fallback height.
      (function() {{
        function resizeToViewport() {{
          try {{
            var target = window.parent.innerHeight;
            if (window.frameElement && target && window.frameElement.style.height !== target + 'px') {{
              window.frameElement.style.height = target + 'px';
              // let the glyph-grid's own resize listener recompute its
              // row/column count against the corrected height
              window.dispatchEvent(new Event('resize'));
            }}
          }} catch (e) {{}}
        }}
        resizeToViewport();
        window.addEventListener('resize', resizeToViewport);
      }})();
    </script>
    """
    components.html(html, height=component_height, scrolling=False)


def apply_theme() -> None:
    st.session_state.dark_mode = True
    st.html(
        GLOBAL_CSS
        + """
        <style>
        .stApp { background: #05070a; color: #d7dee8; }
        div[data-testid='stMetric'] { background:#0c1016; border-color: rgba(0, 229, 255, 0.18); }

        /* ---- sidebar shell ---- */
        section[data-testid="stSidebar"] {
            background: #030405;
            border-right: 1px solid rgba(0, 229, 255, 0.10);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        section[data-testid="stSidebar"] hr { border-top: 1px solid rgba(0, 229, 255, 0.10); }

        /* pin the footer status card to the bottom of the sidebar */
        section[data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .mg-sidebar-spacer { flex-grow: 1; }

        /* ---- mini terminal titlebar at the top of the sidebar ---- */
        .mg-sidebar-titlebar {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 0.5rem 0.1rem 0.9rem 0.1rem;
            margin-bottom: 0.6rem;
            border-bottom: 1px solid rgba(0, 229, 255, 0.10);
        }
        .mg-sidebar-titlebar span.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .mg-sidebar-titlebar .dot.r { background: #ff5f56; }
        .mg-sidebar-titlebar .dot.y { background: #ffbd2e; }
        .mg-sidebar-titlebar .dot.g { background: #27c93f; }
        .mg-sidebar-titlebar .brand {
            color: var(--mg-text);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            margin-left: 0.35rem;
        }

        .mg-sidebar-label {
            color: #566373;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 0.2rem 0 0.5rem 0.1rem;
        }

        /* ---- nav buttons (one real st.button per item, so each can carry
           its own icon/badge and get a proper active-state accent bar —
           a native st.radio can't support that per-option styling) ---- */
        section[data-testid="stSidebar"] .stButton > button {
            text-align: left;
            justify-content: flex-start;
            padding: 0.6rem 0.85rem;
            border-radius: 4px;
            font-size: 0.92rem;
            letter-spacing: 0;
            transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: transparent;
            border: 1px solid transparent;
            border-left: 3px solid transparent;
            color: #7d8b9c;
            font-weight: 500;
            box-shadow: none;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background: rgba(0, 229, 255, 0.06);
            border-left-color: rgba(0, 229, 255, 0.4);
            color: #f5f7fa;
            box-shadow: none;
            transform: none;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(246, 130, 31, 0.14) !important;
            border: 1px solid transparent !important;
            border-left: 3px solid #f6821f !important;
            color: #ffb066 !important;
            font-weight: 700;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background: rgba(246, 130, 31, 0.22) !important;
            transform: none;
        }

        /* ---- status footer card, pinned to the bottom via the spacer above ---- */
        .mg-sidebar-footer {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.8rem 0.9rem;
            border: 1px dashed rgba(0, 229, 255, 0.3);
            border-radius: 6px;
            background: rgba(0, 229, 255, 0.03);
            margin: 0.8rem 0 1rem 0;
        }
        .mg-sidebar-footer .pulse {
            width: 8px; height: 8px; border-radius: 50%;
            background: #39ff88;
            box-shadow: 0 0 8px #39ff88;
            flex-shrink: 0;
        }
        .mg-sidebar-footer .meta { line-height: 1.3; }
        .mg-sidebar-footer .meta .model {
            color: var(--mg-text);
            font-size: 0.78rem;
            font-weight: 700;
        }
        .mg-sidebar-footer .meta .status {
            color: #566373;
            font-size: 0.68rem;
            letter-spacing: 0.04em;
        }
        </style>
        """
    )


RISK_COLORS = {"low": "#39ff88", "medium": "#ffb020", "high": "#ff3b5c"}
RISK_ICONS = {"low": "✅", "medium": "⚠️", "high": "🚨"}


def page_header(icon: str, title: str, subtitle: str, extra_style: str = "") -> None:
    """Render the consistent terminal-style header used on every inner page.
    `extra_style` lets a page fold its own scoped <style> tag into this same
    element instead of issuing a separate, otherwise-empty st.markdown call."""
    st.html(
        f"""
        {extra_style}
        <div class="mg-page-header cf-fade">
            <span class="mg-prompt">{icon}</span>
            <span class="mg-title">{title}</span>
        </div>
        <div class="mg-page-subtitle cf-fade">{subtitle}</div>
        """
    )


def style_fig(fig):
    """Apply the dark terminal theme to a Plotly figure so charts blend with the app."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="JetBrains Mono, monospace",
        font_color="#d7dee8",
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


def stat_card(value, label: str, color: str = "#00e5ff", value_size: str = "1.5rem") -> None:
    """Render a compact KPI stat card matching the Analyze page's card language."""
    st.html(
        f"""
        <div class="mg-stat-card cf-fade">
            <div class="value" style="color:{color}; font-size:{value_size};">{value}</div>
            <div class="label">{label}</div>
        </div>
        """
    )


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
    lines = ["Message Guard - Analysis Report", "", f"Prediction: {result['prediction'].title()}",
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
    with st.expander("Or drag and drop an email or text file (.txt / .eml)"):
        uploaded_file = st.file_uploader("Upload file", type=["txt", "eml"], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".eml"):
                    email_message = BytesParser(
                        policy=policy.default
                    ).parse(uploaded_file)

                    if email_message.is_multipart():
                        body = email_message.get_body(preferencelist=("plain",))
                        uploaded_message = body.get_content() if body else ""
                    else:
                        uploaded_message = email_message.get_content()

                else:
                    uploaded_message = uploaded_file.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

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

            append_history(
                message,
                result["prediction"],
                result["confidence"],
                result["risk_score"]
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


def dashboard() -> None:
    page_header(
        "📊", "Statistics Dashboard", "root@messageguard:~$ tail -f model_metrics.log",
        extra_style="<style>.block-container { max-width: 1280px !important; }</style>",
    )
    data, info, history = pd.read_csv(DATASET_PATH), metrics(), load_history()

    kpi = st.columns(5)
    with kpi[0]:
        stat_card(len(data), "Dataset Rows", color="#00e5ff")
    with kpi[1]:
        stat_card(info.get("best_model", "—"), "Best Model", color="#f6821f", value_size="1.05rem")
    with kpi[2]:
        acc = info.get("models", {}).get(info.get("best_model", ""), {}).get("accuracy")
        stat_card(f"{acc:.1%}" if acc is not None else "—", "Model Accuracy", color="#39ff88")
    with kpi[3]:
        stat_card(len(history), "Total Analyses", color="#ffb020")
    with kpi[4]:
        stat_card(len(info.get("models", {})) or "—", "Algorithms Compared", color="#00e5ff")

    st.html("<div style='margin-top:1rem;'></div>")

    left, right = st.columns(2)
    with left:
        st.html('<div class="mg-panel-title">Dataset Class Distribution</div>')
        st.plotly_chart(
            style_fig(px.pie(data, names="label", color="label", color_discrete_map=RISK_COLORS)),
            use_container_width=True,
        )
    if info:
        with right:
            st.html('<div class="mg-panel-title">Model Comparison</div>')
            scores = pd.DataFrame([{"Model": name, "Accuracy": value["accuracy"], "F1": value["f1"]} for name, value in info["models"].items()])
            st.plotly_chart(
                style_fig(px.bar(
                    scores, x="Model", y=["Accuracy", "F1"], barmode="group",
                    color_discrete_sequence=["#00e5ff", "#f6821f"],
                )),
                use_container_width=True,
            )
            st.caption(f"Selected model: {info['best_model']} · accuracy: {info['models'][info['best_model']]['accuracy']:.1%}")

    if info:
        st.html("<div style='margin-top:1rem;'></div>")
        st.html('<div class="mg-panel-title">Algorithm Comparison</div>')
        model_rows = [
            {
                "Algorithm": name,
                "Type": "Clustering" if "Clustering" in name else "Supervised",
                "Accuracy": m["accuracy"],
                "F1 Score": m["f1"],
                "CV F1": m["cv_f1"],
            }
            for name, m in info["models"].items()
        ]
        models_df = pd.DataFrame(model_rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)

        def _highlight_best(row):
            is_best = row["Algorithm"] == info["best_model"]
            style = "background-color: rgba(246,130,31,0.16); color:#ffb066; font-weight:700;" if is_best else ""
            return [style] * len(row)

        styled_models = (
            models_df.style
            .apply(_highlight_best, axis=1)
            .format({"Accuracy": "{:.1%}", "F1 Score": "{:.3f}", "CV F1": "{:.3f}"})
        )
        st.dataframe(styled_models, use_container_width=True, hide_index=True)

    if not history.empty:
        history["Date"] = pd.to_datetime(history["Date"])
        daily = history.groupby(history["Date"].dt.date).size().reset_index(name="Analyses")
        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            st.html('<div class="mg-panel-title">Daily Analysis Count</div>')
            st.plotly_chart(
                style_fig(px.line(daily, x="Date", y="Analyses", markers=True,
                                   color_discrete_sequence=["#00e5ff"])),
                use_container_width=True,
            )
        with bottom_right:
            st.html('<div class="mg-panel-title">Risk Score Distribution</div>')
            st.plotly_chart(
                style_fig(px.histogram(history, x="Risk Score", nbins=10,
                                        color_discrete_sequence=["#f6821f"])),
                use_container_width=True,
            )
    else:
        st.info("Analyse messages to populate prediction activity charts.")


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
        disabled=["Date", "Message", "Prediction", "Confidence", "Risk Score"],
        column_order=["Select", "Date", "Message", "Prediction", "Confidence", "Risk Score"],
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


def about() -> None:
    # All of the informational content that used to live below the hero on
    # the Home page now lives here, shown once the user has clicked
    # "Get Started" and reached the About tab.
    page_header("ℹ️", "About", "root@messageguard:~$ man message-guard")

    st.markdown("## System Overview")
    cols = st.columns(3)
    cols[0].metric("Detection Classes", "Low · Medium · High Risk")
    cols[1].metric("Best AI Model", metrics().get("best_model", "Train model"))
    cols[2].metric("Training Dataset", metrics().get("dataset_rows", "—"))

    st.divider()

    st.markdown("## Navigation Guide")
    st.markdown(
        """
    **Home** — Learn about the purpose of Message Guard and view system information.

    **Analyze Message** — Paste an email or text message to detect whether it is
    Low, Medium, or High risk. Shows prediction, confidence, risk score, explanation,
    suspicious keywords/URLs, and a downloadable PDF report.

    **Dashboard** — Dataset class distribution, model comparison, daily analysis
    activity, and risk score distribution.

    **History** — View, search, export, or delete previous prediction records.

    **About** — Technologies used, machine learning models, and project purpose.
    """
    )

    st.divider()

    st.markdown("## How Message Guard Works")
    st.markdown(
        """
    1. Paste a message or email into **Analyze Message**.
    2. The system preprocesses the text using NLP techniques.
    3. The trained machine learning model analyses the message.
    4. A prediction, confidence score, risk score, and explanation are generated.
    5. The analysis is saved to the local prediction history.
    """
    )

    st.divider()

    st.write("This educational project compares Multinomial Naive Bayes, Logistic Regression, and SVM on TF-IDF features. The best weighted-F1 model is saved locally.")
    st.warning("Predictions are decision support, not a replacement for security controls. Do not open unexpected links or disclose credentials.")


apply_theme()

pages = {
    "Home": home,
    "Analyze Message": analyze,
    "Analysis Details": analysis_details,
    "Dashboard": dashboard,
    "History": history_page,
    "About": about,
}

NAV_ICONS = {
    "Analyze Message": "🔍",
    "Dashboard": "📊",
    "History": "🕘",
    "About": "ℹ️",
}
NAV_ITEMS = ["Analyze Message", "Dashboard", "History", "About"]

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

        history_count = len(load_history())
        nav_badges = {"History": history_count} if history_count else {}

        for item in NAV_ITEMS:
            label = f"{NAV_ICONS.get(item, '')}  {item}"
            if item in nav_badges:
                label += f"  ·  {nav_badges[item]}"
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