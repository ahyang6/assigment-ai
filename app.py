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
LABELS = ["ham", "spam", "phishing"]

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
    base = 100 * (probabilities.get("spam", 0) + probabilities.get("phishing", 0))
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
    if prediction == "ham" and not parts:
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
    initial_sidebar_state="expanded" if st.session_state.started else "collapsed",
)

PAGES = ["Home", "Analyze Message", "Dashboard", "History", "About"]


# ---------------------------------------------------------------------------
# Global styling (applies outside the hero iframe: buttons, cards, layout)
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
    html, body, [class*="css"]  {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }

    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }

    /* fade-in-up animation applied to page sections as they render */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .cf-fade {
        animation: fadeInUp 0.6s ease-out both;
    }

    /* Get Started / primary buttons: prominent, animated on hover */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f6821f, #ff9d3d);
        border: none;
        border-radius: 6px;
        color: #ffffff;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 1.6rem;
        box-shadow: 0 4px 14px rgba(246, 130, 31, 0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.015);
        box-shadow: 0 8px 22px rgba(246, 130, 31, 0.45);
        background: linear-gradient(135deg, #e0740f, #f6821f);
        color: #ffffff;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px) scale(0.99);
    }

    div[data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #ececec;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        transition: box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
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
    </script>
    """
    components.html(html, height=component_height, scrolling=False)


def apply_theme() -> None:
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.session_state.dark_mode = True
    st.markdown(
        """
        <style>
        .stApp { background: #0b0d12; color: #e5e7eb; }
        div[data-testid='stMetric'] { background:#171a21; border-color:#2a2e38; }

        /* ---- sidebar shell ---- */
        section[data-testid="stSidebar"] {
            background: #05070c;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
        section[data-testid="stSidebar"] hr { border-top: 1px solid rgba(255,255,255,0.08); }

        /* ---- nav radio list (shown after "Get Started") ---- */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 0.6rem 0.85rem;
            border-radius: 9px;
            border: 1px solid transparent;
            color: #aab3c2;
            transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(246, 130, 31, 0.10);
            border-color: rgba(246, 130, 31, 0.25);
            color: #f5f7fa;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(246, 130, 31, 0.16);
            border-color: rgba(246, 130, 31, 0.4);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div {
            color: #ffb066;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
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
             "Explanation:", result['explanation'], "", "Message:"]
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
    # edge-to-edge instead of sitting inside a small centered card.
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_hero("EMAIL DETECTION", dark=True)


def analyze() -> None:
    st.title("Analyze Message")

    sample = "URGENT! Verify your account now at https://secure-check.example or it will be suspended!!"

    # Upload file
    uploaded_file = st.file_uploader(
        "Or drag and drop an email or text file",
        type=["txt", "eml"]
    )

    uploaded_message = ""

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

    # Text area
    message = st.text_area(
        "Paste a text message or email",
        value=uploaded_message or st.session_state.get("message", ""),
        height=170,
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

    color = {
        "ham": "✅",
        "spam": "⚠️",
        "phishing": "🚨"
    }[result["prediction"]]

    st.subheader(f"{color} {result['prediction'].title()}")

    a, b, c = st.columns(3)

    a.metric("Confidence", f"{result['confidence']:.1%}")
    b.metric("Risk Score", f"{result['risk_score']}/100")
    c.metric("Risk Level", result["risk_level"])

    st.progress(result["risk_score"] / 100)

    st.write(result["explanation"])

    words = [
        word.upper()
        for values in result["indicators"]["keywords"].values()
        for word in values
    ]

    st.caption(
        "Detected Keywords: " +
        (", ".join(words) if words else "None")
    )

    st.code(
        f"Prediction: {result['prediction'].title()}\n"
        f"Confidence: {result['confidence']:.1%}\n"
        f"Risk: {result['risk_score']}/100 ({result['risk_level']})\n"
        f"{result['explanation']}",
        language=None,
    )

    if result["indicators"]["urls"]:
        st.warning(
            "Suspicious URL detected: " +
            ", ".join(result["indicators"]["urls"])
        )

    chart = pd.DataFrame({
        "Class": list(result["probabilities"]),
        "Probability": list(result["probabilities"].values())
    })

    st.plotly_chart(
        px.bar(
            chart,
            x="Class",
            y="Probability",
            range_y=[0, 1],
            color="Class"
        ),
        use_container_width=True
    )

    st.download_button(
        "Download Result as PDF",
        result_pdf(st.session_state.message, result),
        "message-analysis.pdf",
        "application/pdf"
    )


def dashboard() -> None:
    st.title("Statistics Dashboard")
    data, info, history = pd.read_csv(DATASET_PATH), metrics(), load_history()
    left, right = st.columns(2)
    left.plotly_chart(px.pie(data, names="label", title="Dataset class distribution"), use_container_width=True)
    if info:
        scores = pd.DataFrame([{"Model": name, "Accuracy": value["accuracy"], "F1": value["f1"]} for name, value in info["models"].items()])
        right.plotly_chart(px.bar(scores, x="Model", y=["Accuracy", "F1"], barmode="group", title="Model comparison"), use_container_width=True)
        st.caption(f"Selected model: {info['best_model']} · accuracy: {info['models'][info['best_model']]['accuracy']:.1%}")
    if not history.empty:
        history["Date"] = pd.to_datetime(history["Date"])
        daily = history.groupby(history["Date"].dt.date).size().reset_index(name="Analyses")
        st.plotly_chart(px.line(daily, x="Date", y="Analyses", markers=True, title="Daily analysis count"), use_container_width=True)
        st.plotly_chart(px.histogram(history, x="Risk Score", nbins=10, title="Risk distribution"), use_container_width=True)
    else:
        st.info("Analyse messages to populate prediction activity charts.")


def history_page() -> None:
    st.title("Prediction History")
    history = load_history(); search = st.text_input("Search messages or predictions")
    if search:
        history = history[history.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)]
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button("Export history as CSV", history.to_csv(index=False).encode(), "prediction-history.csv", "text/csv")
    if st.button("Delete all history"):
        clear_history(); st.rerun()


def about() -> None:
    # All of the informational content that used to live below the hero on
    # the Home page now lives here, shown once the user has clicked
    # "Get Started" and reached the About tab.
    st.title("About")

    st.markdown("## System Overview")
    cols = st.columns(3)
    cols[0].metric("Detection Classes", "Safe · Spam · Phishing")
    cols[1].metric("Best AI Model", metrics().get("best_model", "Train model"))
    cols[2].metric("Training Dataset", metrics().get("dataset_rows", "—"))

    st.divider()

    st.markdown("## Navigation Guide")
    st.markdown(
        """
    **Home** — Learn about the purpose of Message Guard and view system information.

    **Analyze Message** — Paste an email or text message to detect whether it is
    Safe, Spam, or Phishing. Shows prediction, confidence, risk score, explanation,
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
    if st.sidebar.button("← Back to Home", use_container_width=True):
        st.session_state.started = False
        go_to("Home")

    st.sidebar.markdown("#### Navigate")
    current_index = (
        NAV_ITEMS.index(st.session_state.nav_page)
        if st.session_state.nav_page in NAV_ITEMS
        else 0
    )
    selected = st.sidebar.radio(
        "Navigate",
        NAV_ITEMS,
        index=current_index,
        format_func=lambda p: f"{NAV_ICONS.get(p, '')}  {p}",
        label_visibility="collapsed",
    )
    if selected != st.session_state.nav_page:
        st.session_state.nav_page = selected
# else: nothing rendered in the sidebar on Home — it stays collapsed

pages[st.session_state.nav_page]()