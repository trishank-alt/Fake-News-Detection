# server.py

import os
import pickle
import re
import nltk
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer


# -----------------------------
# NLTK Setup
# -----------------------------
nltk.download("stopwords", quiet=True)
nltk.download("vader_lexicon", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
sia = SentimentIntensityAnalyzer()


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "logreg.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")

NEWS_API_KEY = "YOUR_KEY"   # <-- replace this


# -----------------------------
# Global model variables
# -----------------------------
model = None
vectorizer = None


# -----------------------------
# Utility Functions
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    words = [STEMMER.stem(w) for w in words]

    return " ".join(words)


def get_emotional_intensity(text):
    scores = sia.polarity_scores(text)
    return scores["compound"]


def verify_with_trusted_sources(text):
    if NEWS_API_KEY == "YOUR_KEY":
        return 0.0  # fail-safe if key not set

    try:
        url = f"https://newsapi.org/v2/everything?q={text}&language=en&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if data.get("totalResults", 0) > 0:
            return min(data["totalResults"] / 100, 1.0)

        return 0.0
    except:
        return 0.0


def get_suspicious_score(text):
    suspicious_words = [
        "shocking", "breaking", "exposed", "truth revealed",
        "you won't believe", "secret", "urgent", "alert"
    ]

    text_lower = text.lower()
    score = 0

    for word in suspicious_words:
        if word in text_lower:
            score += 0.1

    # CAPS ratio
    caps = sum(1 for c in text if c.isupper())
    ratio = caps / max(len(text), 1)

    if ratio > 0.3:
        score += 0.2

    return min(score, 1.0)


# -----------------------------
# Load Model
# -----------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError(f"Vectorizer not found: {VECTORIZER_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# -----------------------------
# Lifespan (startup)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer

    print("Starting server...")
    print("Loading ML model...")

    model, vectorizer = load_model()

    print("Model loaded successfully.")

    yield

    print("Shutting down server...")


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="Fake News Detection API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Schemas
# -----------------------------
class NewsInput(BaseModel):
    text: str


class Prediction(BaseModel):
    prediction: str
    confidence: float
    is_fake: bool
    emotional_intensity: float
    verification_score: float
    suspicious_score: float
    truth_score: float


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Fake News Detection API Running"}


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict", response_model=Prediction)
def predict(news: NewsInput):

    if model is None or vectorizer is None:
        raise HTTPException(500, "Model not loaded")

    if not news.text.strip():
        raise HTTPException(400, "Empty input")

    # Preprocess
    processed = preprocess_text(news.text)
    vec = vectorizer.transform([processed])

    # Model prediction
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = float(prob[pred])

    # Signals
    emotion = get_emotional_intensity(news.text)
    verification = verify_with_trusted_sources(news.text[:100])
    suspicious = get_suspicious_score(news.text)

    # Truth Score Calculation
    truth_score = (
        confidence
        - (emotion * 0.3)
        + (verification * 0.5)
        - (suspicious * 0.4)
    )

    truth_score = max(0, min(truth_score, 1))

    return Prediction(
        prediction="Fake" if pred == 1 else "Real",
        confidence=confidence,
        is_fake=bool(pred == 1),
        emotional_intensity=emotion,
        verification_score=verification,
        suspicious_score=suspicious,
        truth_score=truth_score
    )