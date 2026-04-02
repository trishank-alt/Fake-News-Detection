# server.py

import os
import pickle
import re
import nltk

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('vader_lexicon')

# Download stopwords (first time only)
nltk.download("stopwords", quiet=True)


# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "logreg.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")


# Load NLP resources
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# Global model variables
model = None
vectorizer = None

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def get_emotional_intensity(text):
    scores = sia.polarity_scores(text)
    return scores["compound"]  

import requests

NEWS_API_KEY = "YOUR_KEY"

def verify_with_trusted_sources(text):
    url = f"https://newsapi.org/v2/everything?q={text}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data.get("totalResults", 0) > 0:
        return min(data["totalResults"] / 100, 1.0)  # normalize score
    return 0.0

# Preprocess function
def preprocess_text(text):

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    words = [STEMMER.stem(w) for w in words]

    return " ".join(words)


# Load trained model
def load_model():

    print("Looking for model at:", MODEL_PATH)
    print("Looking for vectorizer at:", VECTORIZER_PATH)

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(VECTORIZER_PATH):
        raise RuntimeError(f"Vectorizer not found: {VECTORIZER_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# Lifespan handler (modern FastAPI startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, vectorizer

    print("Starting server...")
    print("Loading ML model...")

    model, vectorizer = load_model()

    print("Model loaded successfully.")

    yield

    print("Shutting down server...")


# FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class NewsInput(BaseModel):
    text: str


# Output schema
class Prediction(BaseModel):
    prediction: str
    confidence: float
    is_fake: bool
    emotional_intensity: float
    verification_score: float


# Root endpoint
@app.get("/")
def root():
    return {"message": "Fake News Detection API Running"}


# Health check
@app.get("/health")
def health():
    return {"status": "OK"}


#Prediction endpoint
@app.post("/predict", response_model=Prediction)
def predict(news: NewsInput):

    if model is None or vectorizer is None:
        raise HTTPException(500, "Model not loaded")

    if not news.text.strip():
        raise HTTPException(400, "Empty input")

    processed = preprocess_text(news.text)

    vec = vectorizer.transform([processed])

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    confidence = float(prob[pred])
    emotion = get_emotional_intensity(news.text)
    verification = verify_with_trusted_sources(news.text[:100])

    return Prediction(
        prediction="Fake" if pred == 1 else "Real",
        confidence=confidence,
        is_fake=bool(pred == 1),
        emotional_intensity=emotion,
        verification_score=verification
    )


