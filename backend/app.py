# app.py

import os
import pandas as pd
import nltk
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion


# Download stopwords (first time only)
nltk.download('stopwords', quiet=True)


# Paths
FAKE_PATH = "data/Fake.csv"
REAL_PATH = "data/True.csv"

LIAR_PATH1 = "data/liar/train.tsv"
LIAR_PATH2 = "data/liar/test.tsv"   
LIAR_PATH3 = "data/liar/valid.tsv" 

MODEL_PATH = "model/logreg.pkl"
VECTORIZER_PATH = "model/tfidf.pkl"


# Load resources once
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# Preprocessing
def preprocess_text(text):

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    words = [STEMMER.stem(w) for w in words]

    return " ".join(words)

# Load LIAR dataset
def load_liar_dataset(folder):

    files = ["train.tsv","test.tsv","valid.tsv"]

    dfs = []

    for f in files:
        df = pd.read_csv(f"{folder}/{f}", sep="\t", header=None)

        df = df[[1,2]]
        df.columns = ["label","text"]

        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    fake_labels = ["false","pants-fire","barely-true","half-true"]

    data["label"] = data["label"].apply(
        lambda x: 1 if x in fake_labels else 0
    )

    return data[["text","label"]]

# Load ISOT dataset
def load_isot_dataset(fake_path, real_path):

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df["label"] = 1
    real_df["label"] = 0

    data = pd.concat([fake_df, real_df], ignore_index=True)

    return data[["text","label"]].dropna()

# Load FakeNewsNet dataset
def load_fakenewsnet_dataset(folder):

    pf_real = pd.read_csv(f"{folder}/politifact_real.csv")
    pf_fake = pd.read_csv(f"{folder}/politifact_fake.csv")

    gc_real = pd.read_csv(f"{folder}/gossipcop_real.csv")
    gc_fake = pd.read_csv(f"{folder}/gossipcop_fake.csv")

    pf_real["label"] = 0
    gc_real["label"] = 0

    pf_fake["label"] = 1
    gc_fake["label"] = 1

    data = pd.concat(
        [pf_real, pf_fake, gc_real, gc_fake],
        ignore_index=True
    )

    data = data[["title", "label"]]
    data.rename(columns={"title":"text"}, inplace=True)

    return data.dropna()



# Train model
def train():

    print("Loading dataset...")
    isot_data = load_isot_dataset(FAKE_PATH, REAL_PATH)

    liar_data = load_liar_dataset("data/liar")

    fakenewsnet_data = load_fakenewsnet_dataset("data")
    print("ISOT:", len(isot_data))
    print("LIAR:", len(liar_data))
    print("FakeNewsNet:", len(fakenewsnet_data))
    data = pd.concat(
        [isot_data, liar_data, fakenewsnet_data],
        ignore_index=True
    )

    data = data.sample(frac=1, random_state=42)

    print("Preprocessing text...")

    data["processed"] = data["text"].apply(preprocess_text)

    X = data["processed"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print("Vectorizing text...")

    word_vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,3),
        min_df=2,
        sublinear_tf=True
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3,5),
        max_features=5000
    )

    vectorizer = FeatureUnion([
        ("word", word_vectorizer),
        ("char", char_vectorizer)
    ])

    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    print("Training model...")

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=4.0,
        solver="liblinear"
    )

    model.fit(X_train_tf, y_train)

    print("Evaluating...")

    y_pred = model.predict(X_test_tf)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

    # Save model
    os.makedirs("model", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Model saved successfully.")

    test = "The transportation department confirmed that construction will begin next month."

    processed = preprocess_text(test)
    vec = vectorizer.transform([processed])

    print("Prediction:", model.predict(vec))
    print("Probabilities:", model.predict_proba(vec))
    print(model.classes_)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train()
    
