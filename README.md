# 🧠 Fake News Detection

A system built to distinguish truth from manipulation — using machine learning and structured text analysis.

---

## 🚀 Overview

This project analyzes textual data to classify whether a piece of news is **real or fake**.

It combines:

* Text preprocessing
* Feature extraction
* Machine learning models
* Dataset-driven evaluation

---

## 📂 Project Structure

```
.
├── backend/              # Python backend (model, API, logic)
├── data/                 # (ignored) datasets
├── liar/                 # dataset subset (if used)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create virtual environment

```
python -m venv venv
```

Activate it:

* Windows:

```
venv\Scripts\activate
```

* Mac/Linux:

```
source venv/bin/activate
```

---

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 📊 Dataset

The datasets used for training are **not included** in this repository due to their size and licensing restrictions.

Please download them from the following sources:

- **ISOT Fake News Dataset**  
  https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

- **FakeNewsNet (Political News Dataset)**  
  https://www.kaggle.com/datasets/gevaran/political-news-dataset

- **LIAR Dataset**  
  https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset/data

After downloading, place the dataset files in the appropriate directory as described in the project setup instructions.

```
/data
```

---

## 🧪 Running the Project

Example:

```
python backend/server.py
```

(Modify this based on your actual entry point)

---

## 🧠 How It Works

1. Text is cleaned and normalized
2. Features are extracted (e.g., TF-IDF or embeddings)
3. Model predicts probability of fake vs real
4. Output is returned via backend/API

---

## ⚠️ Notes

* Large datasets are ignored via `.gitignore`
* Environment variables are stored in `.env` (not tracked)
* Ensure correct dataset placement before running

---

## 📌 Future Improvements

* Deep learning models (LSTM / Transformers)
* Real-time news API integration
* Web interface for predictions
* Model explainability (why something is fake)

---

## 🧾 License

This project is for educational and experimental use.

---

## 🧭 Final Thought

Truth is rarely obvious.
This project doesn’t claim certainty — it builds probability.

Use it as a tool, not a verdict.
