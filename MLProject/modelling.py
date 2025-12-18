from pathlib import Path
import pandas as pd

# =========================
# Machine Learning
# =========================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Evaluasi
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
import mlflow.sklearn


# ============================================================
# KONFIGURASI PATH (AMAN DI LOKAL & GITHUB ACTIONS)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_preprocessing" / "imbd_preprocessed.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MLRUNS_DIR = BASE_DIR / "mlruns"

ARTIFACT_DIR.mkdir(exist_ok=True)
MLRUNS_DIR.mkdir(exist_ok=True)

# ============================================================
# KONFIGURASI MLFLOW (WAJIB UNTUK CI)
# ============================================================

mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("IMDB_Modelling_Experiment")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df["clean_text"]
y = df["label"]

# ============================================================
# SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================================
# PIPELINE MODEL (TEXT → TF-IDF → RANDOM FOREST)
# ============================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])

# ============================================================
# TRAINING MODEL
# (JANGAN PAKE mlflow.start_run() KARENA MLflow PROJECT)
# ============================================================

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# ============================================================
# HITUNG METRIK
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred
