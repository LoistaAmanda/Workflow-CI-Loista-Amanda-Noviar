from pathlib import Path
import pandas as pd

# Machine Learning
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


# ============================================
# SETUP MLFLOW 
# ============================================

BASE_DIR = Path(__file__).resolve().parent
MLRUNS_DIR = BASE_DIR / "mlruns"

mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("IMDB_Modelling_Experiment")


# ============================================
# LOAD DATASET
# ============================================

DATA_PATH = BASE_DIR / "dataset_preprocessing" / "imbd_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

X = df["clean_text"]
y = df["label"]


# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================
# PIPELINE MODEL
# ============================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])


# ============================================
# TRAIN MODEL 
# ============================================

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# ============================================
# HITUNG METRIK
# ============================================

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")


# ============================================
# LOG METRIK 
# ============================================

mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision_macro", precision_macro)
mlflow.log_metric("recall_macro", recall_macro)
mlflow.log_metric("f1_macro", f1_macro)


# ============================================
# LOG MODEL 
# ============================================

mlflow.sklearn.log_model(
    pipeline,
    artifact_path="model"
)


# ============================================
# CONFUSION MATRIX 
# ============================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["low", "medium", "high"],
    yticklabels=["low", "medium", "high"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

artifact_dir = BASE_DIR / "artifacts"
artifact_dir.mkdir(exist_ok=True)

cm_path = artifact_dir / "training_confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

mlflow.log_artifact(str(cm_path))


print("Training & logging selesai")
