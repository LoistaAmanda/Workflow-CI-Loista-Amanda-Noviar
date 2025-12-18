import mlflow
import mlflow.sklearn
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PATH
# =========================
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "dataset_preprocessing" / "imbd_preprocessed.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# MODEL
# =========================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# =========================
# TRAIN
# =========================
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# =========================
# METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", acc)

# =========================
# SAVE & LOG MODEL
# =========================
mlflow.sklearn.log_model(pipeline, artifact_path="model")

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Training Confusion Matrix")

cm_path = "training_confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

mlflow.log_artifact(cm_path)

print("Training & logging selesai")
