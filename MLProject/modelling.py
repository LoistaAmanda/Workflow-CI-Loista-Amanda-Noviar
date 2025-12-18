import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MLRUNS_DIR = BASE_DIR / "mlruns"

mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("IMDB_Modelling_Experiment")


# Load data
df = pd.read_csv("dataset_preprocessing/imbd_preprocessed.csv")

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

with mlflow.start_run():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(pipeline, artifact_path="model")

print("Training CI selesai")
