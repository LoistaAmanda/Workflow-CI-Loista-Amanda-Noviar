import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ======================
# AUTOLOG (WAJIB)
# ======================
mlflow.autolog()

# ======================
# LOAD DATA
# ======================
df = pd.read_csv("dataset_preprocessing/imbd_preprocessed.csv")

X = df["clean_text"]
y = df["label"]

# ======================
# SPLIT DATA
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# MODEL
# ======================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
