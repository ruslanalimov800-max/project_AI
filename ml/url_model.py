"""URL-классификатор. Требует обучающий CSV; искусственные метрики не генерируются."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.base import BaseEstimator, TransformerMixin
from ml.url_features import extract_url_features

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "ml/artifacts/url_model.joblib"
REPORT = ROOT / "ml/artifacts/url_model_report.json"

class URLNumericFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows = [list(extract_url_features(x).values()) for x in X]
        return csr_matrix(np.asarray(rows, dtype=float))

def build_url_pipeline():
    return FeatureUnion([
        ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ("numeric", URLNumericFeatures()),
    ])

def train_url_model(csv_path, url_column="url", label_column="label"):
    """label: 1=phishing, 0=legitimate. Деление train/test стратифицированное."""
    df = pd.read_csv(csv_path)
    if not {url_column, label_column}.issubset(df.columns):
        raise ValueError(f"CSV должен содержать столбцы {url_column!r} и {label_column!r}")
    df = df[[url_column, label_column]].dropna()
    X = df[url_column].astype(str).to_numpy()
    y = pd.to_numeric(df[label_column], errors="raise").astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Метки должны быть 0 (легитимный) и 1 (фишинг)")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    features = build_url_pipeline()
    xtr = features.fit_transform(x_train)
    xte = features.transform(x_test)
    clf = LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42)
    clf.fit(xtr, y_train)
    prob = clf.predict_proba(xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    report = {
        "dataset": str(csv_path), "rows": int(len(df)), "train_size": int(len(y_train)),
        "test_size": int(len(y_test)), "label_definition": {"0": "legitimate", "1": "phishing"},
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": features, "classifier": clf}, ARTIFACT)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

def predict_url(url):
    if not ARTIFACT.exists():
        return None
    model = joblib.load(ARTIFACT)
    vector = model["features"].transform([str(url)])
    probability = float(model["classifier"].predict_proba(vector)[0, 1])
    return {"label": int(probability >= 0.5), "probability": probability}
