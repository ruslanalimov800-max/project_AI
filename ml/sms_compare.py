"""Задание 8: сравнение трёх классификаторов и подбор порога на validation."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ml.sms_features import load_clean, load_or_create_split, build_feature_pipeline

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "ml/artifacts/sms_model_comparison.json"
BEST = ROOT / "ml/artifacts/sms_best_model.joblib"

def metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "fp": int(fp), "fn": int(fn)
    }

def run_comparison():
    df = load_clean()
    texts, labels = df.text_raw.to_numpy(dtype=object), df.label.to_numpy()
    split = load_or_create_split(labels)
    tr, va = np.asarray(split["train"]), np.asarray(split["val"])
    features = build_feature_pipeline()
    x_train = features.fit_transform(texts[tr])
    x_val = features.transform(texts[va])
    candidates = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42),
        "MultinomialNB": MultinomialNB(alpha=0.5),
        "ComplementNB": ComplementNB(alpha=0.5),
    }
    results, fitted = {}, {}
    thresholds = np.arange(0.20, 0.81, 0.02)
    for name, clf in candidates.items():
        clf.fit(x_train, labels[tr])
        prob = clf.predict_proba(x_val)[:, 1]
        choices = [metrics(labels[va], prob, t) for t in thresholds]
        # При равном F1 выбираем больший recall, затем меньший порог.
        best = max(choices, key=lambda x: (x["f1"], x["recall"], -x["threshold"]))
        results[name] = {"best_validation": best, "threshold_search": choices}
        fitted[name] = clf
    selected = max(results, key=lambda n: (
        results[n]["best_validation"]["f1"],
        results[n]["best_validation"]["recall"]
    ))
    best_threshold = results[selected]["best_validation"]["threshold"]
    joblib.dump({
        "features": features, "classifier": fitted[selected],
        "threshold": best_threshold, "model_name": selected
    }, BEST)
    report = {
        "split_sizes": {k: len(split[k]) for k in ("train", "val", "test")},
        "selection_rule": "максимальный F1 на validation; tie-break: recall",
        "selected_model": selected,
        "models": results,
        "test_policy": "test не использовался для подбора модели или порога"
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    print(json.dumps(run_comparison(), ensure_ascii=False, indent=2))


_best_cache = None
def predict_sms_best(text):
    """Прогноз выбранной по validation модели задания 8."""
    global _best_cache
    if not BEST.exists():
        from ml.sms_baseline import predict_sms
        return predict_sms(text)
    if _best_cache is None:
        _best_cache = joblib.load(BEST)
    item = _best_cache
    probability = float(item["classifier"].predict_proba(
        item["features"].transform([str(text)])
    )[0, 1])
    return {"label": int(probability >= item["threshold"]), "probability": probability}
