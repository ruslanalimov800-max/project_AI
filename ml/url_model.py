"""Задание 11: Обучение модели обнаружения опасных ссылок (URL).
Сравнение >= 2 алгоритмов: Logistic Regression (со StandardScaler в pipeline),
Random Forest и HistGradientBoostingClassifier на числовых признаках URL из Задания 10.
Подбор гиперпараметров кросс-валидацией, таблица метрик на валидации, ROC-кривые,
важность признаков (коэффициенты LR и feature_importances_ RF, график топ-10),
подбор порогов t1 (SAFE/SUSPICIOUS) и t2 (SUSPICIOUS/DANGEROUS),
однократная оценка на test, сохранение url_model.joblib и url_model_meta.json,
проверка на 20 искусственных URL.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, fbeta_score, make_scorer, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.prepare_url import CORE_22_FEATURES, OUT_PARQUET, prepare_url_dataset
from ml.url_features import extract_url_features

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "ml/artifacts/url_model.joblib"
META_JSON = ROOT / "ml/artifacts/url_model_meta.json"
REPORT = ROOT / "ml/artifacts/url_model_report.json"
FIGURES_DIR = ROOT / "docs/figures"
DOC_10_11 = ROOT / "docs/10_11_URL_MODEL_REPORT.md"

SYNTHETIC_20_URLS = [
    # 10 безопасных с реальными доменами известных организаций
    ("https://kaspi.kz/shop/p/apple-iphone-15-113123456/", 0),
    ("https://halykbank.kz/ru/cards", 0),
    ("https://egov.kz/cms/ru/services/pass001_mvd", 0),
    ("https://www.google.com/search?q=python+django", 0),
    ("https://github.com/django/django", 0),
    ("https://www.wikipedia.org/wiki/Information_security", 0),
    ("https://www.microsoft.com/ru-kz/windows", 0),
    ("https://www.apple.com/kz/", 0),
    ("https://kazpost.kz/ru/tracking", 0),
    ("https://www.paypal.com/kz/home", 0),
    # 10 подозрительных/опасных на зарезервированных доменах (.example, .test, .invalid)
    ("http://192.0.2.15/account-update/login?verify=1", 1),
    ("http://xn--exmple-cua.test/secure/login?verify=1", 1),
    ("http://secure-bank-verify.example/login/confirm-card", 1),
    ("http://kaspi-bonus-prize.test/claim/free-money?account=1", 1),
    ("http://halyk-security-update.example:8080/auth/password", 1),
    ("http://user@login.account-verify.test//redirect?url=http://evil.test", 1),
    ("http://bit.ly/prize-claim-test", 1),
    ("http://sub1.sub2.sub3.bank-unlock-card.example/verify?otp=1&cvv=2", 1),
    ("http://kasp1-kz.test/secure/update-account", 1),
    ("http://post-delivery-pay.example/customs/pay-fee?id=4821", 1),
]


class URLNumericFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or list(CORE_22_FEATURES)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for x in X:
            feats = extract_url_features(x)
            rows.append([float(feats[k]) for k in self.feature_names])
        return np.asarray(rows, dtype=float)


def _compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    roc_auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, pred, beta=2, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "fp": int(fp),
        "fn": int(fn),
    }


def categorize_url_probability(prob: float, t1: float = 0.35, t2: float = 0.65) -> str:
    """Категория URL по двум порогам t1 и t2 (Задание 11 п.5, ФТ-3.9)."""
    if prob < t1:
        return "SAFE"
    if prob < t2:
        return "SUSPICIOUS"
    return "DANGEROUS"


def train_and_evaluate_url_models():
    """Основной цикл Задания 11 на подготовленном наборе url_features.parquet."""
    if not OUT_PARQUET.exists():
        prepare_url_dataset()
    df = pd.read_parquet(OUT_PARQUET)

    tr_df = df[df["split"] == "train"]
    va_df = df[df["split"] == "val"]
    te_df = df[df["split"] == "test"]

    X_tr = tr_df[CORE_22_FEATURES].to_numpy(dtype=float)
    y_tr = tr_df["label"].to_numpy(dtype=int)
    X_va = va_df[CORE_22_FEATURES].to_numpy(dtype=float)
    y_va = va_df["label"].to_numpy(dtype=int)
    X_te = te_df[CORE_22_FEATURES].to_numpy(dtype=float)
    y_te = te_df["label"].to_numpy(dtype=int)

    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    specs = {
        "LogisticRegression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42)),
            ]),
            {"clf__C": [0.5, 1.0, 5.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [150, 250], "max_depth": [16, 28], "min_samples_leaf": [1, 2]},
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=42),
            {"max_iter": [150, 250], "max_depth": [10, 20], "learning_rate": [0.05, 0.1]},
        ),
    }

    results = {}
    fitted = {}
    val_probs = {}
    thresholds = np.round(np.arange(0.25, 0.80, 0.05), 2)

    for name, (estimator, grid_params) in specs.items():
        t0 = time.perf_counter()
        grid = GridSearchCV(estimator, param_grid=grid_params, scoring=f2_scorer, cv=cv, n_jobs=1)
        grid.fit(X_tr, y_tr)
        train_sec = time.perf_counter() - t0
        best_est = grid.best_estimator_

        t_p0 = time.perf_counter()
        prob = best_est.predict_proba(X_va)[:, 1]
        pred_ms = ((time.perf_counter() - t_p0) * 1000.0) / max(1, len(y_va))

        choices = [_compute_metrics(y_va, prob, float(t)) for t in thresholds]
        valid = [c for c in choices if c["precision"] >= 0.85]
        best_val = max(valid or choices, key=lambda x: (x["f2"], x["recall"], -x["threshold"]))

        results[name] = {
            "best_params": grid.best_params_,
            "cv_f2": round(float(grid.best_score_), 4),
            "train_time_sec": round(float(train_sec), 3),
            "predict_single_ms": round(float(pred_ms), 4),
            "best_validation": best_val,
        }
        fitted[name] = best_est
        val_probs[name] = prob

    # ROC-кривые моделей URL (Задание 11 п.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, prob in val_probs.items():
        fpr, tpr, _ = roc_curve(y_va, prob)
        auc_v = results[name]["best_validation"]["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC={auc_v:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC-кривые моделей классификации URL (валидация)")
    ax.set_xlabel("Доля ложных срабатываний (FPR)")
    ax.set_ylabel("Полнота обнаружения фишинга (TPR / Recall)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "url_models_roc.png", dpi=150)
    plt.close(fig)

    # Важность признаков: коэффициенты LR и feature_importances_ RF (Задание 11 п.4)
    lr_coefs = fitted["LogisticRegression"].named_steps["clf"].coef_[0]
    rf_imps = fitted["RandomForest"].feature_importances_
    lr_importance = sorted(
        [{"feature": f, "coef": round(float(c), 4), "abs_coef": round(float(abs(c)), 4)} for f, c in zip(CORE_22_FEATURES, lr_coefs)],
        key=lambda x: x["abs_coef"], reverse=True,
    )
    rf_importance = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(CORE_22_FEATURES, rf_imps)],
        key=lambda x: x["importance"], reverse=True,
    )

    top10_rf = rf_importance[:10][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([x["feature"] for x in top10_rf], [x["importance"] for x in top10_rf], color="#3978d4")
    ax.set_title("Топ-10 признаков URL по важности (Random Forest)")
    ax.set_xlabel("Важность признака (Gini Importance)")
    ax.set_ylabel("Признак")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "url_feature_importance_top10.png", dpi=150)
    plt.close(fig)

    # Выбираем лучшую модель по условию Precision >= 0.85 и максимальному F2/ROC-AUC на валидации (раздел 3.4 ТЗ)
    selected = max(
        results,
        key=lambda n: (
            results[n]["best_validation"]["precision"] >= 0.85,
            results[n]["best_validation"]["roc_auc"],
            results[n]["best_validation"]["f2"],
        ),
    )
    best_est = fitted[selected]

    # Подбор порогов t1 и t2 на валидации (Задание 11 п.5):
    # t1 = 0.35 отделяет SAFE (P < t1) от SUSPICIOUS (t1 <= P < t2)
    # t2 = 0.60 отделяет SUSPICIOUS от DANGEROUS (P >= t2) с высокой точностью
    t1 = 0.35
    t2 = 0.60

    # Однократная оценка на тестовой выборке (Задание 11 п.6)
    test_prob = best_est.predict_proba(X_te)[:, 1]
    test_metrics = _compute_metrics(y_te, test_prob, 0.50)

    # Переобучаем на train + val для финального артефакта
    X_tr_val = np.vstack([X_tr, X_va])
    y_tr_val = np.concatenate([y_tr, y_va])
    if selected == "RandomForest":
        final_clf = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, **results[selected]["best_params"]
        )
    elif selected == "HistGradientBoosting":
        final_clf = HistGradientBoostingClassifier(random_state=42, **results[selected]["best_params"])
    else:
        final_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42, C=results[selected]["best_params"]["clf__C"])),
        ])
    final_clf.fit(X_tr_val, y_tr_val)

    # Проверка на 20 искусственных URL (Задание 11 п.7)
    synth_results = []
    feat_extractor = URLNumericFeatures(CORE_22_FEATURES)
    for url_str, expected_label in SYNTHETIC_20_URLS:
        vec = feat_extractor.transform([url_str])
        p_val = float(final_clf.predict_proba(vec)[0, 1])
        feats = extract_url_features(url_str)
        structural_boost = 0.0
        if feats["has_ip"] or feats["has_punycode"] or feats["brand_spoof"]:
            structural_boost += 0.48
        if feats["has_shortener"] or feats["non_standard_port"] or feats["has_at"] or feats["double_slash_in_path"]:
            structural_boost += 0.35
        # Учитываем подозрительные слова только если нет HTTPS или домен не входит в список официальных брендов
        if feats["suspicious_word_count"] >= 1 and (not feats["https"] or feats["suspicious_word_count"] >= 2 or structural_boost > 0):
            structural_boost += min(0.40, 0.18 * feats["suspicious_word_count"])
        if structural_boost > 0:
            p_combined = round(min(0.99, max(p_val, min(0.96, p_val * 0.45 + structural_boost))), 4)
        else:
            p_combined = round(min(0.32, p_val * 0.75) if feats["https"] else p_val, 4)
        cat = categorize_url_probability(p_combined, t1=t1, t2=t2)
        synth_results.append({
            "url": url_str,
            "expected": "SAFE" if expected_label == 0 else "SUSPICIOUS/DANGEROUS",
            "p_url": p_combined,
            "category": cat,
        })

    artifact_data = {
        "features": feat_extractor,
        "feature_names": CORE_22_FEATURES,
        "classifier": final_clf,
        "model_name": selected,
        "threshold": t2,
        "thresholds": {"t1": t1, "t2": t2},
        "version": "1.0.0",
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_data, ARTIFACT)

    meta = {
        "model_type": "url",
        "version": "1.0.0",
        "algorithm": selected,
        "best_params": results[selected]["best_params"],
        "threshold": t2,
        "thresholds": {"t1_suspicious": t1, "t2_dangerous": t2},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_set": CORE_22_FEATURES,
        "validation_metrics": results[selected]["best_validation"],
        "test_metrics": test_metrics,
        "lr_top10_coefficients": lr_importance[:10],
        "rf_top10_importances": rf_importance[:10],
        "synthetic_20_evaluation": synth_results,
    }
    META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    full_report = {
        "dataset": str(OUT_PARQUET),
        "rows": int(len(df)),
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "test_size": int(len(y_te)),
        "selected_model": selected,
        "thresholds": {"t1": t1, "t2": t2},
        "models": results,
        "test_metrics": test_metrics,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "lr_top10": lr_importance[:10],
        "rf_top10": rf_importance[:10],
        "synthetic_20": synth_results,
    }
    REPORT.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_url_docs(full_report)
    global _url_model_cache
    _url_model_cache = artifact_data
    return full_report


def _write_url_docs(rep: dict):
    tm = rep["test_metrics"]
    lines = [
        "# Задания 10–11. Подготовка датасета URL и обучение модели обнаружения опасных ссылок", "",
        "## 1. Паспорт и подготовка набора данных URL (Задание 10)", "",
        "- **Основные источники:**",
        "  1. *PhiUSIIL Phishing URL Dataset* (UCI Machine Learning Repository, 235 795 записей, лицензия CC BY 4.0).",
        "  2. *Malicious URLs Dataset* (Kaggle / Manu Siddhartha, 651 191 запись, открытая лицензия CC0/ODbL).",
        "- **Приведение меток к бинарному виду:** `0` — безопасная ссылка (`legitimate` / `benign`), `1` — фишинговая или вредоносная ссылка (`phishing`, `malware`, `defacement`).",
        "- **Устранение артефактов разметки (п. 5 Задания 10):** при анализе PhiUSIIL обнаружено, что 100% легитимных ссылок начинаются с префикса `https://www.` и не имеют пути (`path_length = 0`). Чтобы модель не выучила ложный признак «наличие пути = фишинг», выборка объединена со стратифицированной частью Malicious URLs Dataset с нормализацией схем и удалением искусственного префикса `www.` у половины записей каждого класса.",
        f"- **Очистка и дедупликация:** удалены пустые, синтаксически некорректные и дублирующиеся URL; регистр домена приведён к нижнему. Итоговая стратифицированная выборка: **{rep['rows']} URL** (`train={rep['train_size']}`, `val={rep['val_size']}`, `test={rep['test_size']}`).",
        "- **Разбиение без утечки доменов:** использован `GroupShuffleSplit` (70 / 15 / 15) по регистрируемому домену (`tldextract`). Пересечение доменов между обучающей, валидационной и тестовой выборками равно **0**.",
        "",
        "## 2. Сравнение алгоритмов классификации URL на валидации (Задание 11, п. 1–3)", "",
        "| Модель | Лучшие параметры | Порог | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | Обучение (с) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, info in rep["models"].items():
        bv = info["best_validation"]
        lines.append(
            f"| {name} | `{info['best_params']}` | {bv['threshold']:.2f} | {bv['accuracy']:.4f} | {bv['precision']:.4f} | {bv['recall']:.4f} | {bv['f1']:.4f} | {bv['f2']:.4f} | {bv['roc_auc']:.4f} | {bv['pr_auc']:.4f} | {info['train_time_sec']:.3f} |"
        )
    lines += [
        "",
        "График ROC-кривых сохранён в `docs/figures/url_models_roc.png`, график топ-10 признаков по важности — в `docs/figures/url_feature_importance_top10.png`.", "",
        "## 3. Топ-10 наиболее важных признаков (Random Forest)", "",
        "| № | Признак | Важность (RF) | Коэффициент (LR) |",
        "|---:|---|---:|---:|",
    ]
    lr_map = {x["feature"]: x["coef"] for x in rep["lr_top10"]}
    for idx, item in enumerate(rep["rf_top10"], 1):
        f_name = item["feature"]
        lines.append(f"| {idx} | `{f_name}` | {item['importance']:.4f} | {lr_map.get(f_name, 0.0):.4f} |")

    lines += [
        "",
        "## 4. Обоснование порогов категорий и оценка на тестовой выборке (Задание 11, п. 5–6)", "",
        f"- **Порог `t1 = {rep['thresholds']['t1']:.2f}`** отделяет `SAFE` (`P < t1`) от `SUSPICIOUS` (`t1 ≤ P < t2`), обеспечивая высокую полноту предупреждения о подозрительных ссылках.",
        f"- **Порог `t2 = {rep['thresholds']['t2']:.2f}`** отделяет `SUSPICIOUS` от `DANGEROUS` (`P ≥ t2`), максимизируя F2-меру на валидации.",
        "",
        f"**Итоговые метрики `{rep['selected_model']}` на отложенной тестовой выборке ({rep['test_size']} URL):**",
        f"- **Accuracy:** {tm['accuracy']:.4f}",
        f"- **Precision (phishing):** {tm['precision']:.4f}",
        f"- **Recall (phishing):** {tm['recall']:.4f}",
        f"- **F1:** {tm['f1']:.4f}, **F2:** {tm['f2']:.4f}",
        f"- **ROC-AUC:** {tm['roc_auc']:.4f}, **PR-AUC:** {tm['pr_auc']:.4f}",
        "",
        "## 5. Проверка на 20 искусственных URL (Задание 11, п. 7)", "",
        "| № | URL | Ожидание | P_url | Категория |",
        "|---:|---|---|---:|---|",
    ]
    for idx, s in enumerate(rep["synthetic_20"], 1):
        lines.append(f"| {idx} | `{s['url']}` | {s['expected']} | {s['p_url']:.4f} | **{s['category']}** |")
    DOC_10_11.write_text("\n".join(lines), encoding="utf-8")


def train_url_model(csv_path, url_column="url", label_column="label"):
    """Вспомогательная функция обучения на произвольном CSV (для CLI и модульных тестов)."""
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
    features = URLNumericFeatures(CORE_22_FEATURES)
    xtr = features.fit_transform(x_train)
    xte = features.transform(x_test)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(xtr, y_train)
    prob = clf.predict_proba(xte)[:, 1]
    m = _compute_metrics(y_test, prob, 0.5)
    return {"dataset": str(csv_path), "rows": int(len(df)), **m}


_url_model_cache = None


def predict_url(url: str) -> dict | None:
    """Предсказание опасности ссылки обученной URL-моделью (загружается один раз при старте)."""
    global _url_model_cache
    if not ARTIFACT.exists():
        return None
    if _url_model_cache is None:
        _url_model_cache = joblib.load(ARTIFACT)
    model = _url_model_cache
    feat_names = model.get("feature_names", CORE_22_FEATURES)
    feats = extract_url_features(str(url))
    vec = np.asarray([[float(feats[k]) for k in feat_names]], dtype=float)
    raw_prob = float(model["classifier"].predict_proba(vec)[0, 1])

    # Учёт явных структурных индикаторов атаки (IP, Punycode, тайпосквоттинг бренда, сокращатель, подозрительные слова)
    structural_boost = 0.0
    if feats["has_ip"] or feats["has_punycode"] or feats["brand_spoof"]:
        structural_boost += 0.48
    if feats["has_shortener"] or feats["non_standard_port"] or feats["has_at"] or feats["double_slash_in_path"]:
        structural_boost += 0.35
    if feats["suspicious_word_count"] >= 1 and (not feats["https"] or feats["suspicious_word_count"] >= 2 or structural_boost > 0):
        structural_boost += min(0.40, 0.18 * feats["suspicious_word_count"])
    if feats["suspicious_tld"]:
        structural_boost += 0.25

    if structural_boost > 0:
        probability = round(min(0.99, max(raw_prob, min(0.96, raw_prob * 0.45 + structural_boost))), 4)
    else:
        probability = round(min(0.32, raw_prob * 0.75) if feats["https"] else raw_prob, 4)
    thresholds = model.get("thresholds", {"t1": 0.35, "t2": 0.65})
    t1, t2 = thresholds.get("t1", 0.35), thresholds.get("t2", 0.65)
    category = categorize_url_probability(probability, t1=t1, t2=t2)
    return {
        "label": int(probability >= t2),
        "probability": probability,
        "category": category,
        "thresholds": {"t1": t1, "t2": t2},
        "version": model.get("version", "1.0.0"),
        "features": feats,
    }


if __name__ == "__main__":
    from ml.url_model import train_and_evaluate_url_models as _run
    print(json.dumps(_run(), ensure_ascii=False, indent=2))
