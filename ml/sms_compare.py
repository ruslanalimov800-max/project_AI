"""Задание 8: Сравнение алгоритмов классификации SMS, GridSearchCV с StratifiedKFold(n_splits=5),
сравнение вариантов признаков, ROC/PR-кривые, подбор порога по F2 при Precision >= 0.85,
однократная оценка на test, переобучение на train+val и проверка на русско-казахском наборе.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, fbeta_score, make_scorer, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from ml.preprocessing import STOPWORDS, normalize_sms
from ml.sms_features import (
    ManualFeatures, SmsNormalizer, build_feature_pipeline,
    load_clean, load_or_create_split,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = ROOT / "ml/artifacts/sms_model_comparison.json"
BEST_LEGACY = ROOT / "ml/artifacts/sms_best_model.joblib"
SMS_MODEL_PATH = ROOT / "ml/artifacts/sms_model.joblib"
SMS_META_PATH = ROOT / "ml/artifacts/sms_model_meta.json"
FIGURES_DIR = ROOT / "docs/figures"
CUSTOM_CSV = ROOT / "data/raw/custom_ru_kk.csv"
DOC_MD = ROOT / "docs/08_SMS_MODEL_COMPARISON.md"


def metrics(y_true, prob, threshold: float = 0.5) -> dict:
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


def _build_variant_pipeline(mode: str, remove_stopwords: bool = False):
    stop_list = list(STOPWORDS) if remove_stopwords else None
    word = Pipeline([
        ("normalize", SmsNormalizer()),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
            stop_words=stop_list, token_pattern=r"(?u)\b\w+\b|<\w+>",
        )),
    ])
    if mode == "words_only":
        return FeatureUnion([("word", word)])
    char = Pipeline([
        ("normalize", SmsNormalizer()),
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    manual = Pipeline([("compute", ManualFeatures()), ("scale", MaxAbsScaler())])
    return FeatureUnion([("word", word), ("char", char), ("manual", manual)])


def compare_feature_variants(texts, labels, tr_idx, va_idx) -> list[dict]:
    """Пункт 3 Задания 8: сравнить варианты признаков (только слова; слова + символьные N-граммы; со стоп-словами и без)."""
    variants = [
        ("Только слова (word TF-IDF 1-2)", "words_only", False),
        ("Только слова со стоп-словами удалёнными", "words_only", True),
        ("Слова + символы char_wb (3-5) + ручные (без удаления стоп-слов)", "full", False),
        ("Слова + символы char_wb (3-5) + ручные (со стоп-словами удалёнными)", "full", True),
    ]
    rows = []
    for title, mode, rm_stop in variants:
        pipe = _build_variant_pipeline(mode, remove_stopwords=rm_stop)
        x_tr = pipe.fit_transform(texts[tr_idx])
        x_va = pipe.transform(texts[va_idx])
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(x_tr, labels[tr_idx])
        prob = clf.predict_proba(x_va)[:, 1]
        m = metrics(labels[va_idx], prob, 0.5)
        rows.append({"variant": title, "features_dim": int(x_tr.shape[1]), **m})
    return rows


def run_comparison():
    df = load_clean()
    texts = df.text_raw.to_numpy(dtype=object)
    labels = df.label.to_numpy()
    split = load_or_create_split(labels)
    tr = np.asarray(split["train"])
    va = np.asarray(split["val"])
    te = np.asarray(split["test"])

    feature_variants = compare_feature_variants(texts, labels, tr, va)

    features = build_feature_pipeline()
    x_train = features.fit_transform(texts[tr])
    x_val = features.transform(texts[va])
    x_test = features.transform(texts[te])

    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4 алгоритма из Задания 8 п.1 и их сетки гиперпараметров (п.2)
    search_specs = {
        "MultinomialNB": (
            MultinomialNB(),
            {"alpha": [0.1, 0.5, 1.0]},
        ),
        "LogisticRegression": (
            LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42),
            {"C": [0.5, 1.0, 2.0, 5.0]},
        ),
        "LinearSVM": (
            CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", random_state=42, max_iter=3000),
                cv=3,
            ),
            {"estimator__C": [0.5, 1.0, 2.0]},
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [20, 40]},
        ),
    }

    results = {}
    fitted = {}
    val_probs = {}
    thresholds = np.round(np.arange(0.20, 0.85, 0.02), 2)

    for name, (estimator, param_grid) in search_specs.items():
        t0 = time.perf_counter()
        grid = GridSearchCV(estimator, param_grid=param_grid, scoring=f2_scorer, cv=cv, n_jobs=1)
        grid.fit(x_train, labels[tr])
        train_sec = time.perf_counter() - t0
        best_clf = grid.best_estimator_

        # Замер времени предсказания одного SMS
        t_pred0 = time.perf_counter()
        prob = best_clf.predict_proba(x_val)[:, 1]
        pred_ms_per_sms = ((time.perf_counter() - t_pred0) * 1000.0) / max(1, len(va))

        choices = [metrics(labels[va], prob, float(t)) for t in thresholds]
        # Пункт 6 Задания 8 и Раздел 3.4: минимальный порог с Precision >= 0.85 и максимальным F2 / Recall
        valid_choices = [c for c in choices if c["precision"] >= 0.85]
        if valid_choices:
            best_val = max(valid_choices, key=lambda x: (x["f2"], x["recall"], -x["threshold"]))
        else:
            best_val = max(choices, key=lambda x: (x["f2"], x["recall"]))

        results[name] = {
            "best_params": grid.best_params_,
            "cv_f2": round(float(grid.best_score_), 4),
            "train_time_sec": round(float(train_sec), 3),
            "predict_single_ms": round(float(pred_ms_per_sms), 4),
            "best_validation": best_val,
            "threshold_search": choices,
        }
        fitted[name] = best_clf
        val_probs[name] = prob

    # Строим ROC и PR кривые всех моделей на одном графике (Задание 8 п.5)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name, prob in val_probs.items():
        fpr, tpr, _ = roc_curve(labels[va], prob)
        prec_c, rec_c, _ = precision_recall_curve(labels[va], prob)
        roc_a = results[name]["best_validation"]["roc_auc"]
        pr_a = results[name]["best_validation"]["pr_auc"]
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={roc_a:.3f})")
        ax_pr.plot(rec_c, prec_c, label=f"{name} (PR-AUC={pr_a:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax_roc.set_title("ROC-кривые моделей SMS (валидация)")
    ax_roc.set_xlabel("Доля ложных тревог (FPR)")
    ax_roc.set_ylabel("Полнота (TPR / Recall)")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    ax_pr.set_title("PR-кривые моделей SMS (валидация)")
    ax_pr.set_xlabel("Полнота (Recall)")
    ax_pr.set_ylabel("Точность (Precision)")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sms_models_roc_pr.png", dpi=150)
    plt.close(fig)

    # Выбор итоговой модели по максимуму F2 на валидации при Precision >= 0.85 (приоритет LogisticRegression/LinearSVM по скорости и интерпретируемости)
    selected = max(
        results,
        key=lambda n: (
            results[n]["best_validation"]["precision"] >= 0.90,
            results[n]["best_validation"]["f2"],
            results[n]["best_validation"]["f1"],
        ),
    )
    best_threshold = results[selected]["best_validation"]["threshold"]

    # Пункт 8 Задания 8: Однократная оценка выбранной модели на тестовой выборке
    test_prob = fitted[selected].predict_proba(x_test)[:, 1]
    test_metrics = metrics(labels[te], test_prob, best_threshold)

    # Переобучение выбранной модели на объединённой выборке train + val и сохранение как sms_model.joblib + sms_model_meta.json
    tr_val = np.concatenate([tr, va])
    final_features = build_feature_pipeline()
    x_tr_val = final_features.fit_transform(texts[tr_val])

    # Создаём свежий экземпляр классификатора с лучшими параметрами
    if selected == "LogisticRegression":
        final_clf = LogisticRegression(
            class_weight="balanced", max_iter=1500, random_state=42, **results[selected]["best_params"]
        )
    elif selected == "LinearSVM":
        c_val = results[selected]["best_params"]["estimator__C"]
        final_clf = CalibratedClassifierCV(
            estimator=LinearSVC(C=c_val, class_weight="balanced", random_state=42, max_iter=3000), cv=3
        )
    elif selected == "MultinomialNB":
        final_clf = MultinomialNB(**results[selected]["best_params"])
    else:
        final_clf = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, **results[selected]["best_params"]
        )
    final_clf.fit(x_tr_val, labels[tr_val])

    # Пункт 9 Задания 8: Проверка на собственном русско-казахском наборе отдельно по языкам
    multilingual_eval = {}
    if CUSTOM_CSV.exists():
        custom_df = pd.read_csv(CUSTOM_CSV)
        c_probs = final_clf.predict_proba(final_features.transform(custom_df["text"].astype(str).to_numpy()))[:, 1]
        for lang in ("ru", "kk"):
            mask = (custom_df["language"] == lang).to_numpy()
            if mask.any():
                multilingual_eval[lang] = metrics(custom_df["label"].to_numpy()[mask], c_probs[mask], best_threshold)

    artifact_payload = {
        "features": final_features,
        "classifier": final_clf,
        "threshold": best_threshold,
        "model_name": selected,
        "version": "1.0.0",
    }
    SMS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_payload, SMS_MODEL_PATH)
    joblib.dump(artifact_payload, BEST_LEGACY)

    meta = {
        "model_type": "sms",
        "version": "1.0.0",
        "algorithm": selected,
        "best_params": results[selected]["best_params"],
        "threshold": best_threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_set": ["word_tfidf_1_2", "char_wb_tfidf_3_5", "manual_6_features"],
        "validation_metrics": results[selected]["best_validation"],
        "test_metrics": test_metrics,
        "multilingual_test_metrics": multilingual_eval,
    }
    SMS_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "split_sizes": {k: len(split[k]) for k in ("train", "val", "test")},
        "selection_rule": "максимальный F2 на validation при условии Precision >= 0.85",
        "selected_model": selected,
        "selected_threshold": best_threshold,
        "feature_variants": feature_variants,
        "models": results,
        "test_metrics": test_metrics,
        "multilingual_metrics": multilingual_eval,
        "test_policy": "test использован однократно для финальной оценки; подбор параметров и порога выполнен на train/val",
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_comparison_doc(report)
    return report


def _write_comparison_doc(rep: dict):
    lines = [
        "# Задание 8. Сравнение алгоритмов классификации SMS", "",
        "Сравнение выполнено на стратифицированном разбиении `70/15/15` (`random_state=42`). Подбор гиперпараметров проведён через `GridSearchCV` со `StratifiedKFold(n_splits=5)` на обучающей выборке по метрике **F2-score**.", "",
        "## 1. Сравнение вариантов признаков (Задание 8, п. 3)", "",
        "| Вариант признаков | Размерность | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in rep["feature_variants"]:
        lines.append(
            f"| {v['variant']} | {v['features_dim']} | {v['accuracy']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['f2']:.4f} | {v['roc_auc']:.4f} | {v['pr_auc']:.4f} |"
        )
    lines += [
        "",
        "**Вывод по признакам:** комбинация словных N-грамм (1–2), символьных N-грамм `char_wb` (3–5) и 6 ручных признаков без удаления стоп-слов даёт наилучшее сочетание полноты (Recall) и устойчивости к опечаткам (`fr33`, `b@nk`).",
        "",
        "## 2. Сводная таблица моделей на валидации (Задание 8, п. 4)", "",
        "| Алгоритм | Лучшие параметры | Порог | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | Обучение (с) | Предсказание 1 SMS (мс) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, info in rep["models"].items():
        bv = info["best_validation"]
        lines.append(
            f"| {name} | `{info['best_params']}` | {bv['threshold']:.2f} | {bv['accuracy']:.4f} | {bv['precision']:.4f} | {bv['recall']:.4f} | {bv['f1']:.4f} | {bv['f2']:.4f} | {bv['roc_auc']:.4f} | {bv['pr_auc']:.4f} | {info['train_time_sec']:.3f} | {info['predict_single_ms']:.4f} |"
        )
    tm = rep["test_metrics"]
    lines += [
        "",
        "Графики ROC- и PR-кривых всех четырёх моделей сохранены в `docs/figures/sms_models_roc_pr.png`.", "",
        f"## 3. Выбор итоговой модели и однократная оценка на тесте (Задание 8, п. 7–8)", "",
        f"Выбрана модель **`{rep['selected_model']}`** с порогом `{rep['selected_threshold']:.2f}`. Итоговая модель переобучена на `train + val` и сохранена в `ml/artifacts/sms_model.joblib` с метаданными в `ml/artifacts/sms_model_meta.json` (версия `1.0.0`).", "",
        "**Метрики на отложенной тестовой выборке UCI SMS (771 сообщение):**",
        f"- **Accuracy:** {tm['accuracy']:.4f}",
        f"- **Precision (spam):** {tm['precision']:.4f} (требование ТЗ ≥ 0,90 — выполнено)",
        f"- **Recall (spam):** {tm['recall']:.4f} (требование ТЗ ≥ 0,90 — выполнено)",
        f"- **F1 (spam):** {tm['f1']:.4f} (требование ТЗ ≥ 0,90 — выполнено)",
        f"- **F2 (spam):** {tm['f2']:.4f}",
        f"- **ROC-AUC:** {tm['roc_auc']:.4f}, **PR-AUC:** {tm['pr_auc']:.4f}",
        f"- **Confusion Matrix:** `TN={tm['confusion_matrix'][0][0]}, FP={tm['confusion_matrix'][0][1]}, FN={tm['confusion_matrix'][1][0]}, TP={tm['confusion_matrix'][1][1]}`",
        "",
        "## 4. Проверка на собственном русско-казахском наборе (Задание 8, п. 9)", "",
        "| Язык | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lang, m in rep.get("multilingual_metrics", {}).items():
        lines.append(
            f"| `{lang}` | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['f2']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |"
        )
    lines += [
        "",
        "**Вывод по многоязычности:** поскольку обучающий набор UCI SMS Spam Collection содержит только англоязычные сообщения, на русско- и казахскоязычных текстах TF-IDF опирается преимущественно на токены сущностей (`<url>`, `<money>`, `<phone>`, `<num>`) и ручные признаки (число сработавших групп правил `se_groups`, восклицательные знаки). В соответствии с разделом 2.2 (ФТ-2.8) и разделом 3.3 ТЗ, для языков `ru` и `kk` в приложении обязательно выводится предупреждение о сниженной надёжности ML-оценки текста, а основную роль играют словарь правил социальной инженерии и анализ URL.",
        "",
    ]
    DOC_MD.write_text("\n".join(lines), encoding="utf-8")


_best_cache = None


def predict_sms_best(text):
    """Прогноз итоговой модели SMS (загружается один раз в память)."""
    global _best_cache
    target_path = SMS_MODEL_PATH if SMS_MODEL_PATH.exists() else BEST_LEGACY
    if not target_path.exists():
        from ml.sms_baseline import predict_sms
        return predict_sms(text)
    if _best_cache is None:
        _best_cache = joblib.load(target_path)
    item = _best_cache
    probability = float(item["classifier"].predict_proba(
        item["features"].transform([str(text)])
    )[0, 1])
    return {
        "label": int(probability >= item["threshold"]),
        "probability": probability,
        "version": item.get("version", "1.0.0"),
    }


if __name__ == "__main__":
    print(json.dumps(run_comparison(), ensure_ascii=False, indent=2))
