"""Задание 7: baseline-модель SMS (TF-IDF + ручные признаки → Logistic Regression).

Цикл «обучение → оценка на валидации → сохранение → проверка загрузки». Тестовая выборка здесь не используется.

Запуск: ``python -m ml.sms_baseline``. Классы преобразователей определены в ``ml.sms_features``, поэтому
сохранённый joblib не ссылается на ``__main__`` и загружается в новом процессе Python.
"""
import json
import subprocess
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from ml.data.load_sms import ROOT
from ml.sms_features import (ManualFeatures, SmsNormalizer, build_feature_pipeline,  # noqa: F401 (реэкспорт)
                             load_clean, load_or_create_split, split_indices)

ARTIFACT = ROOT / 'ml/artifacts/sms_baseline.joblib'
REPORT = ROOT / 'ml/artifacts/sms_baseline_report.json'
THRESHOLD = 0.5


def build_baseline_pipeline():
    """Единый Pipeline: признаки (fit только на train) + Logistic Regression."""
    return Pipeline([
        ('features', build_feature_pipeline()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)),
    ])


def _metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),  # [[TN, FP], [FN, TP]]
    }


def _fresh_interpreter_check():
    """Загружает артефакт в НОВОМ процессе Python и возвращает вероятность для контрольного текста."""
    code = ("import json; from ml.sms_baseline import predict_sms; "
            "print(json.dumps(predict_sms('Ok, see you at lunch tomorrow')))")
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return done.returncode == 0, (done.stdout.strip() or done.stderr.strip()[-300:])


def train_baseline(save=True):
    """Обучает baseline на train, оценивает на validation, сохраняет pipeline и отчёт."""
    df = load_clean()
    texts = df.text_raw.to_numpy(dtype=object)
    labels = df.label.to_numpy()
    split = load_or_create_split(labels)
    tr, va, te = (np.asarray(split[k]) for k in ('train', 'val', 'test'))

    model = build_baseline_pipeline()
    model.fit(texts[tr], labels[tr])                      # fit — только на обучающей выборке
    prob = model.predict_proba(texts[va])[:, 1]           # метрики — только на валидационной
    pred = (prob >= THRESHOLD).astype(int)

    features = model.named_steps['features']
    report = {
        'threshold': THRESHOLD,
        'sizes': {'train': len(tr), 'val': len(va), 'test': len(te)},
        'spam_share': {k: float(labels[np.asarray(split[k])].mean()) for k in ('train', 'val', 'test')},
        'feature_dimensions': {
            'train': list(features.transform(texts[tr]).shape),
            'val': list(features.transform(texts[va]).shape),
            'test': list(features.transform(texts[te]).shape),
        },
        'validation': _metrics(labels[va], pred),
    }

    fn = np.where((labels[va] == 1) & (pred == 0))[0]
    fp = np.where((labels[va] == 0) & (pred == 1))[0]
    report['error_counts'] = {'FN': int(len(fn)), 'FP': int(len(fp))}

    def describe(kind, idx):
        return [{'type': kind, 'val_row': int(va[i]), 'probability': float(prob[i]), 'text': str(texts[va[i]])} for i in idx]

    errors = describe('FN', fn[np.argsort(prob[fn])][:5]) + describe('FP', fp[np.argsort(-prob[fp])][:5])
    report['errors'] = errors
    # Если реальных FP/FN меньше пяти, ошибки не выдумываются: добавляются «почти ошибки» —
    # верно классифицированные ham с наибольшей вероятностью spam (пограничные случаи).
    if len(fp) < 5:
        ok_ham = np.where((labels[va] == 0) & (pred == 0))[0]
        near = ok_ham[np.argsort(-prob[ok_ham])][:5 - len(fp)]
        report['near_misses_ham'] = describe('near-FP', near)

    if save:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ARTIFACT)
        reloaded = joblib.load(ARTIFACT)
        report['reload_predictions_identical'] = bool(np.allclose(reloaded.predict_proba(texts[va])[:, 1], prob))
        ok, detail = _fresh_interpreter_check()
        report['fresh_interpreter_load_ok'] = ok
        if not ok:
            raise RuntimeError(f'Артефакт не загружается в новом процессе: {detail}')
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


_cached = None


def predict_sms(text):
    """Предсказание по сохранённому pipeline. Нормализация (``normalize_sms``) выполняется внутри pipeline.

    Возвращает ``{'label': 0|1, 'probability': P(spam)}``; метка — по порогу 0,5.
    """
    global _cached
    if _cached is None:
        _cached = joblib.load(ARTIFACT)
    probability = float(_cached.predict_proba([str(text)])[0, 1])
    return {'label': int(probability >= THRESHOLD), 'probability': probability}


if __name__ == '__main__':
    print(json.dumps(train_baseline(), ensure_ascii=False, indent=2))
