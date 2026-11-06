"""Задание 6: разбиение 70/15/15 и извлечение признаков SMS (TF-IDF слов и символов + ручные признаки).

Все преобразования, обучаемые на данных (TF-IDF, масштабирование), находятся внутри ``Pipeline`` и обучаются
только на обучающей выборке: ``fit`` на train, ``transform`` на val/test. Классы лежат в этом модуле,
а не в ``__main__``, поэтому сохранённый joblib загружается в любом новом процессе Python.
"""
import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

from ml.data.load_sms import ROOT
from ml.preprocessing import normalize_sms

RANDOM_STATE = 42
CLEAN_CSV = ROOT / 'data/processed/sms_clean.csv'
SPLIT_JSON = ROOT / 'data/processed/sms_split.json'

# Группы признаков социальной инженерии для ручного признака «число сработавших групп».
# Числа в нормализованном тексте уже заменены на <num>, поэтому «24 часа» ищется как «<num> час».
# Совпадения ищутся в нормализованном тексте; слева — граница слова, справа — основа слова допускается.
SE_GROUPS = {
    'urgency': r'urgent|immediately|asap|hurry|within <num> hours|срочно|немедленно|в течение <num> час|тез арада|шұғыл|дереу',
    'money': r'transfer|payment|\bpay\b|cash|<money>|перевод|перевед|оплат|ақша|төлем|аудар',
    'bank_card': r'bank|card|account|банк|карт[аыуе]|счёт|счет|kaspi|halyk|шот|карта',
    'account_block': r'blocked|suspend|locked|заблок|блокиров|бұғат',
    'prize': r'winner|\bwon\b|prize|congratulations|claim|выигр|приз|ұтыс|жүлде',
    'code_request': r'\bcode\b|\botp\b|password|\bpin\b|код|парол|құпия',
    'link_request': r'click|<url>|\blink\b|перейдите|ссылк|нажмите|сілтеме',
}
_SE_PATTERNS = {name: re.compile(r'(?<!\w)(?:' + body + ')') for name, body in SE_GROUPS.items()}
MANUAL_FEATURE_NAMES = ['length', 'uppercase_ratio', 'exclamation_count', 'has_url', 'has_money', 'se_groups']


def count_se_groups(clean_text):
    """Число групп социальной инженерии, сработавших в нормализованном тексте."""
    return sum(bool(p.search(clean_text)) for p in _SE_PATTERNS.values())


class SmsNormalizer(BaseEstimator, TransformerMixin):
    """Шаг Pipeline: единая функция ``normalize_sms`` (та же, что в приложении)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalize_sms(x) for x in X]


class ManualFeatures(BaseEstimator, TransformerMixin):
    """Ручные признаки по исходному тексту (регистр важен): см. ``MANUAL_FEATURE_NAMES``."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for raw in X:
            text = str(raw)
            clean = normalize_sms(text)
            letters = [c for c in text if c.isalpha()]
            rows.append([
                len(text),
                sum(c.isupper() for c in letters) / max(1, len(letters)),
                text.count('!'),
                int('<url>' in clean),
                int('<money>' in clean),
                count_se_groups(clean),
            ])
        return csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(MANUAL_FEATURE_NAMES)


def build_feature_pipeline():
    """FeatureUnion: TF-IDF слов (1–2), TF-IDF символов char_wb (3–5) и масштабированные ручные признаки."""
    word = Pipeline([
        ('normalize', SmsNormalizer()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
                                  token_pattern=r'(?u)\b\w+\b|<\w+>')),
    ])
    char = Pipeline([
        ('normalize', SmsNormalizer()),
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    manual = Pipeline([('compute', ManualFeatures()), ('scale', MaxAbsScaler())])
    return FeatureUnion([('word', word), ('char', char), ('manual', manual)])


def split_indices(y):
    """Стратифицированное разбиение 70/15/15, random_state=42; возвращает индексы строк."""
    y = np.asarray(y)
    ids = np.arange(len(y))
    train, rest = train_test_split(ids, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=RANDOM_STATE)
    return {'train': train.tolist(), 'val': val.tolist(), 'test': test.tolist()}


def load_clean():
    """Очищенный набор (sms_clean.csv); при отсутствии файла он создаётся скриптом подготовки."""
    if not CLEAN_CSV.exists():
        from ml.data.prepare_sms import prepare
        prepare()
    return pd.read_csv(CLEAN_CSV, keep_default_na=False)


def load_or_create_split(labels):
    """Читает сохранённые индексы; если файла нет — создаёт и сохраняет (воспроизводимость эксперимента)."""
    if SPLIT_JSON.exists():
        split = json.loads(SPLIT_JSON.read_text(encoding='utf-8'))
        if sum(len(v) for v in split.values()) != len(labels):
            raise ValueError('sms_split.json не соответствует sms_clean.csv: пересоздайте разбиение')
        return split
    split = split_indices(labels)
    SPLIT_JSON.write_text(json.dumps(split), encoding='utf-8')
    return split


def make_matrices(texts, split):
    """fit на train, transform на val/test. Возвращает (features_pipeline, X_train, X_val, X_test)."""
    texts = np.asarray(texts, dtype=object)
    features = build_feature_pipeline()
    X_train = features.fit_transform(texts[split['train']])
    X_val = features.transform(texts[split['val']])
    X_test = features.transform(texts[split['test']])
    return features, X_train, X_val, X_test


def feature_names(features):
    """Имена столбцов матрицы признаков (с префиксами word__/char__/manual__)."""
    names = []
    for name, transformer in features.transformer_list:
        last = transformer.steps[-1][1]
        if name == 'manual':
            names += [f'manual__{n}' for n in MANUAL_FEATURE_NAMES]
        else:
            names += [f'{name}__{t}' for t in last.get_feature_names_out()]
    return np.asarray(names)


def top_terms_by_class(features, X_train, y_train, k=15):
    """Топ-k слов/биграмм с наибольшим средним TF-IDF для каждого класса (по обучающей выборке)."""
    names = feature_names(features)
    word_cols = np.where(np.char.startswith(names, 'word__'))[0]
    X_word = X_train.tocsc()[:, word_cols]
    y_train = np.asarray(y_train)
    result = {}
    for cls in (0, 1):
        means = np.asarray(X_word[y_train == cls].mean(axis=0)).ravel()
        order = np.argsort(means)[::-1][:k]
        result[cls] = [(names[word_cols[i]][len('word__'):], float(means[i])) for i in order]
    return result
