"""Задание 6: разбиение и признаки без утечки."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from ml.sms_features import (MANUAL_FEATURE_NAMES, SPLIT_JSON, ManualFeatures, SmsNormalizer, build_feature_pipeline,
                             count_se_groups, make_matrices, split_indices, top_terms_by_class)


def test_split_70_15_15_stratified_and_disjoint():
    y = np.array([0] * 700 + [1] * 300)
    s = split_indices(y)
    assert (len(s['train']), len(s['val']), len(s['test'])) == (700, 150, 150)
    assert sorted(s['train'] + s['val'] + s['test']) == list(range(1000))
    for part in s.values():
        assert y[part].mean() == pytest.approx(0.3, abs=0.01)


def test_split_is_reproducible():
    y = np.array([0, 1] * 50)
    assert split_indices(y) == split_indices(y)


def test_saved_split_matches_clean_dataset():
    clean = pd.read_csv('data/processed/sms_clean.csv', keep_default_na=False)
    saved = json.loads(SPLIT_JSON.read_text(encoding='utf-8'))
    assert saved == split_indices(clean.label.to_numpy())
    assert sum(len(v) for v in saved.values()) == len(clean)


def test_manual_features_values():
    X = ManualFeatures().transform(['WIN!! http://example.com £100', 'hi']).toarray()
    assert X.shape == (2, len(MANUAL_FEATURE_NAMES)) and len(MANUAL_FEATURE_NAMES) >= 5
    length, upper, excl, has_url, has_money, groups = X[0]
    assert length == len('WIN!! http://example.com £100') and excl == 2 and has_url == 1 and has_money == 1
    assert 0 < upper < 1 and groups >= 2                                    # prize + link/money
    assert X[1].tolist()[2:] == [0, 0, 0, 0]


@pytest.mark.parametrize('text,expected', [
    ('срочно переведите оплата заблокирована карта', 4),
    ('hello, how are you', 0),
    ('urgent: your account is blocked, send the code', 3),
])
def test_count_se_groups(text, expected):
    from ml.preprocessing import normalize_sms
    assert count_se_groups(normalize_sms(text)) >= expected if expected else count_se_groups(normalize_sms(text)) == 0


def test_pipeline_fit_on_train_only_vocabulary_has_no_eval_words():
    train = ['free prize call now', 'free entry win cash', 'see you at lunch', 'ok see you soon', 'call me later please']
    evals = ['zzzunseenword appears only in evaluation text']
    pipe = build_feature_pipeline().fit(train)
    vocab = dict(pipe.transformer_list)['word'].named_steps['tfidf'].vocabulary_
    assert 'zzzunseenword' not in vocab
    assert pipe.transform(evals).shape[1] == pipe.transform(train).shape[1]   # размерность задаётся только train


def test_matrices_shapes_consistent():
    texts = np.array(['free prize call %d now' % i if i % 4 == 0 else 'see you at lunch %d ok' % i for i in range(80)], dtype=object)
    y = np.array([1 if i % 4 == 0 else 0 for i in range(80)])
    split = split_indices(y)
    features, X_tr, X_va, X_te = make_matrices(texts, split)
    assert (X_tr.shape[0], X_va.shape[0], X_te.shape[0]) == (56, 12, 12)
    assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1]
    top = top_terms_by_class(features, X_tr, y[split['train']], k=5)
    assert len(top[0]) == len(top[1]) == 5


def test_normalizer_step_uses_normalize_sms():
    assert SmsNormalizer().transform(['HELLO  https://example.com']) == ['hello <url>']
    assert not re.search('Â', ' '.join(SmsNormalizer().transform(['£5'])))
