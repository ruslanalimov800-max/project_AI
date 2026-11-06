"""Задание 7: baseline-модель, функция predict_sms, переносимость артефакта."""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pytest

from ml.sms_baseline import ARTIFACT, REPORT, build_baseline_pipeline, predict_sms

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_pipeline_parameters():
    lr = build_baseline_pipeline().named_steps['classifier']
    assert lr.class_weight == 'balanced' and lr.max_iter == 1000


def test_predict_sms_returns_label_and_probability():
    for text in ('WINNER!! Call 09061701461 to claim your £1000 prize', 'Ok, see you tomorrow', '', 'Сәлем'):
        result = predict_sms(text)
        assert set(result) == {'label', 'probability'}
        assert 0 <= result['probability'] <= 1 and result['label'] in (0, 1)
    assert predict_sms('WINNER!! Call 09061701461 to claim your £1000 prize')['label'] == 1
    assert predict_sms('Ok, see you tomorrow')['label'] == 0


def test_artifact_does_not_reference_main_module():
    """Классы преобразователей должны быть в ml.sms_features: иначе joblib не загрузится в новом процессе."""
    assert b'__main__' not in ARTIFACT.read_bytes()


def test_baseline_loads_in_fresh_interpreter():
    code = ("from ml.sms_baseline import predict_sms; p=predict_sms('hello'); "
            "assert 0 <= p['probability'] <= 1; assert p['label'] in (0, 1)")
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)


def test_saved_pipeline_is_a_single_pipeline_with_features_inside():
    model = joblib.load(ARTIFACT)
    assert [name for name, _ in model.steps] == ['features', 'classifier']
    assert [name for name, _ in model.named_steps['features'].transformer_list] == ['word', 'char', 'manual']


def test_report_is_validation_only_and_consistent():
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    tn, fp = report['validation']['confusion_matrix'][0]
    fn, tp = report['validation']['confusion_matrix'][1]
    assert tn + fp + fn + tp == report['sizes']['val']
    assert report['error_counts'] == {'FN': fn, 'FP': fp}
    assert report['reload_predictions_identical'] and report['fresh_interpreter_load_ok']
    assert report['validation']['recall'] == pytest.approx(tp / (tp + fn))
    assert report['validation']['recall'] > 0.8        # sanity: baseline заметно лучше случайного
