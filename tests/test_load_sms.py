"""Задание 3: загрузчик SMS-набора и собственный русско-казахский набор."""
import re

import pandas as pd
import pytest

from ml.data.load_sms import DEFAULT, ROOT, load_sms


def test_row_count_matches_source_file():
    """Критерий задания 3: число строк совпадает с исходным файлом (ни одна строка не потеряна и не склеена)."""
    lines = [line for line in DEFAULT.read_text(encoding='utf-8').splitlines() if line.strip()]
    df = load_sms()
    assert len(df) == len(lines) == 5574
    assert df.label.value_counts().to_dict() == {0: 4827, 1: 747}


def test_structure_and_types():
    df = load_sms()
    assert list(df.columns) == ['id', 'text', 'label']
    assert df.id.is_unique and set(df.label) == {0, 1}
    assert not df.text.str.contains('\n').any()          # склеенных строк нет


def test_utf8_pound_sign_is_not_mojibake():
    df = load_sms()
    assert not df.text.str.contains('Â').any()
    assert df.text.str.contains('£').sum() > 100


def test_quotes_do_not_merge_rows(tmp_path):
    path = tmp_path / 'sample'
    path.write_text('ham\t"Quoted start\nham\tsecond row\nspam\tthird "row"\n', encoding='utf-8')
    df = load_sms(path)
    assert len(df) == 3 and df.label.tolist() == [0, 0, 1]


def test_non_utf8_file_is_rejected_explicitly(tmp_path):
    path = tmp_path / 'latin1'
    path.write_bytes('ham\tprice £5\n'.encode('latin-1'))
    with pytest.raises(ValueError, match='utf-8'):
        load_sms(path)


def test_unknown_label_is_rejected(tmp_path):
    path = tmp_path / 'bad'
    path.write_text('maybe\ttext\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_sms(path)


def test_processed_raw_csv_has_all_rows():
    raw = pd.read_csv(ROOT / 'data/processed/sms_raw.csv', keep_default_na=False)
    assert len(raw) == 5574 and list(raw.columns) == ['id', 'text', 'label']


# --- собственный тестовый набор RU/KK ---------------------------------------------------------------
@pytest.fixture(scope='module')
def custom():
    return pd.read_csv(ROOT / 'data/raw/custom_ru_kk.csv', keep_default_na=False)


def test_custom_dataset_sizes_and_balance(custom):
    assert (custom.language == 'ru').sum() >= 100 and (custom.language == 'kk').sum() >= 50
    assert abs(custom.label.mean() - 0.5) <= 0.05                       # ≈ 50% мошеннических
    assert custom.annotation_author.str.strip().ne('').all()            # автор разметки указан


def test_custom_dataset_has_no_real_personal_data(custom):
    assert not custom.text.str.contains(r'\d{7,}').any()                # нет номеров телефонов и карт
    assert not custom.text.str.contains(r'\d{4}[ -]\d{4}').any()
    urls = [u for t in custom.text for u in re.findall(r'https?://\S+', t)]
    assert all(re.search(r'\.(example|test|invalid)\b', u) for u in urls)   # только зарезервированные домены


def test_custom_dataset_texts_are_unique_and_kazakh_is_kazakh(custom):
    assert custom.text.is_unique
    kk = custom[custom.language == 'kk'].text
    assert kk.str.contains('[әғқңөұүһі]').all()
