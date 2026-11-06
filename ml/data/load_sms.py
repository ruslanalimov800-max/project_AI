"""Загрузка оригинального UCI SMS Spam Collection без изменения исходного файла (задание 3).

Две ловушки этого набора, которые здесь обработаны явно:
* кодировка — файл в UTF-8 (например, знак фунта «£» записан байтами C2 A3); чтение как latin-1
  превращает его в «Â£», поэтому кодировка декодируется строго и при ошибке выдаётся исключение;
* кавычки — сообщения содержат одиночные ``"``; при обычном разборе CSV такие строки склеиваются
  с соседними, поэтому используется ``quoting=csv.QUOTE_NONE``.
"""
import csv
import hashlib
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'data/raw/sms_dataset/SMSSpamCollection'
ENCODING = 'utf-8'
LABELS = {'ham': 0, 'spam': 1}


def file_sha256(path=None):
    """SHA-256 исходного файла (фиксируется в паспорте набора)."""
    return hashlib.sha256(Path(path or DEFAULT).read_bytes()).hexdigest()


def load_sms(path=None):
    """Возвращает DataFrame со столбцами id, text, label (0 — ham, 1 — spam)."""
    source = Path(path) if path is not None else DEFAULT
    raw = source.read_bytes()
    try:
        content = raw.decode(ENCODING)  # строгая проверка кодировки: без подмены символов
    except UnicodeDecodeError as exc:
        raise ValueError(f'{source.name}: ожидается кодировка {ENCODING}: {exc}') from exc
    df = pd.read_csv(io.StringIO(content), sep='\t', header=None, names=['label', 'text'],
                     dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    if not df.label.isin(LABELS).all():
        raise ValueError('Неожиданные метки классов (допустимы только ham/spam)')
    expected = sum(1 for line in content.splitlines() if line.strip())
    if len(df) != expected:  # критерий задания 3: число строк совпадает с исходным файлом
        raise ValueError(f'Прочитано {len(df)} строк, в файле {expected}')
    df['label'] = df.label.map(LABELS).astype('int64')
    df.insert(0, 'id', range(len(df)))
    return df[['id', 'text', 'label']]


if __name__ == '__main__':
    df = load_sms()
    print('Строк:', len(df))
    print('Типы столбцов:', df.dtypes.astype(str).to_dict())
    print('Распределение классов:', df.label.value_counts().sort_index().to_dict())
    for label in (0, 1):
        print(f'\nКласс {label} — 5 примеров:')
        print(df[df.label == label].head(5).to_string(index=False))
