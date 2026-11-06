"""Воспроизводимая подготовка SMS-набора (задания 3–4).

Результат:
* data/processed/sms_raw.csv   — id, text, label (все строки исходного файла, UTF-8);
* data/processed/sms_clean.csv — id, text_raw, label, text_clean (после нормализации и удаления дубликатов);
* data/processed/sms_processing_report.json и docs/04_PREPROCESSING_REPORT.md — отчёт о преобразованиях.
"""
import json

from ml.data.load_sms import ROOT, load_sms
from ml.preprocessing import normalization_report, normalize_sms

CLASS_NAMES = {0: 'ham', 1: 'spam'}


def _by_class(series_mask, labels):
    return {CLASS_NAMES[c]: int(series_mask[labels == c].sum()) for c in (0, 1)}


def prepare():
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    raw = load_sms()
    raw.to_csv(out / 'sms_raw.csv', index=False, encoding='utf-8')

    df = raw.rename(columns={'text': 'text_raw'})
    labels = df.label

    # таблица «шаг — число затронутых записей»: всего и по классам
    steps = []
    per_class = {c: normalization_report(df.text_raw[labels == c]) for c in (0, 1)}
    for i, (name, total) in enumerate((r['step'], r['affected']) for r in normalization_report(df.text_raw)):
        steps.append({'step': name, 'affected': total,
                      'ham': per_class[0][i]['affected'], 'spam': per_class[1][i]['affected']})

    df['text_clean'] = df.text_raw.map(normalize_sms)

    empty_raw = df.text_raw.str.strip().eq('')
    empty_clean = df.text_clean.eq('')
    empty = empty_raw | empty_clean  # пустой исходный текст или текст, ставший пустым после очистки
    removed_empty = _by_class(empty, labels)
    df = df[~empty].copy()

    duplicated = df.duplicated('text_clean', keep='first')
    removed_duplicates = _by_class(duplicated, df.label)
    df = df[~duplicated].reset_index(drop=True)
    df.to_csv(out / 'sms_clean.csv', index=False, encoding='utf-8')

    counts = df.label.value_counts().sort_index()
    report = {
        'original_rows': len(raw),
        'original_by_class': {CLASS_NAMES[c]: int((raw.label == c).sum()) for c in (0, 1)},
        'empty_removed_by_class': removed_empty,
        'duplicates_removed_by_class': removed_duplicates,
        'clean_rows': len(df),
        'clean_by_class': {CLASS_NAMES[c]: int(counts[c]) for c in (0, 1)},
        'steps': steps,
    }
    (out / 'sms_processing_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    _write_markdown(report)
    return df, report


def _write_markdown(r):
    lines = [
        '# Задание 4. Отчёт о предобработке SMS', '',
        'Сгенерирован `python -m ml.data.prepare_sms`; шаги берутся из `ml/preprocessing.py::NORMALIZATION_STEPS`.', '',
        '## Шаги нормализации: число затронутых записей', '',
        '| Шаг | Всего | ham | spam |', '|---|---:|---:|---:|']
    lines += [f"| {s['step']} | {s['affected']} | {s['ham']} | {s['spam']} |" for s in r['steps']]
    lines += ['',
              'Шаги применяются последовательно; столбец — число записей, изменённых именно этим шагом, поэтому одна запись может входить в несколько строк.', '',
              '## Движение записей', '',
              '| Этап | Всего | ham | spam |', '|---|---:|---:|---:|',
              f"| Исходный файл | {r['original_rows']} | {r['original_by_class']['ham']} | {r['original_by_class']['spam']} |",
              f"| Удалены пустые тексты | {sum(r['empty_removed_by_class'].values())} | {r['empty_removed_by_class']['ham']} | {r['empty_removed_by_class']['spam']} |",
              f"| Удалены дубликаты по `text_clean` | {sum(r['duplicates_removed_by_class'].values())} | {r['duplicates_removed_by_class']['ham']} | {r['duplicates_removed_by_class']['spam']} |",
              f"| **Итог (`sms_clean.csv`)** | **{r['clean_rows']}** | **{r['clean_by_class']['ham']}** | **{r['clean_by_class']['spam']}** |",
              '', 'Дубликаты удаляются до разделения на выборки, чтобы одинаковые сообщения не попали одновременно в обучение и тест.', '']
    (ROOT / 'docs/04_PREPROCESSING_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    print(json.dumps(prepare()[1], ensure_ascii=False, indent=2))
