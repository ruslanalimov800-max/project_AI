"""Предобработка SMS (задание 4).

``normalize_sms(text) -> str`` — чистая функция без побочных эффектов; одна и та же функция
используется в ноутбуках, при обучении и в приложении. Преобразование задано списком явных шагов
``NORMALIZATION_STEPS``: тот же список использует ``normalization_report`` для таблицы
«шаг — число затронутых записей», поэтому отчёт не может разойтись с реальной обработкой.

Казахские буквы (ә ғ қ ң ө ұ ү һ і) и кириллица сохраняются: удаляются только управляющие символы.
"""
import html
import re
import unicodedata

EMAIL_RE = re.compile(r'(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b')
URL_RE = re.compile(r'(?i)\b(?:https?://|www\.)[^\s<>]+|\b(?:[\w-]+\.)+(?:com|org|net|kz|ru|io|gov|edu)\b(?:/[^\s<>]*)?')
PHONE_RE = re.compile(r'(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)')
MONEY_RE = re.compile(
    r'(?i)(?<!\w)(?:[$£€₸]\s*\d(?:[\d.,]*\d)?'
    r'|\d(?:[\d.,]*\d)?\s*(?:руб(?:лей|ля|ль)?|тг|тенге|usd|eur|kzt|[$£€₸]))(?!\w)')
# «<#>» — так в UCI записан числовой заполнитель (в файле он встречается как &lt;#&gt;)
NUMBER_RE = re.compile(r'<#>|\d+(?:[.,]\d+)*')
REPEATED_EXCLAMATION_RE = re.compile(r'!{2,}')
REPEATED_QUESTION_RE = re.compile(r'\?{2,}')

# Минимальный список стоп-слов (en/ru/kk); применяется только по опции remove_stopwords=True.
STOPWORDS = frozenset(
    'the a an and or is are was were be to of in on for at it this that with from as by i you he she we they me my your '
    'и в во на с со по из не что это как а но или к у о об за для от до я ты он она мы вы они '
    'және бұл деп үшін мен сен ол біз сіз олар бір'.split())


def _control_to_space(text):
    return ''.join(' ' if unicodedata.category(ch).startswith('C') else ch for ch in text)


def _repeated_punctuation(text):
    return REPEATED_QUESTION_RE.sub('?', REPEATED_EXCLAMATION_RE.sub('!', text))


NORMALIZATION_STEPS = (
    ('Unicode-нормализация NFKC', lambda s: unicodedata.normalize('NFKC', s)),
    ('HTML-сущности (lt, gt, amp и др.)', html.unescape),
    ('Нижний регистр', str.lower),
    ('Удаление управляющих символов', _control_to_space),
    ('e-mail → <email>', lambda s: EMAIL_RE.sub('<email>', s)),
    ('URL → <url>', lambda s: URL_RE.sub('<url>', s)),
    ('Телефон → <phone>', lambda s: PHONE_RE.sub('<phone>', s)),
    ('Сумма с валютой → <money>', lambda s: MONEY_RE.sub('<money>', s)),
    ('Прочие числа → <num>', lambda s: NUMBER_RE.sub('<num>', s)),
    ('Повторы !!! и ??? → ! и ?', _repeated_punctuation),
    ('Лишние пробелы', lambda s: re.sub(r'\s+', ' ', s).strip()),
)


def normalize_sms(text, remove_stopwords=False):
    """Нормализует SMS: см. ``NORMALIZATION_STEPS``. ``None`` трактуется как пустая строка."""
    s = '' if text is None else str(text)
    for _, step in NORMALIZATION_STEPS:
        s = step(s)
    if remove_stopwords:
        s = ' '.join(w for w in s.split() if w not in STOPWORDS)
    return s


def normalization_report(texts, remove_stopwords=False):
    """Для каждого шага — число записей, которые он изменил (шаги применяются последовательно)."""
    current = ['' if t is None else str(t) for t in texts]
    rows = []
    for name, step in NORMALIZATION_STEPS:
        new = [step(s) for s in current]
        rows.append({'step': name, 'affected': sum(a != b for a, b in zip(current, new))})
        current = new
    if remove_stopwords:
        new = [' '.join(w for w in s.split() if w not in STOPWORDS) for s in current]
        rows.append({'step': 'Удаление стоп-слов (опция)', 'affected': sum(a != b for a, b in zip(current, new))})
    return rows
