"""Задание 4: normalize_sms — чистая функция; сущности, кириллица, казахские буквы, эмодзи, крайние случаи."""
import pytest

from ml.preprocessing import NORMALIZATION_STEPS, normalization_report, normalize_sms

KAZAKH = 'әғқңөұүһі'


def test_empty_string():
    assert normalize_sms('') == ''


def test_none_is_empty():
    assert normalize_sms(None) == ''


def test_only_whitespace_and_control_chars():
    assert normalize_sms('  \t\n\x00\x07 ') == ''


def test_only_url():
    assert normalize_sms('https://example.com/path?x=1') == '<url>'


def test_www_url():
    assert normalize_sms('go to www.example.com now') == 'go to <url> now'


def test_mixed_case():
    assert normalize_sms('HeLLo WoRLD') == 'hello world'


def test_cyrillic_lowercased():
    assert normalize_sms('ПРИВЕТ Мир') == 'привет мир'


def test_kazakh_letters_are_preserved():
    assert normalize_sms(KAZAKH.upper()) == KAZAKH
    assert all(c in normalize_sms('Сәлем, құрметті қолданушы! Өтініш, ұмытпаңыз') for c in 'әқұө')


@pytest.mark.parametrize('text', ['5000 тг', '5000тг', '100 руб', '$50', '£1000', '₸ 2500', '20 usd'])
def test_money(text):
    assert normalize_sms(text) == '<money>'


@pytest.mark.parametrize('text', ['+7 (777) 123-45-67', '87771234567', '09061701461'])
def test_phone(text):
    assert normalize_sms(text) == '<phone>'


def test_email():
    assert normalize_sms('Write to Hello.World@Example.com') == 'write to <email>'


def test_other_numbers_become_num():
    assert normalize_sms('Code 1234, level 3.5') == 'code <num>, level <num>'


def test_emoji_is_kept():
    assert normalize_sms('Сәлем 🙂') == 'сәлем 🙂'


def test_repeated_punctuation_collapsed():
    assert normalize_sms('СРОЧНО!!! Правда???') == 'срочно! правда?'
    assert '!' in normalize_sms('hi!') and '?' in normalize_sms('why?')


def test_html_entities_and_uci_placeholder():
    assert normalize_sms('Call me &lt;#&gt; &amp; wait') == 'call me <num> & wait'


def test_nfkc():
    assert normalize_sms('ＡＢＣ') == 'abc'


def test_extra_spaces_removed():
    assert normalize_sms('a   b \n c') == 'a b c'


def test_stopwords_are_optional():
    assert normalize_sms('the cat and the dog') == 'the cat and the dog'
    assert normalize_sms('the cat and the dog', remove_stopwords=True) == 'cat dog'


def test_pure_and_idempotent():
    text = 'WIN £1000!!! Call 09061701461 or https://example.com'
    original = text
    once = normalize_sms(text)
    assert text == original and normalize_sms(text) == once      # без побочных эффектов, детерминирована
    assert normalize_sms(once) == once                            # повторная нормализация ничего не меняет


def test_report_has_one_row_per_step_and_counts_changes():
    texts = ['Hello', 'WIN £10!!!', 'see https://example.com']
    report = normalization_report(texts)
    assert [r['step'] for r in report] == [name for name, _ in NORMALIZATION_STEPS]
    by_step = {r['step']: r['affected'] for r in report}
    assert by_step['Нижний регистр'] == 2
    assert by_step['URL → <url>'] == 1
    assert by_step['Сумма с валютой → <money>'] == 1
    assert by_step['Повторы !!! и ??? → ! и ?'] == 1
