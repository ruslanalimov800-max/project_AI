"""Извлечение структурных признаков URL (Задание 9, ФТ-3.1..ФТ-3.8).
URL анализируется строго как строка: сетевые запросы (requests, socket, DNS) запрещены.
"""
from collections import Counter
from functools import lru_cache
import ipaddress
import math
import re
from urllib.parse import parse_qsl, unquote, urlsplit
import tldextract

# Отключаем любые попытки tldextract обращаться к сети за обновлением списка PSL
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


@lru_cache(maxsize=65536)
def _extract_tld_cached(host: str):
    return _TLD_EXTRACTOR(host)

# Не менее 15 сервисов сокращения ссылок (ФТ-3.6, Задание 9 п.15)
SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd", "cutt.ly", "ow.ly", "clck.ru",
    "rb.gy", "tiny.cc", "shorturl.at", "v.gd", "buff.ly", "rebrand.ly", "bl.ink", "u.to", "vk.cc",
})

# Подозрительные доменные зоны (Задание 9 п.21; источник: отчёты Spamhaus / APWG по фишинговым TLD)
SUSPICIOUS_TLDS = frozenset({
    "xyz", "top", "click", "zip", "mov", "work", "country", "gq", "tk", "ml",
    "cf", "ga", "icu", "buzz", "loan", "win", "vip", "monster", "rest", "fit",
})

# Подозрительные ключевые слова в URL (ФТ-3.7, Задание 9 п.16)
SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "secure", "account", "update", "bank", "confirm",
    "password", "bonus", "free", "prize", "kaspi", "halyk", "auth", "wallet",
    "unlock", "suspend", "pay", "claim", "otp", "card", "cabinet",
)

# Список >= 20 официальных доменов банков и сервисов для проверки тайпосквоттинга (Задание 9 п.19)
BRAND_DOMAINS = (
    "kaspi.kz", "halykbank.kz", "jusan.kz", "bcc.kz", "forte.kz", "bereke.kz", "eubank.kz",
    "homecredit.kz", "freedomfinance.kz", "egov.kz", "kazpost.kz", "sberbank.ru", "tinkoff.ru",
    "vtb.ru", "alfabank.ru", "gosuslugi.ru", "paypal.com", "google.com", "apple.com",
    "microsoft.com", "amazon.com", "whatsapp.com", "telegram.org", "instagram.com",
)
_BRAND_NAMES = tuple(d.split(".")[0] for d in BRAND_DOMAINS)

FEATURE_NAMES = (
    "url_length",
    "host_length",
    "path_length",
    "dot_count",
    "subdomain_count",
    "domain_hyphen_count",
    "digit_count",
    "domain_digit_ratio",
    "special_char_count",
    "has_ip",
    "https",
    "non_standard_port",
    "has_at",
    "double_slash_in_path",
    "has_shortener",
    "suspicious_word_count",
    "has_punycode",
    "mixed_alphabet",
    "brand_spoof",
    "domain_entropy",
    "suspicious_tld",
    "query_param_count",
    # Совместимые псевдонимы для существующих шаблонов и тестов
    "query_length",
    "hyphen_count",
    "at_count",
    "question_count",
    "equal_count",
    "slash_count",
    "digit_ratio",
    "special_ratio",
    "has_brand_word",
    "has_login_word",
    "has_redirect",
)


def _normalize_url(value):
    value = str(value or "").strip().strip("()[]{}<>,.;!?\"'")
    if not value:
        return ""
    if not re.match(r"(?i)^[a-z][a-z0-9+.-]*://", value):
        value = "http://" + value
    return value


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _normalize_leetspeak(s: str) -> str:
    table = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
    return s.translate(table)


@lru_cache(maxsize=65536)
def _is_brand_spoof(host: str, reg_domain: str, main_domain: str) -> int:
    if not host or reg_domain in BRAND_DOMAINS:
        return 0
    # 1) Бренд присутствует как часть составного домена или поддомена (например kaspi-verify.test)
    for brand, official in zip(_BRAND_NAMES, BRAND_DOMAINS):
        if brand in host and reg_domain != official:
            return 1
    # 2) Расстояние Левенштейна <= 2 между основным именем домена (или его частями) и брендом
    candidates = [main_domain, _normalize_leetspeak(main_domain)] + [p for p in main_domain.split("-") if len(p) >= 4]
    for cand in candidates:
        if len(cand) < 4:
            continue
        for brand in _BRAND_NAMES:
            if abs(len(cand) - len(brand)) <= 2 and 0 < _levenshtein(cand, brand) <= 2:
                return 1
    return 0


def _has_mixed_alphabet(host: str) -> int:
    has_lat = bool(re.search(r"[a-z]", host, re.I))
    has_cyr = bool(re.search(r"[а-яёәғқңөұүһі]", host, re.I))
    return int(has_lat and has_cyr)


def validate_url_with_reason(value) -> tuple[bool, str]:
    """Проверяет формат URL (ФТ-3.1, Задание 9) и возвращает (is_valid, error_message)."""
    raw = str(value or "").strip()
    if not raw:
        return False, "URL не может быть пустым."
    if len(raw) > 2048:
        return False, "Длина URL превышает максимально допустимые 2048 символов."
    if any(ch.isspace() for ch in raw) or any(ord(ch) < 32 for ch in raw):
        return False, "URL содержит пробелы или управляющие символы."
    normalized = _normalize_url(raw)
    try:
        parsed = urlsplit(normalized)
        _ = parsed.port
    except ValueError:
        return False, "Некорректный формат URL или порта."
    if parsed.scheme.lower() not in ("http", "https"):
        return False, "Допустимы только схемы http и https."
    host = (parsed.hostname or "").strip()
    if not host:
        return False, "В URL отсутствует доменное имя."
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True, ""
    except ValueError:
        if "." not in host or host.startswith(".") or host.endswith(".") or ".." in host:
            return False, "Некорректное доменное имя."
        return True, ""


def validate_url(value) -> bool:
    """Булева обёртка над validate_url_with_reason для обратной совместимости."""
    ok, _ = validate_url_with_reason(value)
    return ok


def extract_url_features(value) -> dict:
    """Возвращает упорядоченный словарь числовых признаков URL (22 основных + совместимые ключи)."""
    raw = str(value or "").strip()
    normalized = _normalize_url(raw)
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        parsed = urlsplit("http://invalid")
    host = (parsed.hostname or "").lower().rstrip(".")
    path = unquote(parsed.path or "")
    query = unquote(parsed.query or "")
    try:
        port = parsed.port
    except ValueError:
        port = None

    try:
        ipaddress.ip_address(host.strip("[]"))
        is_ip = 1
    except ValueError:
        is_ip = 0

    ext = _extract_tld_cached(host) if host and not is_ip else None
    subdomain_str = ext.subdomain if ext else ""
    subdomain_count = len([s for s in subdomain_str.split(".") if s]) if subdomain_str else 0
    main_domain = (ext.domain if ext and ext.domain else host.split(".")[0] if host else "")
    suffix = (ext.suffix if ext and ext.suffix else (host.split(".")[-1] if "." in host else ""))
    reg_domain = f"{main_domain}.{suffix}" if main_domain and suffix else host

    digits_url = sum(ch.isdigit() for ch in raw)
    digits_host = sum(ch.isdigit() for ch in host)
    special_chars_set = set("@~%=&_?")
    special_char_count = sum(ch in special_chars_set for ch in raw)

    lower_raw = raw.lower()
    suspicious_words_found = [w for w in SUSPICIOUS_WORDS if w in lower_raw]
    query_params = parse_qsl(parsed.query, keep_blank_values=True) if parsed.query else []

    non_standard_port = int(port is not None and port not in (80, 443))
    double_slash_in_path = int("//" in path)
    has_shortener = int(reg_domain in SHORTENERS or host in SHORTENERS)
    has_punycode = int("xn--" in host)
    mixed_alphabet = _has_mixed_alphabet(raw)
    brand_spoof = _is_brand_spoof(host, reg_domain, main_domain)
    domain_entropy = round(_shannon_entropy(host), 4)
    suspicious_tld = int(suffix.split(".")[-1] in SUSPICIOUS_TLDS if suffix else False)

    return {
        # 22 обязательных признака по таблице Задания 9
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path),
        "dot_count": raw.count("."),
        "subdomain_count": subdomain_count,
        "domain_hyphen_count": host.count("-"),
        "digit_count": digits_url,
        "domain_digit_ratio": round(digits_host / max(1, len(host)), 4),
        "special_char_count": special_char_count,
        "has_ip": is_ip,
        "https": int(parsed.scheme.lower() == "https"),
        "non_standard_port": non_standard_port,
        "has_at": int("@" in raw),
        "double_slash_in_path": double_slash_in_path,
        "has_shortener": has_shortener,
        "suspicious_word_count": len(suspicious_words_found),
        "has_punycode": has_punycode,
        "mixed_alphabet": mixed_alphabet,
        "brand_spoof": brand_spoof,
        "domain_entropy": domain_entropy,
        "suspicious_tld": suspicious_tld,
        "query_param_count": len(query_params),
        # Совместимые признаки
        "query_length": len(query),
        "hyphen_count": raw.count("-"),
        "at_count": raw.count("@"),
        "question_count": raw.count("?"),
        "equal_count": raw.count("="),
        "slash_count": raw.count("/"),
        "digit_ratio": round(digits_url / max(1, len(raw)), 4),
        "special_ratio": round(sum(not c.isalnum() for c in raw) / max(1, len(raw)), 4),
        "has_brand_word": brand_spoof,
        "has_login_word": int(len(suspicious_words_found) > 0),
        "has_redirect": int(bool(re.search(r"redirect|url=|return=|continue=|next=", raw, re.I))),
    }


def explain_url(features: dict) -> list[str]:
    """Формирует человекочитаемые причины подозрительности URL (Задание 9, доп. п.2)."""
    reasons = []
    if features.get("has_ip"):
        reasons.append("Используется IP-адрес вместо доменного имени")
    if features.get("has_punycode"):
        reasons.append("Обнаружена Punycode-кодировка домена (возможная подмена символов xn--)")
    if features.get("mixed_alphabet"):
        reasons.append("Обнаружено смешение кириллицы и латиницы в адресе")
    if features.get("brand_spoof"):
        reasons.append("Домен имитирует известный бренд или банк (тайпосквоттинг)")
    if features.get("has_shortener"):
        reasons.append("Используется сервис сокращения ссылок, скрывающий конечный адрес")
    if features.get("suspicious_tld"):
        reasons.append("Домен находится в зоне повышенного риска (.xyz, .top, .click, .zip и др.)")
    if features.get("has_at"):
        reasons.append("В адресе присутствует символ @, используемый для маскировки хоста")
    if features.get("non_standard_port"):
        reasons.append("Указан нестандартный сетевой порт (не 80 и не 443)")
    if features.get("double_slash_in_path"):
        reasons.append("В пути URL обнаружено двойное косое деление '//' (признак перенаправления)")
    if features.get("suspicious_word_count", 0) > 0:
        reasons.append(f"В адресе присутствуют подозрительные слова (login/verify/secure/bank и др.: {features['suspicious_word_count']})")
    if features.get("subdomain_count", 0) >= 3:
        reasons.append(f"Большое количество поддоменов ({features['subdomain_count']})")
    if features.get("domain_hyphen_count", 0) >= 2:
        reasons.append(f"Несколько дефисов в доменном имени ({features['domain_hyphen_count']})")
    if not features.get("https"):
        reasons.append("Используется незащищённый протокол HTTP (без шифрования HTTPS)")
    return reasons
