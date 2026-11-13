"""Извлечение признаков URL для задания 9. Без сетевых запросов к анализируемому адресу."""
import ipaddress
import re
from urllib.parse import urlsplit, unquote

SUSPICIOUS_TLDS = {"zip", "mov", "top", "xyz", "click", "work", "country", "gq", "tk", "ml", "cf"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "cutt.ly", "rb.gy"}
BRAND_WORDS = {"kaspi", "halyk", "paypal", "bank", "sber", "citi", "visa", "mastercard"}

def _normalize_url(value):
    value = str(value or "").strip().strip("()[]{}<>,.;!?\"'")
    if not value:
        return ""
    if not re.match(r"(?i)^https?://", value):
        value = "http://" + value
    return value

def extract_url_features(value):
    """Возвращает словарь числовых/категориальных признаков для одного URL."""
    raw = str(value or "").strip()
    normalized = _normalize_url(raw)
    parsed = urlsplit(normalized)
    host = (parsed.hostname or "").lower().rstrip(".")
    path = unquote(parsed.path or "")
    query = unquote(parsed.query or "")
    try:
        ipaddress.ip_address(host.strip("[]"))
        is_ip = 1
    except ValueError:
        is_ip = 0
    labels = [x for x in host.split(".") if x]
    tld = labels[-1] if len(labels) > 1 else ""
    digits = sum(ch.isdigit() for ch in raw)
    letters = sum(ch.isalpha() for ch in raw)
    return {
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": raw.count("."),
        "hyphen_count": raw.count("-"),
        "at_count": raw.count("@"),
        "question_count": raw.count("?"),
        "equal_count": raw.count("="),
        "slash_count": raw.count("/"),
        "digit_ratio": digits / max(1, len(raw)),
        "special_ratio": sum(not c.isalnum() for c in raw) / max(1, len(raw)),
        "subdomain_count": max(0, len(labels) - 2),
        "https": int(parsed.scheme.lower() == "https"),
        "has_ip": is_ip,
        "has_at": int("@" in raw),
        "has_punycode": int("xn--" in host),
        "has_shortener": int(host in SHORTENERS),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "has_brand_word": int(any(w in host for w in BRAND_WORDS)),
        "has_login_word": int(bool(re.search(r"login|signin|verify|secure|account|auth|вход|кабинет|тексер", raw, re.I))),
        "has_redirect": int(bool(re.search(r"redirect|url=|return=|continue=", raw, re.I))),
    }

def validate_url(value):
    normalized = _normalize_url(value)
    parsed = urlsplit(normalized)
    host = parsed.hostname or ""
    if not host or any(ch.isspace() for ch in str(value)):
        return False
    # Для публичного URL требуется доменная точка либо корректный IP-адрес.
    try:
        ipaddress.ip_address(host.strip("[]"))
        return parsed.scheme in ("http", "https")
    except ValueError:
        return parsed.scheme in ("http", "https") and "." in host and not host.startswith(".") and not host.endswith(".")
