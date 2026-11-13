"""Задание 12: Совместный интеллектуальный анализ SMS и URL."""
from dataclasses import asdict, dataclass
import re
from urllib.parse import urlsplit

from ml.predictor import get_predictor
from ml.risk import compute_risk
from ml.rules import analyze_text_rules
from ml.url_features import SHORTENERS, _normalize_url, extract_url_features, validate_url, validate_url_with_reason

_SHORTENER_PATTERN = "|".join(re.escape(s) for s in sorted(SHORTENERS, key=len, reverse=True))
# Регулярное выражение извлекает схемы http/https, адреса с www., сокращатели (bit.ly/...), домены с латиницей и кириллицей (Задание 12 п.1)
URL_RE = re.compile(
    rf"(?i)(?:https?://|www\.)[^\s<>\"']+"
    rf"|\b(?:{_SHORTENER_PATTERN})/[^\s<>\"']+"
    rf"|\b(?:[a-z0-9а-яёәғқңөұүһі-]+\.)+(?:com|org|net|kz|ru|io|gov|edu|top|xyz|click|zip|test|example|invalid|рф|қаз)\b(?:/[^\s<>\"']*)?"
)


@dataclass
class AnalysisResult:
    type: str                        # "SMS" | "URL"
    check_type: str                  # "sms" | "url" (совместимость)
    input_text: str
    language: str                    # "ru" | "kk" | "en" | "other"
    language_warning: str | None     # предупреждение ФТ-2.8 для неанглоязычных текстов
    category: str                    # "SAFE" | "SUSPICIOUS" | "FRAUDULENT" | "DANGEROUS"
    legacy_category: str             # "likely_safe" | "suspicious" | "phishing"
    risk_level: str                  # "low" | "medium" | "high"
    risk_level_code: str             # "LOW" | "MEDIUM" | "HIGH"
    risk_score: int                  # 0..100
    p_sms: float | None
    p_url_max: float | None
    urls: list[dict]                 # [{"url": ..., "domain": ..., "p_url": ..., "category": ..., "features": ...}]
    url_strings: list[str]
    features: list[str]              # ["URGENCY", "ACCOUNT_BLOCK", ...]
    detected_features: list[dict]
    url_predictions: list[dict]
    explanation: list[str]
    recommendation: str
    disclaimer: str
    model_versions: dict
    url_model_available: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        # Поле urls для обратной совместимости в тестах ml/analyzer.py может проверяться на список URL
        return d


def detect_language(text: str) -> str:
    """Определение языка текста: ru, kk, en, other (ФТ-2.2)."""
    value = str(text or "")
    if re.search(r"[әғқңөұүһі]", value, re.I):
        return "kk"
    if re.search(r"[а-яё]", value, re.I):
        return "ru"
    if re.search(r"[a-z]", value, re.I):
        return "en"
    return "other"


def extract_urls(text: str) -> list[str]:
    """Извлекает до 3 уникальных корректных URL из текста SMS (ФТ-2.7, Задание 12 п.1)."""
    found: list[str] = []
    for match in URL_RE.finditer(str(text or "")):
        item = match.group(0).rstrip(".,;:!?)]}\"'»")
        if item and validate_url(item) and item not in found:
            found.append(item)
        if len(found) == 3:
            break
    return found


def _extract_domain(url: str) -> str:
    try:
        return (urlsplit(_normalize_url(url)).hostname or "").lower()
    except ValueError:
        return ""


def analyze_url_struct(url: str) -> AnalysisResult:
    """Отдельная проверка ссылки без текста SMS (Задание 12 п.5)."""
    raw = str(url or "").strip()
    ok, err = validate_url_with_reason(raw)
    if not ok:
        raise ValueError(err)

    predictor = get_predictor()
    url_pred = predictor.predict_url(raw)
    feats = url_pred["features"] if url_pred else extract_url_features(raw)
    p_url = url_pred["p_url"] if url_pred else None
    url_cat = url_pred["category"] if url_pred else "SUSPICIOUS"
    domain = _extract_domain(raw)

    risk = compute_risk(
        p_sms=None,
        p_url_max=p_url,
        se_groups=[],
        url_flags=[feats],
        is_url_only=True,
    )

    url_entry = {
        "url": raw,
        "domain": domain,
        "p_url": p_url,
        "probability": p_url,
        "category": url_cat if p_url is not None else risk.category,
        "label": 1 if (url_cat == "DANGEROUS" or risk.risk_level == "HIGH") else 0,
        "features": feats,
    }

    feature_codes = []
    if feats.get("has_ip"):
        feature_codes.append("IP_ADDRESS")
    if feats.get("has_punycode"):
        feature_codes.append("PUNYCODE")
    if feats.get("brand_spoof"):
        feature_codes.append("BRAND_SPOOF")
    if feats.get("has_shortener"):
        feature_codes.append("URL_SHORTENER")
    if feats.get("suspicious_word_count", 0) > 0:
        feature_codes.append("SUSPICIOUS_KEYWORDS")
    if feats.get("suspicious_tld"):
        feature_codes.append("SUSPICIOUS_TLD")

    return AnalysisResult(
        type="URL",
        check_type="url",
        input_text=raw,
        language="en",
        language_warning=None,
        category=risk.category,
        legacy_category=risk.legacy_category,
        risk_level=risk.risk_level_lower,
        risk_level_code=risk.risk_level,
        risk_score=risk.risk_score,
        p_sms=None,
        p_url_max=p_url,
        urls=[url_entry],
        url_strings=[raw],
        features=feature_codes,
        detected_features=[
            {"group": code, "code": code, "title": code, "weight": 20, "phrases": [raw]}
            for code in feature_codes
        ],
        url_predictions=[url_entry],
        explanation=risk.reasons,
        recommendation=risk.recommendation,
        disclaimer=risk.disclaimer,
        model_versions={"sms": predictor.sms_version, "url": predictor.url_version},
        url_model_available=predictor.url_available,
    )


def analyze_sms_struct(text: str) -> AnalysisResult:
    """Полный совместный анализ SMS и найденных в нём до 3 URL (Задание 12)."""
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Текст SMS не может быть пустым.")

    # Если передана ровно одна валидная ссылка без текста — направляем в analyze_url_struct
    if len(raw.split()) == 1 and validate_url(raw):
        return analyze_url_struct(raw)

    predictor = get_predictor()
    lang = detect_language(raw)
    language_warning = None
    if lang in ("ru", "kk", "other"):
        language_warning = (
            f"Язык сообщения ({lang}) не представлен в основном обучающем наборе UCI SMS (en): "
            "ML-оценка текста имеет сниженную надёжность; основную роль играют словарь правил и проверка URL."
        )

    found_urls = extract_urls(raw)
    sms_pred = predictor.predict_sms(raw)
    p_sms_raw = sms_pred["probability"] if sms_pred else None

    rules = analyze_text_rules(raw)
    se_group_codes = list(rules["groups"].keys())
    if found_urls and "LINK_REQUEST" not in se_group_codes and len(se_group_codes) > 0:
        se_group_codes_with_link = se_group_codes + ["LINK_REQUEST"]
    else:
        se_group_codes_with_link = list(se_group_codes)

    # Для неанглоязычных текстов без сработавших правил социальной инженерии и без ссылок
    # демпфируем шум англоязычной TF-IDF модели, чтобы обычные бытовые фразы на ru/kk не давали ложную тревогу
    p_sms_effective = p_sms_raw
    if p_sms_raw is not None and lang in ("ru", "kk") and len(se_group_codes) == 0 and not found_urls:
        p_sms_effective = round(min(p_sms_raw * 0.42, 0.32), 4)

    url_entries = []
    url_flags_list = []
    url_probs = []
    for u in found_urls:
        up = predictor.predict_url(u)
        feats = up["features"] if up else extract_url_features(u)
        url_flags_list.append(feats)
        p_u = up["p_url"] if up else None
        if p_u is not None:
            url_probs.append(p_u)
        url_entries.append({
            "url": u,
            "domain": _extract_domain(u),
            "p_url": p_u,
            "probability": p_u,
            "category": up["category"] if up else "SUSPICIOUS",
            "label": up["label"] if up else None,
            "features": feats,
        })

    p_url_max = max(url_probs) if url_probs else None

    risk = compute_risk(
        p_sms=p_sms_effective,
        p_url_max=p_url_max,
        se_groups=se_group_codes,
        url_flags=url_flags_list,
        is_url_only=False,
    )

    explanations = list(risk.reasons)
    if language_warning and language_warning not in explanations and len(explanations) < 6:
        explanations.append(language_warning)

    detected_list = [
        {
            "group": d["code"],
            "code": d["code"],
            "title": d["title"],
            "weight": d["weight"],
            "phrases": d["phrases"],
        }
        for d in rules["details"]
    ]

    return AnalysisResult(
        type="SMS",
        check_type="sms",
        input_text=raw,
        language=lang,
        language_warning=language_warning,
        category=risk.category,
        legacy_category=risk.legacy_category,
        risk_level=risk.risk_level_lower,
        risk_level_code=risk.risk_level,
        risk_score=risk.risk_score,
        p_sms=p_sms_effective,
        p_url_max=p_url_max,
        urls=url_entries,
        url_strings=found_urls,
        features=se_group_codes_with_link,
        detected_features=detected_list,
        url_predictions=url_entries,
        explanation=explanations,
        recommendation=risk.recommendation,
        disclaimer=risk.disclaimer,
        model_versions={"sms": predictor.sms_version, "url": predictor.url_version},
        url_model_available=predictor.url_available,
    )


def analyze_sms(text: str) -> AnalysisResult:
    return analyze_sms_struct(text)


def analyze_url(url: str) -> AnalysisResult:
    return analyze_url_struct(url)


def analyze_message(text: str) -> dict:
    """Словарное представление результата (для обратной совместимости с существующими тестами и views)."""
    res = analyze_sms_struct(text)
    d = asdict(res)
    # В существующих тестах test_url_and_risk.py ожидается, что result["urls"] — список строк URL,
    # а подробные словари лежат в url_predictions / url_details.
    d["url_details"] = res.urls
    d["urls"] = res.url_strings
    return d
