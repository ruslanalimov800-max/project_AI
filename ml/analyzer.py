"""Совместный анализ SMS и до трёх URL."""
from dataclasses import dataclass, asdict
import re
from ml.rules import analyze_text_rules
from ml.risk import calculate_risk
from ml.url_features import validate_url
from ml.url_model import predict_url
from ml.sms_compare import predict_sms_best

URL_RE = re.compile(r"(?i)(?:https?://|www\.)[^\s<>]+|\b(?:[a-z0-9-]+\.)+(?:com|org|net|kz|ru|io|gov|edu|top|xyz)(?:/[^\s<>]*)?")

@dataclass
class AnalysisResult:
    check_type: str
    input_text: str
    urls: list
    language: str
    category: str
    risk_level: str
    risk_score: int
    p_sms: float | None
    p_url_max: float | None
    explanation: list
    detected_features: list
    url_predictions: list
    recommendation: str
    url_model_available: bool

def detect_language(text):
    value = str(text or "")
    if re.search(r"[әғқңөұүһі]", value, re.I): return "kk"
    if re.search(r"[а-яё]", value, re.I): return "ru"
    return "en"

def extract_urls(text):
    found = []
    for item in URL_RE.findall(str(text or "")):
        item = item.rstrip(".,;:!?)]}")
        if item and validate_url(item) and item not in found:
            found.append(item)
        if len(found) == 3: break
    return found

def analyze_message(text):
    text = str(text or "").strip()
    if not text: raise ValueError("Введите SMS или URL для анализа.")
    urls = extract_urls(text)
    sms_result = predict_sms_best(text) if not (len(text.split()) == 1 and validate_url(text)) else None
    url_results = [predict_url(url) for url in urls]
    available = bool(url_results) and all(x is not None for x in url_results)
    probabilities = [x["probability"] for x in url_results if x is not None]
    url_predictions = [
        {"url": url, "probability": item["probability"] if item else None,
         "label": item["label"] if item else None}
        for url, item in zip(urls, url_results)
    ]
    p_url = max(probabilities) if probabilities else None
    rules = analyze_text_rules(text)
    risk = calculate_risk(sms_result["probability"] if sms_result else None, p_url, rules["group_count"])
    explanation = [f"Сработали группы правил: {', '.join(rules['groups'])}." ] if rules["groups"] else []
    if sms_result: explanation.append(f"Модель SMS: вероятность спама {sms_result['probability']:.3f}.")
    if urls:
        explanation.append("Обнаружено URL: " + ", ".join(urls))
        explanation.append("Модель URL пока не обучена на реальном датасете." if not available else f"Максимальная вероятность фишинга URL: {p_url:.3f}.")
    if not explanation: explanation.append("Явные индикаторы из словаря правил не обнаружены.")
    category = "phishing" if risk["risk_level"] == "high" else "suspicious" if risk["risk_level"] == "medium" else "likely_safe"
    recommendation = {
        "high": "Не переходите по ссылке и не передавайте коды, пароли или платёжные данные. Проверьте сообщение через официальный канал.",
        "medium": "Не выполняйте срочные требования из сообщения. Проверьте отправителя и адрес сайта независимо.",
        "low": "Явных признаков риска не выявлено. Это не гарантирует безопасность сообщения или ссылки."
    }[risk["risk_level"]]
    return asdict(AnalysisResult(
        check_type="url" if len(text.split()) == 1 and validate_url(text) else "sms",
        input_text=text, urls=urls, language=detect_language(text), category=category,
        risk_level=risk["risk_level"], risk_score=risk["risk_score"],
        p_sms=sms_result["probability"] if sms_result else None, p_url_max=p_url,
        explanation=explanation,
        detected_features=[{"group": group, "phrases": phrases} for group, phrases in rules["groups"].items()],
        url_predictions=url_predictions,
        recommendation=recommendation, url_model_available=available
    ))
