"""Сервисный слой анализа SMS и URL (Задание 12: apps/analysis/services.py)."""
from ml.analyzer import (
    AnalysisResult,
    analyze_message,
    analyze_sms,
    analyze_sms_struct,
    analyze_url,
    analyze_url_struct,
    detect_language,
    extract_urls,
)

__all__ = [
    "AnalysisResult",
    "analyze_sms",
    "analyze_url",
    "analyze_sms_struct",
    "analyze_url_struct",
    "analyze_message",
    "detect_language",
    "extract_urls",
]
