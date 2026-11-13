"""Сериализаторы и валидация входных данных REST API (Задание 14 п. 1, ФТ-2.1, ФТ-3.1)."""
import unicodedata
from rest_framework import serializers

from ml.url_features import validate_url_with_reason


def strip_control_chars(value: str) -> str:
    return "".join(" " if unicodedata.category(ch).startswith("C") and ch not in ("\n", "\r", "\t") else ch for ch in str(value or ""))


class SMSCheckSerializer(serializers.Serializer):
    text = serializers.CharField(allow_blank=True, trim_whitespace=False, required=False, default="")

    def validate_text(self, value):
        cleaned = strip_control_chars(value).strip()
        if not cleaned:
            raise serializers.ValidationError("Текст SMS не может быть пустым.")
        if len(cleaned) > 1000:
            raise serializers.ValidationError("Длина текста SMS не должна превышать 1000 символов.")
        return cleaned


class URLCheckSerializer(serializers.Serializer):
    url = serializers.CharField(allow_blank=True, trim_whitespace=True, required=False, default="")

    def validate_url(self, value):
        cleaned = strip_control_chars(value).strip()
        if not cleaned:
            raise serializers.ValidationError("Поле URL не может быть пустым.")
        if len(cleaned) > 2048:
            raise serializers.ValidationError("Длина URL не должна превышать 2048 символов.")
        ok, err = validate_url_with_reason(cleaned)
        if not ok:
            raise serializers.ValidationError(err)
        return cleaned


# Обратная совместимость для /api/v1/analyze/
AnalyzeSerializer = SMSCheckSerializer
