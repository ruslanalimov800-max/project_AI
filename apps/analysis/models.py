"""Модели базы данных проверок (Раздел 4.3 ТЗ, Задание 16 п. 1)."""
from django.conf import settings
from django.db import models


class Check(models.Model):
    CHECK_TYPES = [("sms", "SMS"), ("url", "URL"), ("SMS", "SMS"), ("URL", "URL")]
    RISK_LEVELS = [("low", "Низкий"), ("medium", "Средний"), ("high", "Высокий")]
    CATEGORIES = [
        ("SAFE", "Безопасное"),
        ("SUSPICIOUS", "Подозрительное"),
        ("FRAUDULENT", "Мошенническое"),
        ("DANGEROUS", "Опасная ссылка"),
        ("likely_safe", "Безопасное"),
        ("suspicious", "Подозрительное"),
        ("phishing", "Мошенническое"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="checks")
    check_type = models.CharField(max_length=10, choices=CHECK_TYPES)
    input_text = models.TextField(null=True, blank=True)
    input_masked = models.TextField()
    language = models.CharField(max_length=5, blank=True)
    category = models.CharField(max_length=32, choices=CATEGORIES)
    risk_level = models.CharField(max_length=10, choices=RISK_LEVELS)
    risk_score = models.PositiveSmallIntegerField()
    p_sms = models.FloatField(null=True, blank=True)
    p_url = models.FloatField(null=True, blank=True)
    explanation = models.JSONField(default=list)
    recommendation = models.TextField()
    sms_model_version = models.CharField(max_length=40, blank=True)
    url_model_version = models.CharField(max_length=40, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["category"]),
            models.Index(fields=["risk_level", "-created_at"]),
        ]


class URLResult(models.Model):
    analysis_check = models.ForeignKey(Check, on_delete=models.CASCADE, related_name="url_results")
    url = models.URLField(max_length=2048)
    domain = models.CharField(max_length=255, blank=True, default="")
    p_url = models.FloatField(null=True, blank=True)
    p_phishing = models.FloatField(null=True, blank=True)
    category = models.CharField(max_length=20, blank=True, default="SAFE")
    label = models.CharField(max_length=20, blank=True)
    features_json = models.JSONField(default=dict, blank=True)


# Псевдоним UrlResult по ТЗ (Задание 16 п. 1)
UrlResult = URLResult


class DetectedFeature(models.Model):
    analysis_check = models.ForeignKey(Check, on_delete=models.CASCADE, related_name="detected_features")
    code = models.CharField(max_length=40, blank=True, default="")
    group = models.CharField(max_length=40)
    title = models.CharField(max_length=255, blank=True, default="")
    phrase = models.CharField(max_length=255, blank=True, default="")
    weight = models.FloatField(default=0.0)


class ModelVersion(models.Model):
    model_type = models.CharField(max_length=20)      # "sms" | "url"
    version = models.CharField(max_length=40)
    algorithm = models.CharField(max_length=80, blank=True, default="")
    metrics_json = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    threshold = models.FloatField(default=0.5)
    trained_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
