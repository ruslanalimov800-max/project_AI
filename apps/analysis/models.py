from django.conf import settings
from django.db import models

class Check(models.Model):
    CHECK_TYPES = [("sms", "SMS"), ("url", "URL")]
    RISK_LEVELS = [("low", "Низкий"), ("medium", "Средний"), ("high", "Высокий")]
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="checks")
    check_type = models.CharField(max_length=10, choices=CHECK_TYPES)
    input_text = models.TextField(null=True, blank=True)
    input_masked = models.TextField()
    language = models.CharField(max_length=5, blank=True)
    category = models.CharField(max_length=32)
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
        indexes = [models.Index(fields=["user", "-created_at"]), models.Index(fields=["risk_level", "-created_at"])]

class URLResult(models.Model):
    check = models.ForeignKey(Check, on_delete=models.CASCADE, related_name="url_results")
    url = models.URLField(max_length=2048)
    p_phishing = models.FloatField(null=True, blank=True)
    label = models.CharField(max_length=20, blank=True)

class DetectedFeature(models.Model):
    check = models.ForeignKey(Check, on_delete=models.CASCADE, related_name="detected_features")
    group = models.CharField(max_length=40)
    phrase = models.CharField(max_length=255)

class ModelVersion(models.Model):
    model_type = models.CharField(max_length=20)
    version = models.CharField(max_length=40)
    metrics = models.JSONField(default=dict)
    trained_at = models.DateTimeField(auto_now_add=True)
