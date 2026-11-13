"""Модель пользователя и его профиля (Раздел 4.3 ТЗ, Задание 16 п. 1)."""
from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    store_sms_text = models.BooleanField(default=False)  # по умолчанию выключено (ФТ-2.7 / Задание 16 п. 4)
    preferred_language = models.CharField(max_length=5, default="ru")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Профиль {self.user.username}"
