"""Задание 16 п. 8: Команда python manage.py seed_demo.
Создаёт демонстрационного пользователя demo_user (и администратора admin_demo)
и 30 проверок из искусственных сообщений и ссылок.
"""
from datetime import timedelta
import json
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.accounts.models import UserProfile
from apps.analysis.models import Check, ModelVersion
from apps.analysis.views import _save_check
from ml.analyzer import analyze_message, analyze_url_struct
from ml.predictor import SMS_META_PATH, URL_META_PATH

User = get_user_model()

DEMO_MESSAGES_30 = [
    # 13 обязательных сообщений из раздела 8.3 ТЗ
    "Привет! Встречаемся завтра в 18:00 у входа в библиотеку.",
    "Your order #4821 has been shipped and will arrive on Monday.",
    "Сәлем! Ертең сағат 10:00-де кездесеміз бе?",
    "Ваш код подтверждения: 4821. Никому не сообщайте этот код.",
    "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login",
    "URGENT: Your account will be suspended in 24 hours. Verify now at http://192.0.2.15/account-update",
    "Мама, у меня проблемы, срочно переведи 50 000 на этот номер, потом всё объясню. Никому не говори.",
    "Congratulations! You won a $1000 gift card. Claim your prize: bit.ly/prize-claim-test",
    "Сотрудник службы безопасности банка: назовите код из SMS для отмены подозрительной операции.",
    "Картаңыз бұғатталды. Жедел түрде мына сілтеме арқылы растаңыз: http://bank-kz-verify.test",
    "Ваша посылка ожидает оплаты таможенного сбора 350 ₸: http://post-delivery-pay.example",
    "https://www.example.com/about",
    "http://xn--exmple-cua.test/secure/login?verify=1",
    # Ещё 17 искусственных сообщений и URL для полного набора из 30 проверок
    "Напоминаем: завтра в 09:00 состоится консультация по дипломному проектированию в ауд. 214.",
    "Служба безопасности Нацбанка: зафиксирована подозрительная операция. Срочно переведите деньги на безопасный счёт!",
    "Құттықтаймыз! Сіз 300 000 тг сыйлық ұттыңыз! Комиссия төлеңіз: http://utys-bonus.test/claim",
    "https://kaspi.kz/shop/",
    "http://halyk-security-update.example:8080/auth/password",
    "Курьер привезёт заказ через 20 минут, пожалуйста, будьте на связи.",
    "Внимание! Ваш аккаунт будет заблокирован сегодня. Срочно введите пароль на https://account-lock.example",
    "Анашым, мен сабақтан шықтым, жарты сағатта үйде боламын.",
    "Шұғыл! Шотыңыз бұғатталды, дереу sms кодын беріңіз және операторға кодты айтыңыз!",
    "http://192.0.2.99/bank/verify-card?cvv=1",
    "https://egov.kz/cms/ru",
    "Вы стали победителем акции! Заберите выигрыш 150 000 тг по ссылке: http://prize-kz.invalid/win",
    "Дәрігерге жазылуыңыз жұма күні сағат 11:00-ге расталды.",
    "Полиция: ваш родственник попал в ДТП, срочно сделайте перевод 100 000 тг следователю, строго конфиденциально!",
    "http://kasp1-kz.test/secure/update-account",
    "Please review the meeting agenda for tomorrow morning.",
    "Налоговая служба: оплатите задолженность в течение 24 часов по ссылке http://tax-pay-kz.test",
]


class Command(BaseCommand):
    help = "Создаёт демонстрационного пользователя demo_user и 30 проверок из искусственных сообщений (Задание 16 п. 8)"

    def handle(self, *args, **options):
        user, created = User.objects.get_or_create(
            username="demo_user",
            defaults={"email": "demo_user@example.com"},
        )
        user.set_password("DemoPass123")
        user.save()
        UserProfile.objects.get_or_create(user=user)

        admin, _ = User.objects.get_or_create(
            username="admin_demo",
            defaults={"email": "admin@example.com", "is_staff": True, "is_superuser": True},
        )
        admin.is_staff = True
        admin.is_superuser = True
        admin.set_password("AdminPass123")
        admin.save()
        UserProfile.objects.get_or_create(user=admin)

        # Заполняем справочную таблицу ModelVersion из метаданных моделей
        for meta_path in (SMS_META_PATH, URL_META_PATH):
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                ModelVersion.objects.update_or_create(
                    model_type=meta.get("model_type", "sms"),
                    version=meta.get("version", "1.0.0"),
                    defaults={
                        "algorithm": meta.get("algorithm", ""),
                        "metrics_json": meta.get("test_metrics", {}),
                        "metrics": meta.get("test_metrics", {}),
                        "threshold": float(meta.get("threshold", 0.5)),
                        "is_active": True,
                    },
                )

        # Пересоздаём демо-проверки для demo_user, чтобы их было ровно 30 и они были распределены по дням
        Check.objects.filter(user=user).delete()
        now = timezone.now()
        for idx, text in enumerate(DEMO_MESSAGES_30):
            if text.startswith("http") and " " not in text:
                res = analyze_url_struct(text).to_dict()
                res["url_details"] = analyze_url_struct(text).urls
            else:
                res = analyze_message(text)
            check = _save_check(user, res)
            # Распределяем даты проверок за последние 15 дней для наглядного графика на Dashboard
            offset_days = (29 - idx) // 2
            Check.objects.filter(pk=check.pk).update(created_at=now - timedelta(days=offset_days, hours=idx % 6))

        total = Check.objects.filter(user=user).count()
        self.stdout.write(self.style.SUCCESS(
            f"Демо-данные успешно созданы: пользователь demo_user (пароль DemoPass123), администратор admin_demo (пароль AdminPass123), проверок: {total}."
        ))
