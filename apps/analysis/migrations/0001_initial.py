from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion

class Migration(migrations.Migration):
    initial = True
    dependencies = [migrations.swappable_dependency(settings.AUTH_USER_MODEL)]
    operations = [
        migrations.CreateModel(name="Check", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("check_type", models.CharField(choices=[("sms","SMS"),("url","URL")], max_length=10)),
            ("input_text", models.TextField(blank=True, null=True)), ("input_masked", models.TextField()),
            ("language", models.CharField(blank=True, max_length=5)), ("category", models.CharField(max_length=32)),
            ("risk_level", models.CharField(choices=[("low","Низкий"),("medium","Средний"),("high","Высокий")], max_length=10)),
            ("risk_score", models.PositiveSmallIntegerField()), ("p_sms", models.FloatField(blank=True, null=True)),
            ("p_url", models.FloatField(blank=True, null=True)), ("explanation", models.JSONField(default=list)),
            ("recommendation", models.TextField()), ("sms_model_version", models.CharField(blank=True, max_length=40)),
            ("url_model_version", models.CharField(blank=True, max_length=40)), ("created_at", models.DateTimeField(auto_now_add=True)),
            ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="checks", to=settings.AUTH_USER_MODEL)),
        ]),
        migrations.CreateModel(name="URLResult", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("url", models.URLField(max_length=2048)), ("p_phishing", models.FloatField(blank=True, null=True)),
            ("label", models.CharField(blank=True, max_length=20)),
            ("check", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="url_results", to="analysis.check")),
        ]),
        migrations.CreateModel(name="DetectedFeature", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("group", models.CharField(max_length=40)), ("phrase", models.CharField(max_length=255)),
            ("check", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="detected_features", to="analysis.check")),
        ]),
        migrations.CreateModel(name="ModelVersion", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("model_type", models.CharField(max_length=20)), ("version", models.CharField(max_length=40)),
            ("metrics", models.JSONField(default=dict)), ("trained_at", models.DateTimeField(auto_now_add=True)),
        ]),
        migrations.AddIndex(model_name="check", index=models.Index(fields=["user","-created_at"], name="analysis_c_user_id_8b6f5a_idx")),
        migrations.AddIndex(model_name="check", index=models.Index(fields=["risk_level","-created_at"], name="analysis_c_risk_le_0c7d6b_idx")),
    ]
