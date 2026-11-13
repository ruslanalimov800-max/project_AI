# SMS Guard — интеллектуальная система обнаружения мошеннических SMS и подозрительных URL

Учебный проект (дипломная/курсовая работа) на базе Django + DRF + scikit-learn для комплексной оценки риска SMS-сообщений и ссылок.

## Возможности
- **Анализ SMS**: классификация текста (TF-IDF + LinearSVM), rule-based детектор социальной инженерии (8 групп признаков для `ru`, `kk`, `en`), извлечение до 3 URL.
- **Анализ URL**: извлечение 22+ лексических и структурных признаков (IP-адрес, Punycode, кириллические/смешанные домены, схожесть с брендами, сокращатели, энтропия и др.) и классификация (`RandomForest`).
- **Скоринг риска (0–100)**: объединение `P_sms`, `P_url_max` и правил социальной инженерии по `risk_config.yaml` в категории `SAFE` (0–34), `SUSPICIOUS` (35–69), `FRAUDULENT`/`DANGEROUS` (70–100) с объяснением причин на русском языке.
- **Веб-интерфейс и REST API**: авторизация, защита от перебора паролей (5 попыток -> 60с), ограничение частоты запросов (30/мин), история проверок с фильтрами и экспортом CSV, статистика, настройки приватности (`store_sms_text`) и маскирование карт/телефонов.

## Быстрый старт (Windows PowerShell)

```powershell
# 1. Активация виртуального окружения и установка зависимостей
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Миграции и демо-данные (demo_user / DemoPass123, admin_demo / AdminPass123)
python manage.py migrate
python manage.py seed_demo

# 3. Запуск сервера
python manage.py runserver
```

## Обучение моделей (при необходимости переобучить)

```powershell
python -m ml.data.create_custom_ru_kk
python -m ml.data.prepare_sms
python -m ml.data.prepare_url
python -m ml.sms_compare
python -m ml.url_model
```

## Основные маршруты
- Веб: `/`, `/login/`, `/register/`, `/profile/`, `/check/sms/`, `/check/url/`, `/history/`, `/history/export/`, `/stats/`, `/dashboard/`
- API: `/api/v1/health/`, `/api/v1/auth/register/`, `/api/v1/auth/login/`, `/api/v1/auth/logout/`, `/api/v1/check/sms/`, `/api/v1/check/url/`, `/api/v1/checks/`, `/api/v1/checks/<id>/`, `/api/v1/stats/summary/`

## Тестирование

```powershell
python -m pytest -q
```
