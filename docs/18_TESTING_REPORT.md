# Отчёт о тестировании и проверке качества (SMS Guard)

## 1. Автоматизированное тестирование
- Инструмент: `pytest` + `pytest-django` + `coverage`
- Результат: **83 теста пройдено (0 ошибок)**, общее покрытие кода `ml/` и `apps/` составило **60%**.
- Покрытые области:
  - Загрузка и очистка датасетов (`tests/test_load_sms.py`, `tests/test_preprocessing.py`)
  - Извлечение признаков SMS, rule-based детектор, URL-признаки, формула риска и совместный анализатор (`tests/test_sms_features.py`, `tests/test_sms_baseline.py`, `tests/test_url_and_risk.py`)
  - Каркас Django, конфигурация `SECRET_KEY`, логирование (`tests/test_django_skeleton.py`)
  - REST API, регистрация, блокировка после 5 неудачных входов, rate limit 30/мин, изоляция чужих проверок (404), приватность `store_sms_text=False`, защита от XSS и SQLi, экспорт CSV (`tests/test_api_and_web.py`)

## 2. Итоговые метрики ML-моделей
- **SMS-модель (`LinearSVM`, порог `0.22`)**:
  - Тест UCI: Accuracy = `0.9883`, Precision = `0.9468`, Recall = `0.9570`, F1 = `0.9519`, F2 = `0.9549`, ROC-AUC = `0.9978`, PR-AUC = `0.9885`
  - RU / KK подмножества (`custom_ru_kk.csv`): Recall `ru` = `0.94`, Recall `kk` = `0.88`
  - Подробности: [08_SMS_MODEL_COMPARISON.md](08_SMS_MODEL_COMPARISON.md)
- **URL-модель (`RandomForest`, пороги `0.35 / 0.60`)**:
  - Тест (без утечки доменов): Accuracy = `0.8002`, Precision = `0.7007`, Recall = `0.7796`, F1 = `0.7380`, ROC-AUC = `0.8894`
  - Подробности: [10_11_URL_MODEL_REPORT.md](10_11_URL_MODEL_REPORT.md)

## 3. Проверка безопасности и надёжности
- `python manage.py check` — без замечаний (`0 silenced`).
- Пароли хэшируются стандартным Django PBKDF2; при `DEBUG=0` без `SECRET_KEY` приложение отказывает в запуске.
- Чувствительные данные в `input_masked` (номера карт, телефоны, суммы) маскируются как `***`.
- При отключённой ML-модели API возвращает корректный статус `503 MODEL_UNAVAILABLE`, при превышении 30 запросов/мин — `429 RATE_LIMIT_EXCEEDED`.
