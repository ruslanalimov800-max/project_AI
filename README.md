# SMS Guard — система обнаружения подозрительных SMS и URL

Python 3.11+, Django 5.2, Django REST Framework, scikit-learn.

## Установка и запуск

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py check
python manage.py runserver
```

Откройте `http://127.0.0.1:8000/`. Для Windows PowerShell используйте `copy .env.example .env`.

## Проверки и модели

```bash
python -m pytest -q
python -m ml.sms_baseline
python -m ml.sms_compare
python -m ml.train_url data/raw/url_dataset.csv --url-column url --label-column label
```

URL-модель обучается только на предоставленном реальном размеченном CSV: `0` — легитимный URL, `1` — фишинг. В репозитории URL-датасет отсутствует, поэтому `ml/artifacts/url_model.joblib` не создаётся искусственно; до обучения приложение честно показывает, что URL-модель недоступна. Не интерпретируйте это как подтверждённую классификацию URL.

## Основные маршруты

- `/` — главная страница.
- `/accounts/register/`, `/accounts/login/` — регистрация и вход.
- `/analyze/` — анализ SMS/URL (требуется вход).
- `/api/v1/analyze/` — POST JSON `{"text":"..."}` (требуется вход).
- `/history/` — история текущего пользователя.
- `/dashboard/` — статистика текущего пользователя.
- `/api/v1/health/` — проверка работоспособности.

## Структура

`config/` — Django; `apps/accounts/`, `apps/analysis/`, `apps/dashboard/` — веб-приложения;
`ml/` — предобработка, признаки, SMS- и URL-модели, правила и риск; `data/` — датасеты;
`templates/`, `static/` — интерфейс; `tests/` — тесты; `docs/` — отчёты ТЗ.

## Ограничения

UCI SMS Spam Collection англоязычная, а класс spam не равен финансовому мошенничеству. Поэтому SMS-вероятность — вспомогательный сигнал, а не гарантия. URL-модель требует отдельного реального размеченного набора и оценки на отложенной выборке. Проверка не выполняет сетевые запросы к URL и не гарантирует безопасность.
