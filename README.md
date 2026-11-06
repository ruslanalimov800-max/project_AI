# SMS Guard — интеллектуальная система обнаружения мошеннических SMS и ссылок

Веб-приложение на Django (Python 3.11+). Выполнены **задания 1–7**: анализ предметной области, каркас проекта, данные SMS,
предобработка, EDA, признаки TF-IDF и baseline-модель. Остальные задания (8–18) намеренно не реализованы.

## Запуск (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python manage.py check
python manage.py runserver
```

Откройте http://127.0.0.1:8000/ (главная-заглушка) и http://127.0.0.1:8000/api/v1/health/ — ответ `{"status": "ok"}`.
Linux/macOS: `source .venv/bin/activate` и `cp .env.example .env`.
Файл `.env` в репозиторий не входит; при `DEBUG=0` обязателен `SECRET_KEY`.

## Воспроизведение заданий 3–7

```powershell
python -m ml.data.prepare_sms
python -m ml.sms_baseline
python -m pytest -q
```

Команды выполняются из корня проекта по порядку: подготовка данных (`data/processed/`, отчёт `docs/04_PREPROCESSING_REPORT.md`) →
обучение baseline (`ml/artifacts/sms_baseline.joblib`) → тесты. Ноутбуки `notebooks/01_eda_sms.ipynb` и
`notebooks/02_sms_features_baseline.ipynb` содержат выполненные ячейки; пересоздают графики и документы заданий 5–7 при перезапуске.
Для новых экспериментов используйте сохранённое разбиение `data/processed/sms_split.json` и не подбирайте параметры по тесту.

## Структура

```
config/            настройки Django, urls.py, wsgi.py
apps/accounts/     регистрация, вход, профиль (задания 10+)
apps/analysis/     главная, health; позднее — сервис анализа и API
apps/dashboard/    статистика (задание 15)
ml/                preprocessing.py, sms_features.py, sms_baseline.py, data/ (load_sms.py, prepare_sms.py), artifacts/
notebooks/         01_eda_sms.ipynb, 02_sms_features_baseline.ipynb
data/raw|processed исходные и обработанные данные, индексы разбиения
templates/ static/ HTML-шаблоны и статика
tests/             pytest
docs/              отчёты по заданиям, паспорт набора, диаграмма Use Case
```

## Документы

| Задание | Документ |
|---|---|
| 1 | `docs/01_domain_analysis.md`, `docs/diagrams/use_case.puml` |
| 3 | `docs/SMS_DATASET_PASSPORT.md` (в пункте «Дата загрузки» нужно указать фактическую дату скачивания) |
| 4 | `docs/04_PREPROCESSING_REPORT.md` |
| 5 | `docs/05_EDA_FINDINGS.md`, `docs/figures/` |
| 6 | `docs/06_TFIDF_FEATURES.md` |
| 7 | `docs/07_BASELINE_REPORT.md` |

## Ограничения

Обучающий набор UCI SMS Spam Collection англоязычный, а «spam» не равен финансовому мошенничеству: метрики на нём не доказывают
качество для русского и казахского языков. Оценка носит вероятностный характер и не гарантирует безопасность.

Задания 8–18 (сравнение моделей, URL-модуль, оценка риска, API, интерфейс, история, Dashboard) пока не реализованы.
