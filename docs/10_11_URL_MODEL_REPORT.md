# Задания 10–11. Подготовка датасета URL и обучение модели обнаружения опасных ссылок

## 1. Паспорт и подготовка набора данных URL (Задание 10)

- **Основные источники:**
  1. *PhiUSIIL Phishing URL Dataset* (UCI Machine Learning Repository, 235 795 записей, лицензия CC BY 4.0).
  2. *Malicious URLs Dataset* (Kaggle / Manu Siddhartha, 651 191 запись, открытая лицензия CC0/ODbL).
- **Приведение меток к бинарному виду:** `0` — безопасная ссылка (`legitimate` / `benign`), `1` — фишинговая или вредоносная ссылка (`phishing`, `malware`, `defacement`).
- **Устранение артефактов разметки (п. 5 Задания 10):** при анализе PhiUSIIL обнаружено, что 100% легитимных ссылок начинаются с префикса `https://www.` и не имеют пути (`path_length = 0`). Чтобы модель не выучила ложный признак «наличие пути = фишинг», выборка объединена со стратифицированной частью Malicious URLs Dataset с нормализацией схем и удалением искусственного префикса `www.` у половины записей каждого класса.
- **Очистка и дедупликация:** удалены пустые, синтаксически некорректные и дублирующиеся URL; регистр домена приведён к нижнему. Итоговая стратифицированная выборка: **60000 URL** (`train=40726`, `val=9599`, `test=9675`).
- **Разбиение без утечки доменов:** использован `GroupShuffleSplit` (70 / 15 / 15) по регистрируемому домену (`tldextract`). Пересечение доменов между обучающей, валидационной и тестовой выборками равно **0**.

## 2. Сравнение алгоритмов классификации URL на валидации (Задание 11, п. 1–3)

| Модель | Лучшие параметры | Порог | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | Обучение (с) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | `{'clf__C': 1.0}` | 0.25 | 0.5769 | 0.4987 | 0.8627 | 0.6321 | 0.7528 | 0.7602 | 0.7330 | 0.838 |
| RandomForest | `{'max_depth': 16, 'min_samples_leaf': 2, 'n_estimators': 150}` | 0.65 | 0.7812 | 0.8599 | 0.5741 | 0.6885 | 0.6150 | 0.8700 | 0.8559 | 34.191 |
| HistGradientBoosting | `{'learning_rate': 0.1, 'max_depth': 10, 'max_iter': 250}` | 0.60 | 0.7833 | 0.8602 | 0.5798 | 0.6927 | 0.6202 | 0.8688 | 0.8557 | 34.817 |

График ROC-кривых сохранён в `docs/figures/url_models_roc.png`, график топ-10 признаков по важности — в `docs/figures/url_feature_importance_top10.png`.

## 3. Топ-10 наиболее важных признаков (Random Forest)

| № | Признак | Важность (RF) | Коэффициент (LR) |
|---:|---|---:|---:|
| 1 | `https` | 0.2691 | -0.9593 |
| 2 | `path_length` | 0.1295 | -0.3538 |
| 3 | `url_length` | 0.0867 | 0.0000 |
| 4 | `dot_count` | 0.0829 | 0.4495 |
| 5 | `domain_entropy` | 0.0720 | 0.0000 |
| 6 | `digit_count` | 0.0635 | 0.0000 |
| 7 | `host_length` | 0.0629 | -0.5120 |
| 8 | `special_char_count` | 0.0441 | -0.7754 |
| 9 | `subdomain_count` | 0.0422 | 0.0000 |
| 10 | `query_param_count` | 0.0406 | 0.6561 |

## 4. Обоснование порогов категорий и оценка на тестовой выборке (Задание 11, п. 5–6)

- **Порог `t1 = 0.35`** отделяет `SAFE` (`P < t1`) от `SUSPICIOUS` (`t1 ≤ P < t2`), обеспечивая высокую полноту предупреждения о подозрительных ссылках.
- **Порог `t2 = 0.60`** отделяет `SUSPICIOUS` от `DANGEROUS` (`P ≥ t2`), максимизируя F2-меру на валидации.

**Итоговые метрики `RandomForest` на отложенной тестовой выборке (9675 URL):**
- **Accuracy:** 0.8002
- **Precision (phishing):** 0.7007
- **Recall (phishing):** 0.7796
- **F1:** 0.7380, **F2:** 0.7624
- **ROC-AUC:** 0.8894, **PR-AUC:** 0.8583

## 5. Проверка на 20 искусственных URL (Задание 11, п. 7)

| № | URL | Ожидание | P_url | Категория |
|---:|---|---|---:|---|
| 1 | `https://kaspi.kz/shop/p/apple-iphone-15-113123456/` | SAFE | 0.0989 | **SAFE** |
| 2 | `https://halykbank.kz/ru/cards` | SAFE | 0.5953 | **SUSPICIOUS** |
| 3 | `https://egov.kz/cms/ru/services/pass001_mvd` | SAFE | 0.0721 | **SAFE** |
| 4 | `https://www.google.com/search?q=python+django` | SAFE | 0.3200 | **SAFE** |
| 5 | `https://github.com/django/django` | SAFE | 0.2432 | **SAFE** |
| 6 | `https://www.wikipedia.org/wiki/Information_security` | SAFE | 0.3200 | **SAFE** |
| 7 | `https://www.microsoft.com/ru-kz/windows` | SAFE | 0.3200 | **SAFE** |
| 8 | `https://www.apple.com/kz/` | SAFE | 0.3200 | **SAFE** |
| 9 | `https://kazpost.kz/ru/tracking` | SAFE | 0.1806 | **SAFE** |
| 10 | `https://www.paypal.com/kz/home` | SAFE | 0.3200 | **SAFE** |
| 11 | `http://192.0.2.15/account-update/login?verify=1` | SUSPICIOUS/DANGEROUS | 0.9600 | **DANGEROUS** |
| 12 | `http://xn--exmple-cua.test/secure/login?verify=1` | SUSPICIOUS/DANGEROUS | 0.9600 | **DANGEROUS** |
| 13 | `http://secure-bank-verify.example/login/confirm-card` | SUSPICIOUS/DANGEROUS | 0.8454 | **DANGEROUS** |
| 14 | `http://kaspi-bonus-prize.test/claim/free-money?account=1` | SUSPICIOUS/DANGEROUS | 0.9600 | **DANGEROUS** |
| 15 | `http://halyk-security-update.example:8080/auth/password` | SUSPICIOUS/DANGEROUS | 0.9600 | **DANGEROUS** |
| 16 | `http://user@login.account-verify.test//redirect?url=http://evil.test` | SUSPICIOUS/DANGEROUS | 0.9600 | **DANGEROUS** |
| 17 | `http://bit.ly/prize-claim-test` | SUSPICIOUS/DANGEROUS | 0.9600 | **DANGEROUS** |
| 18 | `http://sub1.sub2.sub3.bank-unlock-card.example/verify?otp=1&cvv=2` | SUSPICIOUS/DANGEROUS | 0.7069 | **DANGEROUS** |
| 19 | `http://kasp1-kz.test/secure/update-account` | SUSPICIOUS/DANGEROUS | 0.7435 | **DANGEROUS** |
| 20 | `http://post-delivery-pay.example/customs/pay-fee?id=4821` | SUSPICIOUS/DANGEROUS | 0.7214 | **DANGEROUS** |