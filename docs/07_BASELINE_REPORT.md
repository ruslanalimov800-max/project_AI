# Задание 7. Baseline: Logistic Regression

Документ создаётся ноутбуком `notebooks/02_sms_features_baseline.ipynb`; код — `ml/sms_baseline.py`. Обучение — на train (3595 SMS), метрики — на validation (771 SMS). **Тестовая выборка не использовалась.** Модель: `LogisticRegression(class_weight="balanced", max_iter=1000)` в одном `Pipeline` с признаками задания 6; порог 0,50. Артефакт: `ml/artifacts/sms_baseline.joblib`.

## Метрики на валидации (класс «мошенничество» = spam)

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.9870 | 0.9663 | 0.9247 | 0.9451 |

## Матрица ошибок (абсолютные числа)

| | предсказано ham | предсказано spam |
|---|---:|---:|
| **истинно ham** | TN = 675 | FP = 3 |
| **истинно spam** | FN = 7 | TP = 86 |

График: `docs/figures/sms_baseline_confusion.png`.

## Разбор ошибок

На валидации 7 FN и 3 FP. Показаны 5 FN и 3 FP + 2 «почти FP». Реальные ошибки не выдумываются: «почти FP» — верно классифицированные ham с наибольшей вероятностью спама, они помечены отдельно. Причины — гипотезы по содержанию, а не результат анализа вкладов признаков.

| Тип | P(spam) | Сообщение | Возможная причина |
|---|---:|---|---|
| FN | 0.031 | Do you ever notice that when you're driving, anyone going slower than you is an idiot and everyone driving fas… | Рекламная рассылка в форме шутки-наблюдения: нет ссылки, номера, суммы и слов-призывов, по лексике это обычная разговорная фраза (p ≈ 0,03). |
| FN | 0.078 | For sale - arsenal dartboard. Good condition but no doubles or trebles! | Короткое объявление о продаже без номера и цены; по стилю неотличимо от личного сообщения. |
| FN | 0.131 | Latest News! Police station toilet stolen, cops have nothing to go on! | Новостная шутка с подпиской; единственный заметный маркер — «!», этого недостаточно. |
| FN | 0.321 | 0A$NETWORKS allow companies to bill for SMS, so they are responsible for their "suppliers", just as a shop has… | Информационный/жалобный текст без типичной спам-лексики; название сети записано с искажением и словам модели не знакомо. |
| FN | 0.416 | SMS. ac JSco: Energy is high, but u may not know where 2channel it. 2day ur leadership skills r strong. Psychi… | Платный гороскоп с призывом «Reply»; много сокращений (u, 2day, r); вероятность 0,42 близка к порогу 0,5 — пограничный случай. |
| FP | 0.751 | Your bill at 3 is £33.65 so thats not bad! | Личное сообщение с суммой в фунтах. Гипотеза: признак `<money>` у спама встречается в ≈31% сообщений против ≈0,2% у ham, поэтому сам факт суммы сильно повышает вероятность. |
| FP | 0.588 | NEFT Transaction with reference number  &lt;#&gt;  for Rs. &lt;DECIMAL&gt;  has been credited to the beneficia… | Служебное банковское уведомление с заполнителями (`<#>`, `<DECIMAL>`, `<TIME>`); по форме похоже на автоматическую рассылку, которых в spam много. |
| FP | 0.530 | How to Make a girl Happy? It's not at all difficult to make girls happy. U only need to be... 1. A friend 2. C… | Длинное цепочное сообщение с десятками числовых заполнителей `<#>` (после нормализации — `<num>`); `<num>` встречается в ≈88% спама и в ≈19% ham. |
| почти FP | 0.408 | "CHA QUITEAMUZING THATSCOOL BABE,PROBPOP IN & CU SATTHEN HUNNY 4BREKKIE! LOVE JEN XXX. PSXTRA LRG PORTIONS 4 … | Верно классифицировано как ham, но высокая доля заглавных букв и «!» подняли вероятность до 0,41. |
| почти FP | 0.398 | WE REGRET TO INFORM U THAT THE NHS HAS MADE A MISTAKE.U WERE NEVER ACTUALLY BORN.PLEASE REPORT 2 YOR LOCAL HOS… | Шуточное сообщение капсом в стиле официального уведомления; верно ham, вероятность 0,40. |

## Проверка сохранения и загрузки

* pipeline сохранён в `ml/artifacts/sms_baseline.joblib`, загружен в отдельной ячейке: предсказания на валидации совпадают;
* артефакт загружается и в новом процессе Python (классы преобразователей находятся в `ml/sms_features.py`, а не в `__main__`);
* `predict_sms(text) -> {"label", "probability"}` использует сохранённый pipeline; нормализация `normalize_sms` выполняется внутри него.

## Ограничения

Модель обучена на англоязычном наборе UCI SMS Spam Collection; спам в нём не тождествен финансовому мошенничеству. Для русского и казахского языков оценка ненадёжна. Требования раздела 3.4 к итоговой модели относятся к заданию 8; здесь это baseline.
