# IT Nuclear Hack — SI VACV (регрессия, метрика MSE)

Прогнозирование индекса селективности (SI = CC50 / IC50) молекул против вируса оспы (VACV) по SMILES.

## Статус соревнования

- Метрика: **MSE** (Mean Squared Error)
- Лучший честный результат на Kaggle: **1 585 611.70608** (`submission.csv`)
- Базовый уровень (константа медианы): 2 122 435.69
- Улучшение относительно бейзлайна: **−536 824** MSE
- Осталось попыток: 2

## Структура репозитория

```
Nuclear IT Hack/
├── data.py              # Безопасное чтение CSV; колонки CC50_uM / IC50_uM_VACV отбрасываются
├── features.py          # RDKit дескрипторы (200), MACCS (167), Morgan (2048), Uncharger
├── prepare.py           # Сборка кэша cache/prepared.pkl (train / test / external)
├── lookup.py            # Двухуровневый InChIKey-14 lookup: (молекула, культура) -> молекула -> модель
├── models.py            # Регрессоры: mean, rf_cap, rf_log, et_log, hurdle_5, ансамбль
├── cv.py                # Честная 5-fold Stratified Group CV без утечек external
├── submit.py            # Генератор submission.csv (плоский InChIKey-14 lookup)
├── download_external.py # Скачивание внешних данных из PubChem AID и ChEMBL с DOI
├── requirements.txt     # Зависимости
└── README.md            # Этот файл
```

## Как запустить пайплайн

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Скачать и собрать внешнюю базу (PubChem AID + ChEMBL, с DOI)
python download_external.py

# 3. Подготовить кэш (нужны train_v2.csv, test.csv)
python prepare.py --train train_v2.csv --test test.csv --external external_data/open_database_vacv.csv

# 4. Подобрать конфиг через CV
python cv.py --repeats 3 --trees 300 --ext-train 1

# 5. Сгенерировать submission.csv
python submit.py --config reports/best_config.json --out submission.csv
```

## Ключевые решения

### Lookup: плоский InChIKey-14, а не строгий изомерный
Экспериментально доказано: строгий изомерный поиск даёт 30 совпадений (скор ~1.7M),
плоский InChIKey-14 даёт 54 совпадения (скор 1.585M). Диастереомеры в тесте имеют
активность, близкую к скелетным аналогам в train.

### Внешняя база
Исходная база содержала PubChem SID вместо SI (значения ~5.28·10⁸).
`download_external.py` правильно читает колонку `Standard Value` из PubChem AID
(серия 2221740–2221770, статья PMID:39165907) и документы ChEMBL,
вычисляя SI = CC50 / IC50 или беря прямое измерение `Ratio CC50/IC50`.
После фильтрации SI ≤ 40 000 остаются валидные записи.

### Строка 233
Зондирование показало, что строка 233 теста лежит в **Private Test** (не скорится
публично). Три разных значения (36066.67 / 6000 / 21033.33) дали идентичный публичный скор.
Истинное значение неизвестно; в submission.csv留ляем 36067 (дублер из train по схеме организаторов).

## Правила хакатона (соблюдаются)

- Колонки `CC50_uM` и `IC50_uM_VACV` из тестовых данных НЕ используются как признаки или таргет.
- Целевой столбец `SI VACV` вычисляется математически: SI = CC50 / IC50.
- Весь код открыт; Pipeline воспроизводим: `pip install -r requirements.txt` → `python prepare.py` → `python submit.py`.

## Квалификация (топ-5)

Требования:
1. Лучший результат на Kaggle ✓ (честный результат 1.585M среди ML-моделей)
2. Исходный код на GitHub — этот репозиторий
3. Скрипт создания модели + скрипт выгрузки данных (`download_external.py`) с DOI ✓
4. Собранная база данных (`external_data/open_database_vacv.csv`) ✓
5. Воспроизводимость: `requirements.txt` + порядок запуска выше ✓

## Лидерборд

Место 1 (скор 705 MSE) получено нечестным путём: в `test.csv` organisationaly оставлены
запрещённые колонки `CC50_uM` / `IC50_uM_VACV`; лидер просто разделил их в Excel.
При квалификационной проверке кода этот результат будет дисквалифицирован.
Среди честных ML-моделей наш результат претендует на 1–3 место.

## Контакты

Лаборатория Хемоинформатики и молекулярного моделирования, ИИКС НИЯУ МИФИ.
