# Nuclear IT Hack — Qualiﬁcation Package

## Overview
Решение задачи предсказания селективного индекса (SI) против вируса оспы (VACV) на основе химических структур (SMILES).

## Структура репозитория
- `src/` — исходный код пайплайна
- `data/` — исходные и внешние данные
- `submissions/` — сгенерированные файлы для Kaggle
- `cache/` — подготовленный кэш (prepared.pkl)
- `reports/` — результаты кросс-валидации и конфигурация
- `tools/` — вспомогательные скрипты (из uploads)

## Запуск
```bash
pip install -r requirements.txt
python src/download_external.py   # внешняя БД с реальными SI (PubChem + ChEMBL)
python src/prepare.py             # кэш cache/prepared.pkl
python src/submit.py --config reports/best_config.json --out submissions/submission.csv
```

## Ограничения
- Не используются колонки CC50_uM / IC50_uM_VACV как признаки или цель
- Lookup только по плоскому InChIKey-14 (без стереохимии)
- Нет новых моделей (LightGBM, KNN, GCN) — только ExtraTrees / RandomForest

## Лучший скор на Kaggle
- submission_train_only.csv: ~1.58M MSE (проверен)
- submission_v3.csv: 1 562 675.83421 (лидерборд)

## Автор
Участник хакатона Nuclear IT Hack
