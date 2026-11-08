"""
compare_subs.py
Сравнивает новый сабмит (submission_attempt1.csv) с рекордным submission.csv (1 585 611.70608)
и показывает результаты CV именно для et_log и et_rf_blend.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# 1. Посмотрим CV результаты именно для честных моделей et_log и et_rf_blend (calib=0, variant=train)
cv_df = pd.read_csv("reports/cv_results.csv")
sub_cv = cv_df[
    cv_df["model"].isin(["et_log", "rf_log", "et_rf_blend"])
    & (cv_df["calib"] == 0)
    & (cv_df["variant"] == "train")
]
wide = sub_cv.pivot_table(
    index=["model", "lk_type", "analog_thr", "analog_w"],
    columns="mode",
    values="mse_mean"
)
wide["score"] = wide[["group", "random"]].mean(axis=1)
print("=== CV для et_log / et_rf_blend (calib=0, variant='train') ===")
print(wide.sort_values("random").round(1).head(15).to_string())

# 2. Сравним submission_attempt1.csv с рекордным submission.csv
p_old = Path("submission.csv")
p_new = Path("submission_attempt1.csv")
if p_old.exists() and p_new.exists():
    old = pd.read_csv(p_old)["SI VACV"].to_numpy()
    new = pd.read_csv(p_new)["SI VACV"].to_numpy()
    te = pd.read_csv("test.csv", usecols=["index", "Title", "Culture", "strain"])

    diff = np.abs(new - old)
    print("\n=== СРАВНЕНИЕ submission_attempt1.csv VS submission.csv (1 585 611) ===")
    print(f"Старый (submission.csv)      : медиана={np.median(old):.2f}, среднее={np.mean(old):.2f}, макс={np.max(old):.2f}")
    print(f"Новый (submission_attempt1)  : медиана={np.median(new):.2f}, среднее={np.mean(new):.2f}, макс={np.max(new):.2f}")
    print(f"Строк с разницей > 10 единиц : {int((diff > 10).sum())}")
    print(f"Строк с разницей > 50 единиц : {int((diff > 50).sum())}")

    top_idx = np.argsort(-diff)[:20]
    rows = []
    for i in top_idx:
        if diff[i] < 1.0:
            continue
        rows.append({
            "index": int(te["index"].iloc[i]),
            "Title": str(te["Title"].iloc[i])[:22],
            "Culture": str(te["Culture"].iloc[i])[:18],
            "old_SI": round(float(old[i]), 2),
            "new_SI": round(float(new[i]), 2),
            "delta": round(float(new[i] - old[i]), 2),
        })
    print("\nТоп-20 самых больших изменений прогноза (где Near-Analog нашёл активных соседей):")
    print(pd.DataFrame(rows).to_string(index=False))
else:
    print(f"\nФайлы для сравнения: submission.csv ({p_old.exists()}), submission_attempt1.csv ({p_new.exists()})")