import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REFRESH = "--refresh-ext" in sys.argv     # пересобрать внешние данные из ChEMBL
WITH_CV = "--cv" in sys.argv              # дополнительно посчитать CV-отчёт (10-15 мин)
ext_exists = (ROOT / "data" / "external_chembl.csv").exists()

STEPS = [["prepare_data.py"]]
if REFRESH or not ext_exists:
    STEPS.append(["collect_external.py"])
else:
    print("data/external_chembl.csv найден, сбор пропущен (--refresh-ext для пересбора)")
STEPS += [["validate_ext.py"], ["make_submission2.py", "5000"], ["apply_ext.py"]]
if WITH_CV:
    STEPS.append(["cv_baseline.py"])


def guard():
    bad = {"CC50_uM", "IC50_uM_VACV"}
    for f in ["data/train_clean.csv", "data/test_clean.csv"]:
        cols = set(pd.read_csv(ROOT / f, nrows=0).columns)
        assert not (cols & bad), f"{f}: запрещённые колонки {cols & bad}"


for s in STEPS:
    print(">>", " ".join(s), flush=True)
    subprocess.run([sys.executable, *s], cwd=ROOT, check=True)
    if s[0] == "prepare_data.py":
        guard()
print("готово: submission_v3.csv, data/ext_substitutions.csv")