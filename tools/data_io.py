"""Загрузка CSV. Колонки CC50_uM и IC50_uM_VACV не читаются вовсе (правила хакатона)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "SI VACV"
FORBIDDEN = {"cc50_um", "ic50_um_vacv"}

_CULTURE_RULES = [
    ("vero", "Vero"), ("foreskin", "HFF"), ("hff", "HFF"), ("hela", "HeLa"),
    ("a549", "A549"), ("hel", "HEL"), ("bsc", "BSC"), ("cem", "CEM"),
    ("mt-4", "MT-4"), ("mt4", "MT-4"), ("e6sm", "E6SM"),
]


def _clean_name(name):
    return str(name).replace("\ufeff", "").strip()


def normalize_culture(text):
    low = str(text or "").lower()
    for pattern, label in _CULTURE_RULES:
        if pattern in low:
            return label
    return "Other"


def _read(path, required):
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        usecols=lambda c: _clean_name(c).casefold() not in FORBIDDEN,
    )
    df.columns = [_clean_name(c) for c in df.columns]
    leaked = FORBIDDEN & {c.casefold() for c in df.columns}
    if leaked:
        raise RuntimeError(f"{Path(path).name}: запрещённые колонки {sorted(leaked)}")
    df = df.apply(lambda col: col.str.strip()).replace("", np.nan)
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{Path(path).name}: нет колонок {sorted(missing)}, есть {list(df.columns)}"
        )
    return df


def _add_culture(df):
    if "Culture" in df.columns:
        df["culture"] = df["Culture"].fillna("").map(normalize_culture)
    else:
        df["culture"] = "Other"
    return df


def load_train(path):
    df = _read(path, {"SMILES", TARGET})
    n_raw = len(df)
    si = pd.to_numeric(df[TARGET], errors="coerce")
    n_bad = int(df[TARGET].notna().sum() - si.notna().sum())
    if n_bad:
        print(f"[warn] {Path(path).name}: нечисловых {TARGET}: {n_bad} (строки отброшены)")
    df[TARGET] = si
    df = df[df["SMILES"].notna() & df[TARGET].notna() & (df[TARGET] >= 0)]
    print(f"[data] {Path(path).name}: строк {n_raw}, после фильтра {len(df)}")
    return _add_culture(df.reset_index(drop=True))


def load_test(path):
    df = _read(path, {"index", "SMILES"})
    df["index"] = df["index"].astype(int)
    return _add_culture(df.reset_index(drop=True))