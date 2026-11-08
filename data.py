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


def load_external(path):
    """Читает smiles, SI, IC50, CC50. Отсекает мусорные ID веществ (>40 000 в колонке SI)."""
    raw = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    raw.columns = [_clean_name(c) for c in raw.columns]
    cols = {c.casefold(): c for c in raw.columns}

    need = {"smiles"}
    if not need <= set(cols):
        raise ValueError(f"{Path(path).name}: нужна колонка smiles, есть {list(raw.columns)}")

    smiles = raw[cols["smiles"]].str.strip().replace("", np.nan)

    si = None
    if "si" in cols:
        si = pd.to_numeric(raw[cols["si"]].str.strip().replace("", np.nan), errors="coerce")

    ic50 = None
    if "ic50" in cols:
        ic50 = pd.to_numeric(raw[cols["ic50"]].str.strip().replace("", np.nan), errors="coerce")

    cc50 = None
    if "cc50" in cols:
        cc50 = pd.to_numeric(raw[cols["cc50"]].str.strip().replace("", np.nan), errors="coerce")

    df = pd.DataFrame({"SMILES": smiles})
    if si is not None:
        df[TARGET] = si
    if ic50 is not None:
        df["IC50"] = ic50
    if cc50 is not None:
        df["CC50"] = cc50

    df = df[df["SMILES"].notna()].reset_index(drop=True)

    # Фильтруем физически реальный диапазон SI (отсекаем PubChem SID > 40 000)
    if TARGET in df.columns:
        n_before = len(df)
        df = df[(df[TARGET] >= 0) & (df[TARGET] <= 40000)].reset_index(drop=True)
        print(f"[data] external: строк {len(raw)}, валидных строк с SI <= 40k: {len(df)} (было {n_before})")
    else:
        print(f"[data] external: строк {len(raw)}, валидных строк: {len(df)}")
    return df
