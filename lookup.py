"""
Lookup по точному совпадению молекулы (InChIKey-14, плоский ключ без стереохимии).
Приоритет: (молекула, культура) -> молекула -> модель.
combined_lookup оставлен для совместимости с explore_external.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_lookup(keys, cultures, y):
    """Медианы SI по паре (молекула, культура) и по молекуле. Строки без ключа пропускаются."""
    df = pd.DataFrame({
        "k": list(keys),
        "c": list(cultures),
        "y": np.asarray(y, dtype=float),
    })
    df = df[df["k"].notna()]
    by_kc = df.groupby(["k", "c"])["y"].median().to_dict()
    by_k = df.groupby("k")["y"].median().to_dict()
    return by_kc, by_k


def query_lookup(lk, keys, cultures):
    """Для каждой строки возвращает медиану SI или NaN, если совпадения нет."""
    by_kc, by_k = lk
    out = np.full(len(keys), np.nan)
    for i, (k, c) in enumerate(zip(keys, cultures)):
        if k is None:
            continue
        v = by_kc.get((k, c))
        if v is None:
            v = by_k.get(k)
        if v is not None:
            out[i] = v
    return out


def apply_lookup(pred, lk_tr, lk_ext, variant):
    """Приоритет: train -> external -> модель. variant: 'none' | 'train' | 'train+ext'."""
    out = np.array(pred, dtype=float, copy=True)
    if variant in ("train", "train+ext"):
        m = ~np.isnan(lk_tr)
        out[m] = lk_tr[m]
    if variant == "train+ext":
        m = np.isnan(lk_tr) & ~np.isnan(lk_ext)
        out[m] = lk_ext[m]
    return out


def combined_lookup(tr, ext):
    """Совместимость с explore_external.py: словари медиан только по молекуле (train и external)."""
    _, med_tr = fit_lookup(tr["key_flat"], tr["culture"], tr["y"])
    _, med_ext = fit_lookup(ext["key_flat"], ext["culture"], ext["y"])
    return med_tr, med_ext
