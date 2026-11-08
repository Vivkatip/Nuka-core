"""
Шаг 1. Стандартизация SMILES, ключи для lookup, признаки.
Результат: cache/prepared.pkl (его читают explore_external.py, cv.py, submit.py).

Запуск:
    python prepare.py --train train_v2.csv --test test.csv --external external_data/open_database_vacv.csv
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.ML.Cluster import Butina

from data import TARGET, load_external, load_test, load_train
from features import canonical_key, culture_onehot, featurize, standardize

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

CLUSTER_DIST = 0.6  # порог Butina по расстоянию (1 - Tanimoto)
STRAIN_WORDS = ["copenhagen", "western reserve", "tiantan", "cowpox", "variola", "ectromelia"]


def inchikey14(mol):
    """Ключ молекулы без стереохимии (первые 14 символов InChIKey)."""
    if mol is None:
        return None
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return ik[:14] if ik else None


def source_feats(titles, strains):
    """Признаки происхождения записи. CC50/IC50 здесь не используются."""
    t = pd.Series(list(titles)).fillna("").astype(str).str.strip()
    s = pd.Series(list(strains)).fillna("").astype(str).str.lower()
    cols = [
        (t != "").astype(float),                                   # есть ли Title
        t.str.lower().str.startswith("sid").astype(float),         # PubChem-скрининг
        t.str.fullmatch(r"\d+").astype(float),                     # номер вида 15
        t.str.fullmatch(r"\d+[a-zA-Z]").astype(float),             # номер вида 3a
        t.str.contains(r"\d[a-zA-Z]+\d*").astype(float),           # код вида 9h-qs777
        t.str.len().clip(upper=60).astype(float),
    ] + [s.str.contains(w, regex=False).astype(float) for w in STRAIN_WORDS]
    return np.column_stack(cols)


def tanimoto_matrix(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def butina_groups(fp, thr=CLUSTER_DIST):
    S = tanimoto_matrix(fp, fp)
    n = len(fp)
    dists = []
    for i in range(1, n):
        dists.extend((1.0 - S[i, :i]).tolist())
    clusters = Butina.ClusterData(dists, n, thr, isDistData=True)
    groups = np.zeros(n, dtype=int)
    for cid, members in enumerate(clusters):
        groups[list(members)] = cid
    return groups


def _col(df, name):
    return df[name].tolist() if name in df.columns else [""] * len(df)


def make_block(parents, culture_names, titles, strains):
    F = featurize(parents)
    return {
        "desc": F["desc"],
        "maccs": F["maccs"],
        "fp": F["fp"],
        "valid": F["valid"],
        "cult": culture_onehot(list(culture_names)),
        "src": source_feats(titles, strains),
        "key_flat": [inchikey14(p) for p in parents],
        "key_iso": [canonical_key(p) for p in parents],
        "culture": list(culture_names),
    }


def assemble(block, use_morgan=False):
    """Матрица признаков для моделей. Morgan выключен по умолчанию."""
    parts = [block["desc"], block["maccs"].astype(float), block["cult"], block["src"]]
    if use_morgan:
        parts.append(block["fp"].astype(float))
    return np.hstack(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_v2.csv")
    ap.add_argument("--test", default="test.csv")
    ap.add_argument("--external", default="external_data/open_database_vacv.csv")
    ap.add_argument("--cache", default="cache/prepared.pkl")
    args = ap.parse_args()
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)

    # ---------- обучающая выборка ----------
    tr = load_train(args.train)
    parents = [standardize(s) for s in tr["SMILES"]]
    ok = np.array([p is not None for p in parents])
    print(f"[train] SMILES, которые не разобрал RDKit: {int((~ok).sum())}")
    tr = tr.loc[ok].reset_index(drop=True)
    parents = [p for p in parents if p is not None]

    tr["key_iso"] = [canonical_key(p) for p in parents]
    tr["key_flat"] = [inchikey14(p) for p in parents]
    tr["_pos"] = np.arange(len(tr))
    tr["y_med"] = tr.groupby(["key_iso", "culture"])[TARGET].transform("median")
    rep = tr.drop_duplicates(["key_iso", "culture"], keep="first").reset_index(drop=True)
    print(f"[train] строк {len(tr)}, уникальных (молекула, культура) {len(rep)}")

    B_tr = make_block(
        [parents[i] for i in rep["_pos"]],
        rep["culture"].tolist(), _col(rep, "Title"), _col(rep, "strain"),
    )
    B_tr["y"] = rep["y_med"].to_numpy(float)
    B_tr["groups"] = butina_groups(B_tr["fp"])
    print(f"[train] кластеров Butina: {B_tr['groups'].max() + 1}")

    # ---------- тест ----------
    te = load_test(args.test)
    te_parents = [standardize(s) for s in te["SMILES"]]
    B_te = make_block(te_parents, te["culture"].tolist(), _col(te, "Title"), _col(te, "strain"))
    B_te["index"] = te["index"].to_numpy()
    print(f"[test] распознано SMILES: {int(B_te['valid'].sum())} из {len(te)}")

    # ---------- внешняя база: одна запись на ключ (медиана SI) ----------
    ext = load_external(args.external)
    ext_parents = [standardize(s) for s in ext["SMILES"]]
    ext_df = pd.DataFrame({
        "key": [inchikey14(p) for p in ext_parents],
        "key_iso": [canonical_key(p) for p in ext_parents],
        "y": ext[TARGET].to_numpy(float) if TARGET in ext.columns else np.nan,
        "pos": np.arange(len(ext)),
    }).dropna(subset=["key"])
    g = ext_df.groupby("key")
    ext_med = g["y"].median()
    ext_first = g["pos"].first().loc[ext_med.index]
    n_ext = len(ext_med)
    B_ext = make_block(
        [ext_parents[int(p)] for p in ext_first],
        ["Unknown"] * n_ext, [""] * n_ext, [""] * n_ext,
    )
    B_ext["y"] = ext_med.to_numpy(float)
    B_ext["groups"] = -np.ones(n_ext, dtype=int)
    print(f"[external] строк {len(ext)}, уникальных ключей {n_ext}")

    blocks = {
        "tr": B_tr, "te": B_te, "ext": B_ext,
        "meta": {"train_file": args.train, "test_file": args.test, "external_file": args.external},
    }
    with open(args.cache, "wb") as f:
        pickle.dump(blocks, f)
    print(f"[ok] кэш записан: {args.cache}")


if __name__ == "__main__":
    main()
