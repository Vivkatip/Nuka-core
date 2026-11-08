# cv_baseline.py — CV-отчёт основного пайплайна: random, GroupKFold(flat, culture), scaffold
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, GroupKFold
from catboost import CatBoostRegressor

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT, EVAL_CAP, REPEATS = "SI VACV", 5000, 2

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
allx = pd.concat([tr, te], ignore_index=True)
n = len(tr)

# ---------- признаки ----------
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def mol_feats(smi):
    m = Chem.MolFromSmiles(smi)
    fp = gen.GetCountFingerprintAsNumPy(m).astype(float)
    d = np.array([f(m) for _, f in Descriptors.descList], float)
    return np.concatenate([fp, d])


M = np.vstack(allx["smi"].map(mol_feats))
M = np.nan_to_num(M, nan=0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)

st_cols = [c for c in allx.columns if c.startswith("st_")] + ["n_virus"]
meta = pd.concat([pd.get_dummies(allx[["culture", "src"]]).astype(float),
                  allx[st_cols].astype(float)], axis=1).values
X_all = np.hstack([M, meta])
X, Xte = X_all[:n], X_all[n:]
y = tr[TGT].values.astype(float)
yc = np.clip(y, 0, EVAL_CAP)
cult = tr["culture"].values
groups = tr["group"].values
print("X:", X.shape, "| дисперсия y (cap 5000):", round(yc.var()), "| дисперсия y raw:", round(y.var()))


# ---------- модели ----------
def m_const(a, b, cap):
    return np.full(len(b), np.clip(y[a], 0, cap).mean())


def m_culture(a, b, cap):
    d = pd.DataFrame({"c": cult[a], "y": np.clip(y[a], 0, cap)})
    mu = d.groupby("c").y.mean()
    return pd.Series(cult[b]).map(mu).fillna(d.y.mean()).values


def m_et(a, b, cap):
    mdl = ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=0)
    return mdl.fit(X[a], np.clip(y[a], 0, cap)).predict(X[b])


def _cb():
    return CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0,
                             random_seed=0, thread_count=-1)


def m_cb_log(a, b, cap):
    return np.expm1(_cb().fit(X[a], np.log1p(np.clip(y[a], 0, cap))).predict(X[b]))


def m_blend(a, b, cap):
    return 0.5 * m_et(a, b, cap) + 0.5 * m_cb_log(a, b, cap)


RUNS = [("const", m_const, 5000),
        ("culture_mean", m_culture, 5000),
        ("ET", m_et, 1000), ("ET", m_et, 5000), ("ET", m_et, 40000),
        ("CB_log", m_cb_log, 5000),
        ("blend ET+CB_log", m_blend, 5000)]

# группы (flat, culture): дубли и повторы одной молекулы в одной культуре не попадают в разные фолды
gid_fc = pd.factorize(tr["flat"].astype(str) + "|" + tr["culture"].astype(str))[0]
no_top = y < 30000          # исключаем единственную точку 36067 из сырой метрики


def splits(kind, seed):
    if kind == "random":
        return list(KFold(5, shuffle=True, random_state=seed).split(X))
    g = groups if kind == "scaffold" else gid_fc
    perm = np.random.RandomState(seed).permutation(g.max() + 1)
    return list(GroupKFold(5).split(X, groups=perm[g]))


rows = []
for kind in ("random", "group_flat_culture", "scaffold"):
    print(f"\n--- CV схема: {kind} ---")
    for name, fn, cap in RUNS:
        res = []
        for seed in range(REPEATS):
            oof = np.zeros(n)
            for a, b in splits(kind, seed):
                oof[b] = fn(a, b, cap)
            oof = np.clip(oof, 0, None)
            res.append((((oof - yc) ** 2).mean(),
                        ((oof - y)[no_top] ** 2).mean(),
                        spearmanr(oof, y)[0]))
        r = np.mean(res, axis=0)
        rows.append(dict(cv=kind, model=name, cap=cap,
                         MSE_cap5k=round(r[0]), MSE_raw_no_top1=round(r[1]),
                         spearman=round(r[2], 3)))
        print(rows[-1], flush=True)

out = pd.DataFrame(rows).sort_values(["cv", "MSE_cap5k"])
print("\nИтоговая таблица CV:\n", out.to_string(index=False))
out.to_csv(ROOT / "data" / "cv_report.csv", index=False)
print("Отчёт записан в", ROOT / "data" / "cv_report.csv")