from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from catboost import CatBoostRegressor

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT, CAP, REPEATS = "SI VACV", 5000, 3

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
allx = pd.concat([tr, te], ignore_index=True)
n = len(tr)

# ---------- 1. признаки (с добавлением хирального Morgan 512) ----------
g_cnt = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
g_chi = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=512, includeChirality=True
)
g_bit = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Стереохимия берется напрямую из smi
mols = [Chem.MolFromSmiles(s) for s in allx["smi"]]
M = np.vstack(
    [
        np.concatenate(
            [
                g_cnt.GetCountFingerprintAsNumPy(m).astype(float),
                g_chi.GetCountFingerprintAsNumPy(m).astype(
                    float
                ),  # <- Хиральный блок
                [f(m) for _, f in Descriptors.descList],
            ]
        )
        for m in mols
    ]
)
M = np.nan_to_num(M, nan=0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)

st = [c for c in allx.columns if c.startswith("st_")] + ["n_virus"]
meta = pd.concat(
    [pd.get_dummies(allx[["culture", "src"]]).astype(float), allx[st].astype(float)],
    axis=1,
).values
X_all = np.hstack([M, meta])
X, Xte = X_all[:n], X_all[n:]

y = tr[TGT].values.astype(float)
yc = np.clip(y, 0, CAP)
cult, groups, flat = tr["culture"].values, tr["group"].values, tr["flat"].values

# ---------- 2. диагностика: сходство test с train ----------
fps = [g_bit.GetFingerprint(m) for m in mols]


def max_sim(q, ref):
    return np.array(
        [
            max(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in ref]))
            for i in q
        ]
    )


tr_idx = np.arange(n)
s_test = max_sim(range(n, len(allx)), tr_idx)
print("\nmax Tanimoto до ближайшего соседа в train (квантили 10/25/50/75/90):")
print("  test      ", np.quantile(s_test, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
for kind in ("random", "scaffold"):
    s = []
    sp = (
        KFold(5, shuffle=True, random_state=0).split(X)
        if kind == "random"
        else GroupKFold(5).split(X, groups=groups)
    )
    for a, b in sp:
        s.append(max_sim(b, a))
    print(
        f"  CV {kind:8s}",
        np.quantile(np.concatenate(s), [0.1, 0.25, 0.5, 0.75, 0.9]).round(2),
    )


# ---------- 3. модели ----------
def m_const(a, b):
    return np.full(len(b), yc[a].mean())


def m_culture(a, b):
    mu = pd.DataFrame({"c": cult[a], "y": yc[a]}).groupby("c").y.mean()
    return pd.Series(cult[b]).map(mu).fillna(yc[a].mean()).values


def m_et(a, b):
    return (
        ExtraTreesRegressor(
            500,
            min_samples_leaf=2,
            max_features=0.3,
            n_jobs=-1,
            random_state=0,
        )
        .fit(X[a], yc[a])
        .predict(X[b])
    )


def m_rf(a, b):
    return (
        RandomForestRegressor(
            500,
            min_samples_leaf=2,
            max_features=0.3,
            n_jobs=-1,
            random_state=0,
        )
        .fit(X[a], yc[a])
        .predict(X[b])
    )


def m_cblog(a, b):
    mdl = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=0
    )
    return np.expm1(mdl.fit(X[a], np.log1p(yc[a])).predict(X[b]))


MODELS = {"culture": m_culture, "ET": m_et, "RF": m_rf, "CBlog": m_cblog}


def lookup(a, b):
    d = (
        pd.DataFrame({"k": list(zip(flat[a], cult[a])), "y": yc[a]})
        .groupby("k")
        .y.mean()
        .to_dict()
    )
    return np.array([d.get(k, np.nan) for k in zip(flat[b], cult[b])])


def stack_nested(P, t, seed):
    Z = np.c_[np.ones(len(P)), P]
    out = np.zeros(len(P))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(Z):
        w, _ = nnls(Z[a], t[a])
        out[b] = Z[b] @ w
    return out


mse = lambda p: ((np.clip(p, 0, None) - yc) ** 2).mean()

# ---------- 4. цикл валидации ----------
rows = []
last_oof = None

for kind in ("random", "scaffold"):
    for seed in range(REPEATS):
        if kind == "random":
            sp = list(KFold(5, shuffle=True, random_state=seed).split(X))
        else:
            perm = np.random.RandomState(seed).permutation(groups.max() + 1)
            sp = list(GroupKFold(5).split(X, groups=perm[groups]))

        oof = {k: np.zeros(n) for k in MODELS}
        lk = np.full(n, np.nan)
        cst = np.zeros(n)
        p_avg_calib = np.zeros(n)

        for a, b in sp:
            cst[b] = m_const(a, b)
            lk[b] = lookup(a, b)
            for k, fn in MODELS.items():
                oof[k][b] = fn(a, b)

            # Вложенная калибровка среднего предсказания по культуре:
            # Считаем коэффициент по обучающей части фолда
            fold_pred_a = (
                oof["ET"][a] + oof["CBlog"][a]
            ) / 2  # либо базовое среднее
            df_a = pd.DataFrame(
                {"c": cult[a], "y": yc[a], "p": np.maximum(fold_pred_a, 1.0)}
            )
            scale_map = (
                df_a.groupby("c")
                .apply(lambda g: np.clip(g["y"].mean() / g["p"].mean(), 0.5, 2.0))
                .to_dict()
            )

            p_raw_b = (oof["ET"][b] + oof["CBlog"][b]) / 2
            mults = pd.Series(cult[b]).map(scale_map).fillna(1.0).values
            p_avg_calib[b] = p_raw_b * mults

        P = np.column_stack([oof[k] for k in MODELS])
        stk = stack_nested(P, yc, seed)

        res = {
            "const": mse(cst),
            **{k: mse(v) for k, v in oof.items()},
            "avg(ET,CBlog)": mse((oof["ET"] + oof["CBlog"]) / 2),
            "calib_avg": mse(p_avg_calib),
            "stack": mse(stk),
            "stack+lookup": mse(np.where(np.isnan(lk), stk, lk)),
        }
        res.update(
            cv=kind, seed=seed, lookup_cover=round(np.isfinite(lk).mean(), 3)
        )
        rows.append(res)
        print(
            kind,
            seed,
            {
                k: round(v) if isinstance(v, float) and v > 5 else v
                for k, v in res.items()
            },
            flush=True,
        )

        last_oof = oof

df = pd.DataFrame(rows)
num = [c for c in df.columns if c not in ("cv", "seed", "lookup_cover")]
print("\nMSE_cap5k: среднее ± std по повторам")
for kind, d in df.groupby("cv"):
    print(f"\n[{kind}]  lookup покрывает {d.lookup_cover.mean():.1%}")
    print(
        pd.DataFrame(
            {"mean": d[num].mean().round(0), "std": d[num].std().round(0)}
        )
        .sort_values("mean")
        .to_string()
    )

# ---------- 5. диагностика калибровки по культурам (по последнему OOF) ----------
print("\n" + "=" * 60)
print("Диагностика калибровки культур (OOF последнего повтора):")
p_ens = (last_oof["ET"] + last_oof["CBlog"]) / 2
diag_df = pd.DataFrame({"c": cult, "y": yc, "p": p_ens})
table = diag_df.groupby("c").agg(
    n=("y", "size"), y_mean=("y", "mean"), p_mean=("p", "mean")
)
table["ratio (y/p)"] = (table["y_mean"] / np.maximum(table["p_mean"], 1)).round(
    2
)
print(table.round(1))
print("=" * 60)