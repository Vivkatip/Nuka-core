import sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT = "SI VACV"
CAP = float(sys.argv[1]) if len(sys.argv) > 1 else 5000

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
allx = pd.concat([tr, te], ignore_index=True)
n = len(tr)

# ---------- признаки: Morgan count + Chiral Morgan + Descriptors ----------
gen_cnt = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
gen_chi = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=512, includeChirality=True
)


def feats(s):
    m = Chem.MolFromSmiles(s)
    return np.concatenate(
        [
            gen_cnt.GetCountFingerprintAsNumPy(m).astype(float),
            gen_chi.GetCountFingerprintAsNumPy(m).astype(float),
            [f(m) for _, f in Descriptors.descList],
        ]
    )


print("Генерация молекулярных дескрипторов и фингерпринтов...")
M = np.nan_to_num(
    np.vstack(allx["smi"].map(feats)), nan=0, posinf=1e6, neginf=-1e6
).clip(-1e6, 1e6)
st = [c for c in allx.columns if c.startswith("st_")] + ["n_virus"]
meta = pd.concat(
    [pd.get_dummies(allx[["culture", "src"]]).astype(float), allx[st].astype(float)],
    axis=1,
).values
X_all = np.hstack([M, meta])
X, Xte = X_all[:n], X_all[n:]
y = np.clip(tr[TGT].values.astype(float), 0, CAP)

# ---------- обучение ансамбля ----------
print(f"Обучение ET (5 сидов) и CBlog (3 сида) с CAP={CAP}...")
p_et = np.mean(
    [
        ExtraTreesRegressor(
            500,
            min_samples_leaf=2,
            max_features=0.3,
            n_jobs=-1,
            random_state=s,
        )
        .fit(X, y)
        .predict(Xte)
        for s in range(5)
    ],
    axis=0,
)

p_cb = np.mean(
    [
        np.expm1(
            CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                verbose=0,
                random_seed=s,
            )
            .fit(X, np.log1p(y))
            .predict(Xte)
        )
        for s in range(3)
    ],
    axis=0,
)

pred = np.clip((p_et + p_cb) / 2, 0, None)

# ---------- lookup замена точных совпадений по flat + culture ----------
d = (
    pd.DataFrame({"k": list(zip(tr.flat, tr.culture)), "y": y})
    .groupby("k")
    .y.mean()
    .to_dict()
)
hit = np.array([d.get(k, np.nan) for k in zip(te.flat, te.culture)])
replaced = int(np.isfinite(hit).sum())
pred = np.where(np.isnan(hit), pred, hit)
print(f"Lookup заменил: {replaced} строк из {len(te)}")

# ---------- сохранение ----------
sub = pd.DataFrame({"index": te["index"].astype(int), TGT: pred})
name = f"submission_v2_cap{int(CAP)}.csv"
sub.to_csv(ROOT / name, index=False)
print("\nСтатистика сабмита:")
print(sub[TGT].describe().round(1))
print(f"Файл успешно сохранён: {name}")