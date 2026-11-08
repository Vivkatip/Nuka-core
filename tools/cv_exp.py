# cv_exp.py — эксперименты с признаками/моделями/ext в group и scaffold CV
# Критерий принятия задан заранее (см. блок ВЕРДИКТ в конце).
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT, CAP, REPEATS, W_EXT = "SI VACV", 5000, 2, 0.5     # W_EXT экспертный, не подбирается

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
allx = pd.concat([tr, te], ignore_index=True)
n = len(tr)
y = tr[TGT].values.astype(float)
yc = np.clip(y, 0, CAP)
no_top = y < 30000                      # исключаем единственную точку 36067 из сырой метрики
lfc = rdMolStandardize.LargestFragmentChooser()

# ---------------- признаки ----------------
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
SMARTS = {
    "nucleoside":    "[#7;R][CX4;R1]1[OX2][CX4][CX4][CX4]1",
    "thio_sugar":    "[#7;R][CX4;R1]1[SX2][CX4][CX4][CX4]1",
    "phosphonate":   "[PX4](=O)[#6]",
    "phosphate":     "[PX4](=O)([OX2])[OX2]",
    "adamantane":    "C1C2CC3CC1CC(C2)C3",
    "bornane":       "C1CC2CCC1C2",
    "nitroarene":    "c[N+](=O)[O-]",
    "phenothiaz":    "[#16+]",
    "imidazole_N":   "[CX4][nX3]1cc[nX2]c1",
    "long_chain":    "[CH2][CH2][CH2][CH2][CH2][CH2][CH2]",
    "cytosine_like": "[NX3;H2]c1nc(=O)n(*)cc1",
}
PATS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
assert all(p is not None for p in PATS.values()), "битый SMARTS"


def core_feats(m):
    fp = gen.GetCountFingerprintAsNumPy(m).astype(float)
    d = np.array([f(m) for _, f in Descriptors.descList], float)
    return np.concatenate([fp, d])


def flag_feats(m):
    return np.array([len(m.GetSubstructMatches(p)) for p in PATS.values()], float)


def clean(a):
    return np.nan_to_num(a, nan=0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)


def mk_scaf(m):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return ""


mols = [Chem.MolFromSmiles(s) for s in allx["smi"]]
core = clean(np.vstack([core_feats(m) for m in mols]))
flags = np.vstack([flag_feats(m) for m in mols])
print("флаги (доля строк с признаком, train+test):",
      dict(zip(SMARTS, (flags > 0).mean(0).round(2))))

st_cols = [c for c in allx.columns if c.startswith("st_")] + ["n_virus"]
dum = pd.get_dummies(allx[["culture", "src"]]).astype(float)
dum_cols = dum.columns
meta = pd.concat([dum, allx[st_cols].astype(float)], axis=1).values

X_ref = np.hstack([core, meta])                                    # как в основном пайплайне
X_fl = np.hstack([core, flags, meta, np.zeros((len(allx), 1))])    # + флаги + is_ext
XR, XF = X_ref[:n], X_fl[:n]

tr_scaf = np.array([mk_scaf(m) for m in mols[:n]])
tr_flat = tr["flat"].astype(str).values
tr_cult = tr["culture"].values
groups = tr["group"].values
gid_fc = pd.factorize(pd.Series(tr_flat) + "|" + pd.Series(tr_cult))[0]

# ---------------- внешние данные ----------------
ext = pd.read_csv(ROOT / "data" / "external_chembl.csv")
print("колонки ext:", ext.columns.tolist())
scol = next((c for c in ["smiles", "smi", "canonical_smiles", "SMILES"] if c in ext.columns), None)
if scol is None or "SI" not in ext.columns or "culture" not in ext.columns:
    raise SystemExit("Нужны колонки smiles-подобная, SI и culture. Пришлите список колонок выше.")
if "inactive" in ext.columns:
    ext = ext[~ext["inactive"].astype(str).str.lower().isin(["true", "1"])]
if "virus" in ext.columns:
    ext = ext[ext["virus"].astype(str).str.lower().eq("vaccinia")]
ext = ext[ext["SI"] > 0].copy()
print("ext: vaccinia, измеренные, SI>0:", len(ext))

rec = []
for s, c, si in zip(ext[scol], ext["culture"], ext["SI"]):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        continue
    m = lfc.choose(m)
    rec.append((Chem.MolToSmiles(m, isomericSmiles=False), mk_scaf(m), c, si, m))
E = pd.DataFrame(rec, columns=["flat", "scaf", "culture", "SI", "mol"])
known = set(allx["culture"].unique())
E["culture"] = E["culture"].where(E["culture"].isin(known), "Other")
# агрегат: геометрическое среднее по (flat, culture)
agg = (E.groupby(["flat", "culture"])
         .agg(scaf=("scaf", "first"),
              SI=("SI", lambda s: float(np.exp(np.log(s).mean()))),
              mol=("mol", "first"))
         .reset_index())
print("ext после агрегации:", len(agg), "| пересечение с train по flat:",
      int(agg["flat"].isin(set(tr_flat)).sum()))

e_core = clean(np.vstack([core_feats(m) for m in agg["mol"]]))
e_flags = np.vstack([flag_feats(m) for m in agg["mol"]])
e_dum = (pd.get_dummies(pd.DataFrame({"culture": agg["culture"], "src": "chembl"}))
           .astype(float).reindex(columns=dum_cols, fill_value=0.0))
e_st = pd.DataFrame(0.0, index=agg.index, columns=st_cols)
e_st["n_virus"] = 1.0
XEF = np.hstack([e_core, e_flags, pd.concat([e_dum, e_st], axis=1).values,
                 np.ones((len(agg), 1))])
assert XEF.shape[1] == XF.shape[1], (XEF.shape, XF.shape)
y_ext = agg["SI"].values
ext_flat, ext_scaf = agg["flat"], agg["scaf"]


def ext_keep(b, kind):
    """Убираем из ext молекулы валидационного фолда (и их скаффолды в scaffold-схеме)."""
    keep = ~ext_flat.isin(set(tr_flat[b]))
    if kind == "scaffold":
        keep &= ~ext_scaf.isin(set(tr_scaf[b]))
    return keep.values


# ---------------- модели ----------------
def fit_blend(Xa, ya, wa, Xb, cap=CAP):
    c = np.clip(ya, 0, cap)
    et = ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1,
                             random_state=0).fit(Xa, c, sample_weight=wa)
    cb = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=0,
                           random_seed=0, thread_count=-1).fit(Xa, np.log1p(c), sample_weight=wa)
    return 0.5 * et.predict(Xb) + 0.5 * np.expm1(cb.predict(Xb))


INNER = [1e-9, 1.5, 5, 20, 100, 1000]       # бины: 0 | (0,1.5) | ... | >1000


def fit_ord(Xa, ya, Xb, cap=CAP):
    c = np.clip(ya, 0, cap)
    k = np.digitize(c, INNER)
    means = {j: c[k == j].mean() for j in np.unique(k)}
    clf = ExtraTreesClassifier(500, min_samples_leaf=2, max_features=0.3, n_jobs=-1,
                               random_state=0).fit(Xa, k)
    return clf.predict_proba(Xb) @ np.array([means[j] for j in clf.classes_])


def m_ref(a, b, kind):
    return fit_blend(XR[a], y[a], None, XR[b])


def m_flags(a, b, kind):
    return fit_blend(XF[a], y[a], None, XF[b])


def m_ord(a, b, kind):
    return fit_ord(XF[a], y[a], XF[b])


def m_ext(a, b, kind):
    k = ext_keep(b, kind)
    Xa = np.vstack([XF[a], XEF[k]])
    ya = np.r_[y[a], y_ext[k]]
    wa = np.r_[np.ones(len(a)), np.full(k.sum(), W_EXT)]
    return fit_blend(Xa, ya, wa, XF[b])


def m_hgb(a, b, kind):
    mdl = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.03, max_iter=400,
                                        max_leaf_nodes=15, min_samples_leaf=5,
                                        l2_regularization=1.0, random_state=0)
    return mdl.fit(XF[a], np.clip(y[a], 0, CAP)).predict(XF[b])


# hgb последним: если он упадёт, остальное уже в кэше
RUNS = {"ref_blend": m_ref, "blend_flags": m_flags, "ord_flags": m_ord,
        "blend_flags_ext": m_ext, "hgb_poisson_flags": m_hgb}


def splits(kind, seed):
    g = groups if kind == "scaffold" else gid_fc
    perm = np.random.RandomState(seed).permutation(g.max() + 1)
    return list(GroupKFold(5).split(XF, groups=perm[g]))


# ---------------- запуск с кэшем OOF ----------------
# Кэш зависит от признаков и параметров. Меняли SMARTS/модели — удалите файл.
CACHE = ROOT / "data" / "cv_exp_cache.pkl"
oofs = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
print("из кэша:", sorted(oofs))

for kind in ("group_flat_culture", "scaffold"):
    sp = {s: splits(kind, s) for s in range(REPEATS)}
    for name, fn in RUNS.items():
        if (kind, name) in oofs:
            continue
        L = []
        for s in range(REPEATS):
            o = np.zeros(n)
            for a, b in sp[s]:
                o[b] = fn(a, b, kind)
            L.append(np.clip(o, 0, None))
        oofs[(kind, name)] = L
        pickle.dump(oofs, open(CACHE, "wb"))
        print(kind, name, "готово", flush=True)
    base = [k for k in ("blend_flags", "ord_flags", "hgb_poisson_flags") if (kind, k) in oofs]
    oofs[(kind, "mix_noext")] = [np.mean([oofs[(kind, k)][s] for k in base], 0)
                                 for s in range(REPEATS)]
    mix_e = [k for k in ("blend_flags_ext", "ord_flags") if (kind, k) in oofs]
    oofs[(kind, "mix_ext")] = [np.mean([oofs[(kind, k)][s] for k in mix_e], 0)
                               for s in range(REPEATS)]

# ---------------- отчёт ----------------
rows = []
for (kind, name), L in oofs.items():
    mse = [((o - yc) ** 2).mean() for o in L]
    rows.append(dict(cv=kind, model=name, MSE_cap5k=round(np.mean(mse)),
                     seed_std=round(np.std(mse)),
                     MSE_raw_no_top1=round(np.mean([((o - y)[no_top] ** 2).mean() for o in L])),
                     spearman=round(np.mean([spearmanr(o, y)[0] for o in L]), 3)))
R = pd.DataFrame(rows).sort_values(["cv", "MSE_cap5k"])
print("\n", R.to_string(index=False))
R.to_csv(ROOT / "data" / "cv_exp.csv", index=False)

# ---------- вердикт по критерию, заданному заранее ----------
PARENT = {"blend_flags": "ref_blend", "ord_flags": "blend_flags",
          "hgb_poisson_flags": "blend_flags", "blend_flags_ext": "blend_flags",
          "mix_noext": "blend_flags", "mix_ext": "blend_flags"}


def g(kind, m, col):
    return float(R[(R.cv == kind) & (R.model == m)][col].iloc[0])


print("\nВЕРДИКТ (scaffold: выигрыш > 15k; group: не хуже -5k; Spearman scaffold не хуже -0.02)")
for cand, par in PARENT.items():
    if ("scaffold", cand) not in oofs or ("scaffold", par) not in oofs:
        continue
    gs = g("scaffold", par, "MSE_cap5k") - g("scaffold", cand, "MSE_cap5k")
    gg = g("group_flat_culture", par, "MSE_cap5k") - g("group_flat_culture", cand, "MSE_cap5k")
    ds = g("scaffold", cand, "spearman") - g("scaffold", par, "spearman")
    ok = gs > 15000 and gg > -5000 and ds > -0.02
    print(f"{cand:18s} vs {par:12s} | scaffold {gs:+8.0f} | group {gg:+8.0f} | "
          f"dSpearman {ds:+.3f} -> {'ПРИНЯТЬ' if ok else 'отклонить'}")