"""(A) хвост ext без фильтра и ближайшие test; (B) согласие train<->ext; (C) submission_v3.
Требует: data/train_clean.csv, data/test_clean.csv, data/external_chembl.csv, submission_v2_cap5000.csv"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT = "SI VACV"
STRATEGY = "mean"      # при конфликте ext и train: "mean" | "ext" | "train"
pd.set_option("display.width", 230, "display.max_colwidth", 38)

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
ext = pd.read_csv(ROOT / "data" / "external_chembl.csv")
base = pd.read_csv(ROOT / "submission_v2_cap5000.csv").set_index("index")[TGT]


def mol_of(s):
    return Chem.MolFromSmiles(s) if isinstance(s, str) else None


def ik14(s):
    m = mol_of(s)
    return Chem.MolToInchiKey(m).split("-")[0] if m is not None else None


def flat_smi(s):
    m = mol_of(s)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None


for d, col in ((tr, "smi"), (te, "smi"), (ext, "smiles")):
    d["ik"] = d[col].map(ik14)
for d in (tr, te):
    if "flat" not in d.columns:
        d["flat"] = d["smi"].map(flat_smi)

ext["inactive"] = ext["inactive"].astype(bool)
ext["cc50_cens"] = ext["cc50_cens"].astype(bool)
ext = ext[ext.virus == "vaccinia"].copy()          # cowpox/variola/monkeypox не смешиваем

# ================= A. хвост ext =================
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp_te = [gen.GetFingerprint(mol_of(s)) for s in te.smi]
print("=== A. ext с SI >= 5000 (без фильтра) и ближайшие test ===")
tail = ext[ext.SI >= 5000].sort_values("SI", ascending=False)
print("строк:", len(tail), "| из них SI>40000:", int((ext.SI > 40000).sum()))
for _, r in tail.iterrows():
    m = mol_of(r.smiles)
    if m is None:
        continue
    sims = np.array(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fp_te))
    nn = np.argsort(-sims)[:2]
    in_tr = bool((tr.ik == r.ik).any())
    print(f"SI={r.SI:>10.0f} {str(r.culture):6s} IC50={r.IC50_uM:.4g} CC50={r.CC50_uM:.4g} "
          f"cens={r.cc50_cens} doi={str(r.doi)[:30]} in_train={in_tr}")
    for j in nn:
        idx = te["index"][j]
        print(f"      test idx={idx} {str(te.Title[j])[:18]:18s} {te.culture[j]:6s} sim={sims[j]:.2f} "
              f"same_ik={te.ik[j] == r.ik} same_culture={te.culture[j] == r.culture} pred={base[idx]:.0f}")

# ================= B. согласие train <-> ext =================
e = (ext.groupby(["ik", "culture"])
        .agg(ext_SI=("SI", "median"), ext_ratio=("SI_ratio", "median"),
             ext_inact=("inactive", "min"), ext_cens=("cc50_cens", "max"))
        .reset_index())
j = tr.merge(e, on=["ik", "culture"])


def report(name, d, col):
    if len(d) == 0:
        print(f"  {name}: пусто")
        return
    a, b = d[TGT].values, d[col].values
    lr = np.abs(np.log10((a + 1) / (b + 1)))
    rho = pd.Series(a).corr(pd.Series(b), method="spearman")
    print(f"  {name} [{col}] n={len(d)} spearman={rho:.3f} медиана|lg|={np.median(lr):.3f} "
          f"<12%: {(lr < 0.05).sum()} | в 3x: {(lr < 0.48).sum()} | "
          f"train=0,ext>0: {((a == 0) & (b > 0)).sum()} | train>0,ext=0: {((a > 0) & (b == 0)).sum()}")


print(f"\n=== B. train∩ext по (ik, culture): {len(j)} строк ===")
for col in ("ext_SI", "ext_ratio"):
    report("все", j, col)
    report("ext точные", j[~j.ext_inact.astype(bool)], col)
    report("ext точные, CC50 не censored", j[~j.ext_inact.astype(bool) & ~j.ext_cens.astype(bool)], col)
    report("ext censored CC50", j[~j.ext_inact.astype(bool) & j.ext_cens.astype(bool)], col)
inact = j[j.ext_inact.astype(bool)]
print(f"  ext inactive: n={len(inact)} | train==0 у них: {int((inact[TGT] == 0).sum())} "
      f"| train>0: {int((inact[TGT] > 0).sum())}")
print("\nрасхождения >3x (ext_SI):")
lr_all = np.abs(np.log10((j[TGT] + 1) / (j.ext_SI + 1)))
print(j.loc[lr_all >= 0.48, ["Title", "culture", TGT, "ext_SI", "ext_ratio", "ext_inact", "ext_cens"]]
        .round(1).head(25).to_string(index=False))

e_any = ext.groupby("ik").SI.median()
print("\nпокрытие без учета культуры: train", int(tr.ik.isin(e_any.index).sum()),
      "| с культурой:", len(j), "| test без культуры:", int(te.ik.isin(e_any.index).sum()),
      "| test с культурой:", int(te.merge(e, on=["ik", "culture"]).shape[0]))

# ================= C. submission_v3 =================
d_ext = e.set_index(["ik", "culture"]).ext_SI.to_dict()
d_tr = tr.groupby(["smi", "culture"])[TGT].mean().to_dict()          # со стереохимией
d_trf = tr.groupby(["flat", "culture"])[TGT].mean().to_dict()


def combine(ex, t):
    if STRATEGY == "ext":
        return ex
    if STRATEGY == "train":
        return t
    return (ex + t) / 2


pred, how = [], []
for _, r in te.iterrows():
    p0 = base[r["index"]]
    ex = d_ext.get((r.ik, r.culture))
    ts = d_tr.get((r.smi, r.culture))
    if ts is None:
        ts = d_trf.get((r.flat, r.culture))
    if ex is not None and ts is not None:
        pred.append(combine(ex, ts)); how.append(f"ext+train({STRATEGY})")
    elif ex is not None:
        pred.append(ex); how.append("ext")
    else:
        pred.append(p0); how.append("v2")

out = pd.DataFrame({"index": te["index"].astype(int), TGT: np.clip(pred, 0, None)})
out.to_csv(ROOT / "submission_v3.csv", index=False)
print("\n=== C. источники предсказаний:", pd.Series(how).value_counts().to_dict())
chg = out.assign(v2=out["index"].map(base), how=how).query("how != 'v2'")
print("макс |v3-v2|:", float((chg[TGT] - chg.v2).abs().max()) if len(chg) else 0.0,
      "| max v3:", float(out[TGT].max()))
print(chg.assign(Title=te.set_index("index").Title.reindex(chg["index"]).values)
        .sort_values(TGT, ascending=False).head(15).round(1).to_string(index=False))