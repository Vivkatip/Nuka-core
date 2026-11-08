"""Аудит внешних данных: пересечение с test/train, совместимость SI, тековиримат."""
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT = "SI VACV"
pd.set_option("display.width", 220, "display.max_colwidth", 45)

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
ext = pd.read_csv(ROOT / "data" / "external_chembl.csv")
sub = pd.read_csv(ROOT / "submission_v2_cap5000.csv").set_index("index")[TGT]

def flat(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else None
ext["smi"] = ext.smiles
ext["flat"] = ext.smiles.map(flat)

print("=== внешние данные ===")
print("строк:", len(ext), "| doi пусто:", int((ext.doi.fillna("") == "").sum()))
print(ext.groupby("virus").SI.agg(["count", "median", "max"]).round(1).to_string())
print(ext.groupby("culture").SI.agg(["count", "median"]).round(1).to_string())
print("SI квантили:", ext.SI.quantile([.5, .75, .9, .99]).round(1).to_dict())

print("\n=== пересечение ===")
for key in ("smi", "flat"):
    print(f"{key:5s}: test в ext {te[key].isin(set(ext[key])).sum()}/{len(te)} "
          f"| train в ext {tr[key].isin(set(ext[key])).sum()}/{len(tr)}")

g = ext.groupby("flat").agg(ext_SI=("SI", "median"), ext_culture=("culture", lambda s: ",".join(sorted(set(s)))),
                            virus=("virus", lambda s: ",".join(sorted(set(s)))),
                            doi=("doi", lambda s: ";".join(sorted(set(s)))[:40])).reset_index()
m = te.reset_index().merge(g, on="flat", how="inner")
m["pred_v2"] = m["index"].map(sub)
print("\n=== строки test, найденные во внешних данных ===")
print(m[["index", "Title", "culture", "ext_culture", "virus", "ext_SI", "pred_v2", "doi"]]
      .round(1).to_string(index=False))
print("\nтековиримат (test #70) во внешних:", bool((m["index"] == 70).any()))

print("\n=== совместимость SI: train ∩ ext (только vaccinia) ===")
a = tr.groupby("flat")[TGT].median().rename("train_SI")
b = ext[ext.virus == "vaccinia"].groupby("flat").SI.median().rename("ext_SI")
j = pd.concat([a, b], axis=1, join="inner")
print("общих молекул:", len(j))
if len(j) >= 5:
    print("Spearman: %.2f" % spearmanr(j.train_SI, j.ext_SI)[0])
    print("медиана log10(ext/train): %.2f" % np.median(np.log10((j.ext_SI + 1) / (j.train_SI + 1))))
    print(j.round(1).head(20).to_string())