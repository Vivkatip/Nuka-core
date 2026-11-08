from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT = "SI VACV"
pd.set_option("display.width", 230, "display.max_colwidth", 30)

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")

ext_path = ROOT / "data" / "external_chembl.csv"
if not ext_path.exists():
    raise SystemExit("Файл external_chembl.csv не найден: запустите collect_external.py")

ext = pd.read_csv(ext_path)
if "inactive" not in ext.columns:
    raise SystemExit("external_chembl.csv от старой версии: сначала запустите collect_external.py")

sub_path = ROOT / "submission_v2_cap5000.csv"
sub = pd.read_csv(sub_path).set_index("index")[TGT] if sub_path.exists() else None

def ik(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return Chem.MolToInchiKey(m).split("-")[0] if m else None

tr["ik"] = tr.smi.map(ik)
te["ik"] = te.smi.map(ik)
ext["ik"] = ext.smiles.map(ik)

print("ext: строк", len(ext), "| inactive", int(ext.inactive.sum()), "| max SI", round(ext.SI.max()))
print("\n=== ext с SI >= 10000 (все вирусы) ===")
big = ext[ext.SI >= 10000].sort_values("SI", ascending=False)
for _, r in big.iterrows():
    t = te[te.ik == r.ik]
    print(f"SI={r.SI:>9.0f} {r.culture:5s} {r.virus:9s} IC50={r.IC50_uM:.3g} CC50={r.CC50_uM:.3g} "
          f"doi={str(r.doi)[:22]} | в train: {int((tr.ik == r.ik).sum())} | в test: {len(t)}")
    if sub is not None:
        for _, x in t.iterrows():
            print(f"      TEST idx={x['index']} {x.Title[:18]} {x.culture} pred={sub[x['index']]:.0f}")

print("\n=== тековиримат: test, train и ext (любая культура) ===")
teco = te[te.Title.str.lower() == "tecovirimat"]
if len(teco):
    k = teco["ik"].iloc[0]
    print("test:\n", te[te.ik == k][["index", "Title", "culture"]].to_string(index=False))
    print("train:\n", tr[tr.ik == k][["Title", "culture", TGT]].to_string(index=False))
    print("ext:\n", ext[ext.ik == k][["culture", "virus", "IC50_uM", "CC50_uM", "SI", "inactive", "doi"]].to_string(index=False))
else:
    print("Тековиримат в test не найден по Title == 'tecovirimat'")

print("\n=== train=0 vs ext.inactive: подтверждается ли правило SI=0 ===")
e = ext.groupby(["ik", "culture"]).agg(ext_SI=("SI", "median"), ext_inact=("inactive", "min")).reset_index()
j = tr.merge(e, on=["ik", "culture"])
print(pd.crosstab(j[TGT] == 0, j.ext_inact, rownames=["train SI=0"], colnames=["ext inactive"]))