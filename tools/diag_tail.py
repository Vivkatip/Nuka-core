# diag_tail2.py
import pandas as pd, numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg

bad = {"CC50_uM", "IC50_uM_VACV"}
tr = pd.read_csv("train_v2.csv", usecols=lambda c: c not in bad)
te = pd.read_csv("test.csv",     usecols=lambda c: c not in bad)
if "index" not in te.columns:
    te["index"] = te.index
sub = pd.read_csv("submission_v3.csv").set_index("index")["SI VACV"]
te["pred"] = sub.reindex(te["index"]).values
T = "SI VACV"

def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None
tr["can"] = tr.SMILES.map(canon); te["can"] = te.SMILES.map(canon)

# 1) Title: метка соединения? совпадение Title даёт ту же молекулу?
print("unique Title train/test:", tr.Title.nunique(), te.Title.nunique(), "из", len(tr), len(te))
m = te.merge(tr[["Title", "can", "Culture", T]], on="Title", suffixes=("", "_tr"))
print("пар по Title:", len(m), "| та же молекула:", round((m.can == m.can_tr).mean(), 3))

# 2) аналоги хвостовых молекул train среди test
gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
def fp(s):
    mol = Chem.MolFromSmiles(s)
    return gen.GetFingerprint(mol) if mol else None
trf = [fp(s) for s in tr.SMILES]; tef = [fp(s) for s in te.SMILES]
ok = [f is not None for f in tef]

top = tr.sort_values(T, ascending=False).head(8)
for i, r in top.iterrows():
    sims = np.array([DataStructs.TanimotoSimilarity(trf[i], f) if f is not None else -1 for f in tef])
    j = np.argsort(-sims)[:3]
    print(f"\n[train] {r.Title} | {r.Culture} | SI={r[T]:.0f}")
    print(te.iloc[j][["index", "Title", "Culture", "pred"]].assign(tanimoto=sims[j].round(2)).to_string(index=False))

# 3) сколько test-строк вообще имеют близкого соседа (>=0.7) среди train-строк с SI>1000
hot = [trf[i] for i in tr.index[tr[T] > 1000]]
mx = np.array([max(DataStructs.BulkTanimotoSimilarity(f, hot)) if f is not None else 0 for f in tef])
print("\ntest-строк с Tanimoto>=0.7 к train SI>1000:", int((mx >= .7).sum()),
      "| >=0.5:", int((mx >= .5).sum()))
print("их pred:", np.sort(te.pred.values[mx >= .7]).round(0))