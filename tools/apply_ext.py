import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parent
T = "SI VACV"
W_MEAS, W_BOUND, CAP_SAFE = 0.9, 0.7, 20000      # веса экспертные, по LB/CV не подбирались
EXCLUDE_CULTURES = {"Other"}                      # сборная категория, не одна клеточная линия

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
ext = pd.read_csv(ROOT / "data" / "external_chembl.csv")
base = pd.read_csv(ROOT / "submission_v2_cap5000.csv").set_index("index")[T]


def keys(s):
    """(плоский ключ, ключ со стереохимией) из InChIKey."""
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is None:
        return None, None
    p = Chem.MolToInchiKey(m).split("-")
    return p[0], "-".join(p[:2])


for d, col in ((te, "smi"), (ext, "smiles")):
    kk = d[col].map(keys)
    d["ik"] = kk.map(lambda t: t[0])
    d["ikf"] = kk.map(lambda t: t[1])

# ext: только vaccinia, с doi; SI = CC50/IC50 (для границы "> X" это CC50/X, конвенция train)
ext["inactive"] = ext["inactive"].astype(str).str.lower().isin(["true", "1"])
e = ext[(ext.virus == "vaccinia") & ext.doi.notna() & (ext.doi != "")].copy()
e["SI_fix"] = e.CC50_uM / e.IC50_uM
e = e[np.isfinite(e.SI_fix) & (e.SI_fix > 0)]


def agg(g):
    meas = g[~g.inactive]
    use = meas if len(meas) else g
    return pd.Series(dict(
        ext_SI=use.SI_fix.median(),
        w=W_MEAS if len(meas) else W_BOUND,
        bound=not len(meas),
        ikfs=";".join(sorted(set(use.ikf.dropna()))),
        doi=";".join(sorted(set(";".join(use.doi.astype(str)).split(";")))),
        chembl_doc=";".join(sorted(set(";".join(use.chembl_doc.astype(str)).split(";"))))))


E = e.groupby(["ik", "culture"]).apply(agg, include_groups=False).reset_index()

train_keys = set(zip(tr.smi, tr.culture))
te["in_train_smi"] = [(s, c) in train_keys for s, c in zip(te.smi, te.culture)]
m = te.merge(E, on=["ik", "culture"], how="left")
assert len(m) == len(te)
m["v2"] = m["index"].map(base)

use = m.ext_SI.notna() & ~m.in_train_smi & ~m.culture.isin(EXCLUDE_CULTURES)
m["stereo_match"] = [(k in s.split(";")) if isinstance(s, str) else np.nan
                     for k, s in zip(m.ikf, m.ikfs)]

m["final"] = m.v2
m.loc[use, "final"] = np.expm1(m.w[use] * np.log1p(m.ext_SI[use])
                               + (1 - m.w[use]) * np.log1p(m.v2[use]))
m["final"] = m.final.clip(0, CAP_SAFE)

print("test:", len(m), "| есть в ext (ik, culture):", int(m.ext_SI.notna().sum()),
      "| уже покрыты train lookup:", int((m.ext_SI.notna() & m.in_train_smi).sum()),
      "| исключено (Other):", int((m.ext_SI.notna() & ~m.in_train_smi
                                    & m.culture.isin(EXCLUDE_CULTURES)).sum()),
      "| заменено:", int(use.sum()), "| из них только граница:", int((use & m.bound).sum()))
print("макс ext_SI среди подстановок:", round(float(m.ext_SI[use].max())),
      "| макс final:", round(float(m.final.max())),
      "| сработал CAP_SAFE:", bool((m.final >= CAP_SAFE).any()))
sm = use & (m.stereo_match == False)          # noqa: E712
print("подстановок с несовпадением стереохимии:", int(sm.sum()), "из", int(use.sum()),
      "| из них final > 500:", int((sm & (m.final > 500)).sum()))
print(m[sm & (m.final > 500)][["index", "Title", "culture", "v2", "ext_SI", "final", "doi"]]
      .to_string(index=False))
print("сумма (final-v2)^2 / n:", round(float(((m.final - m.v2) ** 2).sum() / len(m))))
print(m[use].assign(d=lambda d: (d.final - d.v2).abs()).sort_values("d", ascending=False)
      [["index", "Title", "culture", "v2", "ext_SI", "final", "doi"]].head(15).to_string(index=False))

m[use][["index", "Title", "culture", "v2", "ext_SI", "w", "stereo_match", "final", "doi",
        "chembl_doc"]].to_csv(ROOT / "data" / "ext_substitutions.csv", index=False)

out = m[["index", "final"]].rename(columns={"final": T})
assert len(out) == 597 and out[T].notna().all() and (out[T] >= 0).all()
old_path = ROOT / "submission_v3.csv"
if old_path.exists():                          # самопроверка: что изменилось против прошлого v3
    old = pd.read_csv(old_path).set_index("index")[T]
    diff = (out.set_index("index")[T] - old.reindex(out["index"]).values).abs()
    print("строк изменилось против прошлого submission_v3.csv:", int((diff > 1e-6).sum()))
out.to_csv(old_path, index=False)
print("сохранено submission_v3.csv")