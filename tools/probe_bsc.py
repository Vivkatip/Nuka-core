"""Диагностика: (A) предсказания для подозрительных строк test + соседи в train,
(B) аудит lookup по smi vs flat (LOO на train + расхождения внутри flat-групп)."""
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
TGT, CAP = "SI VACV", 5000
pd.set_option("display.width", 220, "display.max_colwidth", 40)

tr = pd.read_csv(ROOT / "data" / "train_clean.csv")
te = pd.read_csv(ROOT / "data" / "test_clean.csv")
subs = {p.stem: pd.read_csv(p).set_index("index")[TGT] for p in sorted(ROOT.glob("submission*.csv"))}

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp_tr = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr.smi]
fp_te = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in te.smi]

# ---------- A. подозрительные строки ----------
WATCH = [70, 62, 320, 297, 100, 551, 271, 329, 501]
bsc_pos = list(te.index[te.culture == "BSC"])
print(f"BSC в test: {len(bsc_pos)} строк\n")

def show(pos, k=3):
    r = te.iloc[pos]
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_te[pos], fp_tr))
    nn = np.argsort(-sims)[:k]
    preds = {n: round(float(s.get(r["index"], np.nan)), 1) for n, s in subs.items()}
    print(f"pos={pos} index={r['index']} Title={r.Title} culture={r.culture} preds={preds}")
    for j in nn:
        t = tr.iloc[j]
        print(f"    sim={sims[j]:.2f} SI={t[TGT]:>9.1f} {t.culture:5s} {t.Title[:25]:25s} same_smi={t.smi == r.smi}")

print("=== WATCH (как позиции) ===")
for p in WATCH:
    if p < len(te): show(p)
print("\n=== WATCH (как значения колонки index) ===")
for p in WATCH:
    m = te.index[te["index"] == p]
    if len(m): show(m[0])

print("\n=== все BSC test: сводка предсказаний ===")
rows = []
for p in bsc_pos:
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_te[p], fp_tr))
    nn = np.argsort(-sims)[:5]
    w = sims[nn] ** 4
    rows.append({"pos": p, "index": te.at[p, "index"], "Title": te.at[p, "Title"][:20],
                 "max_sim": sims[nn[0]],
                 "knn_SI_wmean": float(np.average(tr[TGT].values[nn], weights=w)),
                 **{n: float(s.get(te.at[p, "index"], np.nan)) for n, s in subs.items()}})
print(pd.DataFrame(rows).sort_values("max_sim", ascending=False).round(1).to_string(index=False))
print("\nBSC в train: mean SI = %.0f, min = %.1f" % (tr[tr.culture == 'BSC'][TGT].mean(),
                                                    tr[tr.culture == 'BSC'][TGT].min()))

# ---------- B. аудит lookup ----------
yc = np.clip(tr[TGT].values, 0, CAP)
print("\n=== B1. flat-группы с разными стереоизомерами (одна культура) ===")
g = tr.assign(yc=yc).groupby(["flat", "culture"])
bad = []
for k, d in g:
    if d.smi.nunique() > 1:
        bad.append((k[1], d.Title.str[:18].tolist(), d[TGT].round(1).tolist()))
print("групп:", len(bad))
for b in bad[:25]: print("  ", b)

print("\n=== B2. LOO-качество lookup на train (MSE по строкам, где есть совпадение) ===")
cult_mean = pd.Series(yc).groupby(tr.culture).mean()
for key in ("smi", "flat"):
    pred = np.full(len(tr), np.nan)
    for i in range(len(tr)):
        m = (tr[key].values == tr[key].values[i]) & (tr.culture.values == tr.culture.values[i])
        m[i] = False
        if m.any(): pred[i] = yc[m].mean()
    ok = np.isfinite(pred)
    base = tr.culture.map(cult_mean).values
    print(f"{key:5s}: покрытие {ok.sum()} строк | MSE lookup {np.mean((pred[ok]-yc[ok])**2):,.0f} "
          f"| MSE culture-mean на тех же {np.mean((base[ok]-yc[ok])**2):,.0f} "
          f"| Spearman {pd.Series(pred[ok]).corr(pd.Series(yc[ok]), method='spearman'):.2f}")

print("\n=== B3. test: совпадения только по flat (не по smi) ===")
d_s = pd.DataFrame({"k": list(zip(tr.smi, tr.culture)), "y": yc}).groupby("k").y.mean().to_dict()
d_f = pd.DataFrame({"k": list(zip(tr.flat, tr.culture)), "y": yc}).groupby("k").y.mean().to_dict()
rows = []
for p in range(len(te)):
    ks, kf = (te.smi[p], te.culture[p]), (te.flat[p], te.culture[p])
    if kf in d_f and ks not in d_s:
        rows.append({"pos": p, "index": te["index"][p], "Title": te.Title[p][:20],
                     "culture": te.culture[p], "flat_lookup": d_f[kf],
                     **{n: float(s.get(te["index"][p], np.nan)) for n, s in subs.items()}})
print(f"только flat: {len(rows)}; smi-совпадений: {sum((te.smi[p], te.culture[p]) in d_s for p in range(len(te)))}")
print(pd.DataFrame(rows).round(1).to_string(index=False))