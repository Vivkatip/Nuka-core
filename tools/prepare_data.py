from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from data_io import load_train, load_test, TARGET

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"; OUT.mkdir(exist_ok=True)


def find(*names):
    dirs = [ROOT, ROOT / "data", Path.cwd(), Path.cwd() / "data"]
    for d in dirs:
        for n in names:
            if (d / n).exists():
                print(f"[file] {n} -> {d / n}")
                return d / n
    raise FileNotFoundError(f"Не найден {names}. Искал в: {[str(d) for d in dirs]}")


# ---------- стандартизация SMILES ----------
_lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_unch = rdMolStandardize.Uncharger()


def std(smi, iso=True):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return None
    try:
        m = _unch.uncharge(_lfc.choose(m))      # убираем соли и противоионы
        return Chem.MolToSmiles(m, isomericSmiles=iso)
    except Exception:
        return None


def scaf(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) or smi
    except Exception:
        return smi


# ---------- легальные признаки источника (Title / strain) ----------
def title_src(t):
    t = str(t).lower()
    if t.startswith("sid"):    return "pubchem"
    if t.startswith("chembl"): return "chembl"
    if t.startswith("pav"):    return "pav"
    return "other"


STRAIN_FLAGS = ["copenhagen", "western reserve", "lister", "tiantan", "cowpox",
                "ectromelia", "variola", "mutants", "ihd", "grishak"]


def add_meta(df):
    df["src"] = df["Title"].map(title_src) if "Title" in df else "other"
    s = df["strain"].fillna("").str.lower() if "strain" in df else pd.Series("", index=df.index)
    for f in STRAIN_FLAGS:
        df["st_" + f.replace(" ", "_")] = s.str.contains(f).astype(int)
    df["n_virus"] = s.str.count("virus")
    return df


def prep(df):
    df["smi"] = df["SMILES"].map(std)
    df["flat"] = df["SMILES"].map(lambda s: std(s, iso=False))   # без стереохимии
    return add_meta(df)


tr = prep(load_train(find("train_v2.csv", "train.csv")))
te = prep(load_test(find("test.csv")))

# ---------- диагностика ----------
print("\ntrain/test:", tr.shape, te.shape)
print("не распарсились: train", tr.smi.isna().sum(), "| test", te.smi.isna().sum())
print(tr[tr.smi.isna()][["Title", "SMILES"]].head(10).to_string())

y = tr[TARGET]
print("\nSI==0:", round((y == 0).mean(), 3))
print("квантили SI:\n", y.quantile([.5, .75, .9, .95, .99, .995, 1]).round(2).to_string())
print("MSE константы (mean):", round(((y - y.mean()) ** 2).mean()), "| (0):", round((y ** 2).mean()))
print("\nтоп-15 по SI:")
print(tr.nlargest(15, TARGET)[["Title", TARGET, "culture", "src"]].to_string())

# вклад хвоста в дисперсию: ключ к стратегии
srt = np.sort(y.values)[::-1]
for k in (1, 3, 5, 10, 20):
    print(f"топ-{k} дают {100 * ((srt[:k] - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum():.1f}% суммы квадратов")

print("\nпо источнику:")
print(tr.groupby("src")[TARGET].agg(["count", "mean", "median", lambda s: (s == 0).mean()]).round(2).to_string())
print("\nпо культуре:")
print(tr.groupby("culture")[TARGET].agg(["count", "mean", "median", lambda s: (s == 0).mean()]).round(2).to_string())
print("\ntest: src", te.src.value_counts().to_dict(), "| culture", te.culture.value_counts().to_dict())

# дубликаты
g = tr.groupby("smi")[TARGET].agg(["count", "min", "max"])
dup = g[g["count"] > 1]
print("\nуникальных молекул:", len(g), "| с повторами:", len(dup),
      "| конфликт (max-min>1):", int(((dup["max"] - dup["min"]) > 1).sum()))

# пересечение train/test
for key in ("smi", "flat"):
    print(f"test в train по {key}:", int(te[key].isin(set(tr[key].dropna())).sum()), "из", len(te))

# ---------- сохранение (без запрещённых колонок) ----------
tr["scaffold"] = tr.smi.fillna("").map(scaf)
tr["group"] = tr.scaffold.factorize()[0]
te["scaffold"] = te.smi.fillna("").map(scaf)
tr.to_csv(OUT / "train_clean.csv", index=False)
te.to_csv(OUT / "test_clean.csv", index=False)
print("\nсохранено в", OUT)