"""Сбор внешних данных из ChEMBL (https://www.ebi.ac.uk/chembl/). Лицензия: CC BY-SA 3.0.
Выход: data/external_chembl.csv
 smiles, IC50_uM, CC50_uM, SI, inactive, SI_gt40k, doi, source, virus, culture, chembl_molecule, chembl_doc
Ответы API кэшируются в data/cache_chembl, повторный запуск использует локальный кэш."""
import hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd, requests
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from data_io import normalize_culture

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"; OUT.mkdir(exist_ok=True)
CACHE = OUT / "cache_chembl"; CACHE.mkdir(exist_ok=True)
API = "https://www.ebi.ac.uk/chembl/api/data"
VIRUSES = ["vaccinia", "cowpox", "monkeypox", "variola"]
TERMS2 = ("vaccinia", "cowpox", "monkeypox", "variola", "ectromelia", "orthopox")
TO_UM = {"nM": 1e-3, "uM": 1.0, "µM": 1.0, "um": 1.0, "mM": 1e3, "pM": 1e-6}
S = requests.Session()


def get(path, **params):
    params.setdefault("format", "json")
    key = hashlib.md5((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    last = None
    for attempt in range(5):
        try:
            r = S.get(f"{API}/{path}", params=params, timeout=120)
            r.raise_for_status()
            j = r.json()
            f.write_text(json.dumps(j), encoding="utf-8")
            return j
        except Exception as e:
            last = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            print(f"  попытка {attempt+1}/5: {path} код={code} {type(e).__name__}: {str(e)[:160]}")
            if code is not None and 400 <= code < 500 and code != 429:
                break
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"ChEMBL недоступен: {path} {params}") from last


def paged(path, key, **params):
    params["limit"] = 1000
    offset = 0
    while True:
        j = get(path, offset=offset, **params)
        yield from j[key]
        if not j["page_meta"]["next"]:
            break
        offset += 1000


def um(row):
    f = TO_UM.get(row.get("standard_units"))
    if f is None or row.get("standard_value") is None:
        return np.nan
    return float(row["standard_value"]) * f


# ---- 1. таргеты ----
targets = {}
for v in VIRUSES:
    for t in paged("target", "targets", pref_name__icontains=v):
        targets[t["target_chembl_id"]] = (v, t["pref_name"])
    for t in paged("target", "targets", organism__icontains=v):
        targets.setdefault(t["target_chembl_id"], (v, t["pref_name"]))
print("таргетов:", len(targets))

# ---- 2. антивирусные активности (все relation) ----
_assay_cache = {}
def assay_info(aid):
    if aid not in _assay_cache:
        a = get(f"assay/{aid}")
        _assay_cache[aid] = (a.get("assay_cell_type") or "", a.get("description") or "")
    return _assay_cache[aid]

rows = []
for tid, (virus, tname) in targets.items():
    for a in paged("activity", "activities", target_chembl_id=tid,
                   standard_type__in="IC50,EC50"):
        if a.get("canonical_smiles") is None:
            continue
        cell, desc = assay_info(a["assay_chembl_id"])
        rows.append(dict(act=a["activity_id"], mol=a["molecule_chembl_id"],
                         smiles=a["canonical_smiles"],
                         doc=a["document_chembl_id"], virus=virus, rel=a["standard_relation"],
                         value_uM=um(a), culture=normalize_culture(cell + " " + desc)))
n1 = len(rows)
print("проход 1, измерений:", n1)

failed = 0
assay_ids = set()
for term in TERMS2:
    try:
        for a in paged("assay", "assays", description__icontains=term):
            assay_ids.add(a["assay_chembl_id"])
            _assay_cache[a["assay_chembl_id"]] = (a.get("assay_cell_type") or "", a.get("description") or "")
    except RuntimeError as e:
        failed += 1
        print(f"ПРЕДУПРЕЖДЕНИЕ: поиск ассеев '{term}' пропущен: {e}")
print("ассеев по описанию:", len(assay_ids))

ids = sorted(assay_ids)
for i in range(0, len(ids), 20):
    try:
        for a in paged("activity", "activities", assay_chembl_id__in=",".join(ids[i:i + 20]),
                       standard_type__in="IC50,EC50"):
            if a.get("canonical_smiles") is None:
                continue
            cell, desc = assay_info(a["assay_chembl_id"])
            rows.append(dict(act=a["activity_id"], mol=a["molecule_chembl_id"],
                             smiles=a["canonical_smiles"], doc=a["document_chembl_id"],
                             virus="vaccinia?", rel=a["standard_relation"], value_uM=um(a),
                             culture=normalize_culture(cell + " " + desc)))
    except RuntimeError as e:
        failed += 1
        print(f"ПРЕДУПРЕЖДЕНИЕ: пачка ассеев {i}-{i+20} пропущена: {e}")
print("проход 2, измерений:", len(rows) - n1, "| пропущено запросов:", failed)

if not rows:
    raise SystemExit("ChEMBL не вернул ни одного измерения")

ant = pd.DataFrame(rows).dropna(subset=["value_uM"])
ant = ant.drop_duplicates("act")
ant = ant[ant.rel.isin(["=", "<", "<=", ">", ">="])].copy()
ant["inactive"] = ant.rel.isin([">", ">="])          # IC50 > X трактуется как граница CC50/X

key = ["mol", "doc", "virus", "culture"]
has_active = ant.groupby(key)["inactive"].transform(lambda s: (~s).any())
ant = ant[~(ant["inactive"] & has_active)]

print("антивирусных измерений:", len(ant), "| из них inactive (граница):", int(ant.inactive.sum()),
      "| документов:", ant.doc.nunique())

# ---- 3. цитотоксичность из тех же документов ----
rows = []
for doc in ant.doc.unique():
    for a in paged("activity", "activities", document_chembl_id=doc,
                   standard_type__in="CC50,TC50,CC90"):
        if a["standard_relation"] not in ("=", ">", ">="):
            continue
        cell, desc = assay_info(a["assay_chembl_id"])
        rows.append(dict(mol=a["molecule_chembl_id"], doc=doc, CC50_uM=um(a),
                         culture=normalize_culture(cell + " " + desc)))
tox = pd.DataFrame(rows, columns=["mol", "doc", "CC50_uM", "culture"]).dropna(subset=["CC50_uM"])
print("измерений CC50:", len(tox))

# ---- 4. соединение ----
ant_g = (ant.groupby(["mol", "doc", "virus", "culture", "smiles", "inactive"], as_index=False)
            .value_uM.median().rename(columns={"value_uM": "IC50_uM"}))
tox_c = tox.groupby(["mol", "doc", "culture"], as_index=False).CC50_uM.median()
tox_d = tox.groupby(["mol", "doc"], as_index=False).CC50_uM.median().rename(columns={"CC50_uM": "CC50_any"})
df = ant_g.merge(tox_c, on=["mol", "doc", "culture"], how="left").merge(tox_d, on=["mol", "doc"], how="left")
df["CC50_uM"] = df.CC50_uM.fillna(df.CC50_any)
df = df.drop(columns="CC50_any")

# Строки без цитотоксичности или с нулями не могут сформировать SI
df = df[(df.CC50_uM > 0) & (df.IC50_uM > 0)]

# ---- 5. стандартизация SMILES, doi, дедупликация ----
lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
unch = rdMolStandardize.Uncharger()
def std(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(unch.uncharge(lfc.choose(m)))
    except Exception:
        return None
df["smiles"] = df.smiles.map(std)
df = df.dropna(subset=["smiles"])

doi_cache = {}
def doi(d):
    if d not in doi_cache:
        doi_cache[d] = get(f"document/{d}").get("doi")
    return doi_cache[d]
df["doi"] = df.doc.map(doi)
df["source"] = "ChEMBL"

df = df.groupby(["smiles", "culture", "virus"], as_index=False).agg(
    IC50_uM=("IC50_uM", "median"), CC50_uM=("CC50_uM", "median"), inactive=("inactive", "min"),
    doi=("doi", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
    source=("source", "first"), chembl_molecule=("mol", "first"),
    chembl_doc=("doc", lambda s: ";".join(sorted(set(s)))))

# Расчёт SI: одинаков для точных и граничных значений
df["SI"] = df.CC50_uM / df.IC50_uM
df["SI_gt40k"] = df.SI > 40000

# ---- 6. валидация ----
bad = df[~np.isfinite(df.SI) | (df.SI < 0) | df.smiles.isna()]
print("некорректных строк:", len(bad))
if len(bad):
    print(bad[["smiles", "culture", "virus", "IC50_uM", "CC50_uM", "inactive", "SI", "chembl_doc"]].head(10).to_string())
df = df.drop(bad.index)

print("строк:", len(df), "| inactive (граница):", int(df.inactive.sum()),
      "| без doi:", int((df.doi == "").sum()), "| SI>40000:", int(df.SI_gt40k.sum()),
      "| max SI:", round(df.SI.max()))
df.to_csv(OUT / "external_chembl.csv", index=False)
print("сохранено", OUT / "external_chembl.csv")