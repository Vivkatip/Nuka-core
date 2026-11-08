"""Прямой запрос к ChEMBL: тековиримат, цидофовир, бринцидофовир.
Печатает все EC50/IC50/CC50 с relation, единицами, клетками, документом. Сохраняет data/probe_teco.csv"""
import time
from pathlib import Path
import pandas as pd
import requests

API = "https://www.ebi.ac.uk/chembl/api/data"
ROOT = Path(__file__).resolve().parent
(ROOT / "data").mkdir(exist_ok=True)
S = requests.Session()
TYPES = "IC50,EC50,CC50,TC50,CC90,IC90,EC90,LC50"


def get(path, **p):
    p.setdefault("format", "json")
    for a in range(5):
        try:
            r = S.get(f"{API}/{path}", params=p, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"ChEMBL недоступен: {path} {p}")


def find_ids(name):
    ids = {}
    for m in get("molecule", pref_name__iexact=name, limit=100)["molecules"]:
        ids[m["molecule_chembl_id"]] = m.get("pref_name")
    try:
        for m in get("molecule/search", q=name, limit=20)["molecules"]:
            ids.setdefault(m["molecule_chembl_id"], m.get("pref_name"))
    except RuntimeError:
        pass
    return ids


rows = []
for name in ("TECOVIRIMAT", "ST-246", "CIDOFOVIR", "BRINCIDOFOVIR"):
    ids = find_ids(name)
    print("\n==", name, ids)
    for mid in ids:
        off = 0
        while True:
            j = get("activity", molecule_chembl_id=mid, standard_type__in=TYPES, limit=1000, offset=off)
            for a in j["activities"]:
                rows.append(dict(
                    query=name, mol=mid, type=a["standard_type"], rel=a["standard_relation"],
                    value=a["standard_value"], units=a["standard_units"],
                    cell=(a.get("assay_cell_type") or ""), target=(a.get("target_pref_name") or ""),
                    organism=(a.get("target_organism") or ""),
                    desc=(a.get("assay_description") or ""), doc=a["document_chembl_id"],
                    assay=a["assay_chembl_id"]))
            if not j["page_meta"]["next"]:
                break
            off += 1000

df = pd.DataFrame(rows).drop_duplicates()
df.to_csv(ROOT / "data" / "probe_teco.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 500)
if df.empty:
    print("ничего не найдено")
else:
    for q, g in df.groupby("query"):
        print(f"\n===== {q}: {len(g)} измерений =====")
        print(g.sort_values(["doc", "type"])[
            ["mol", "type", "rel", "value", "units", "cell", "target", "doc", "desc"]].to_string(index=False))
    print("\nrelation по типам:\n", df.groupby(["query", "type", "rel"]).size())
    print("\nединицы:\n", df.groupby(["query", "units"]).size())