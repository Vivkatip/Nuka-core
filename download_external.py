"""
download_external.py
====================

Скрипт загрузки данных из открытых баз для хакатона IT Nuclear Hack (SI VACV).
Собирает значения IC50, CC50 и индекс селективности SI = CC50 / IC50 для молекул,
активных против вируса оспы (Vaccinia virus) и родственных ортопоксвирусов.

Источники данных:
  1. PubChem BioAssay (PUG REST) — серия AID 2221740–2221770 и 1996820–1996845.
     Статья: PMID:39165907 (ингибиторы ортопоксвирусов, Vaccinia / Cowpox /
     Ectromelia / Variola в клетках Vero).
     Публикация: https://pubmed.ncbi.nlm.nih.gov/39165907/
  2. ChEMBL — документы CHEMBL1139008, CHEMBL1629620, CHEMBL1145474, CHEMBL1146075
     (селективность и цитотоксичность против Vaccinia virus в клетках HFF, Vero,
     HEL, E6SM).
     ChEMBL API: https://www.ebi.ac.uk/chembl/api/data/

Важное исправление относительно старого скрипта v1/build_open_database.py:
  * Читаем колонку "Standard Value" / "PubChem Standard Value" / "Activity Value",
    НИКОГДА не PUBCHEM_SID (в старом скрипте PUBCHEM_SID ошибочно попадал в колонку SI,
    выдавая значения ~5.28·10^8 вместо реальных SI).
  * SI вычисляется как CC50 / IC50 (в одинаковых единицах, обычно нМ) или берется
    напрямую из измерений типа "Ratio CC50/IC50" (AID 2221755, 2221757, 2221759, 2221761).

Результат: external_data/open_database_vacv.csv со колонками:
    smiles, SI, IC50_uM, CC50_uM, source, doi
(IC50_uM и CC50_uM сохранены для справки; они НЕ используются как признаки/таргет
 при обучении модели — только для формирования внешней базы.)

Запуск:
    python download_external.py
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PUBCHEM_PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# DOI статьи по серии PubChem AID 22217xx (и частично 19968xx)
PUBMED_DOI_PMID = "PMID:39165907"
PUBMED_DOI_URL = "https://pubmed.ncbi.nlm.nih.gov/39165907/"

# Диапазоны AID, которые содержат активность против ортопоксвирусов и цитотоксичность
AID_RANGES = [
    list(range(2221740, 2221771)),   # статья PMID:39165907 (VACV, Cowpox, Ectromelia, Variola)
    list(range(1996820, 1996846)),   # старая серия (VACV Copenhagen, ectromelia, cowpox и др.)
]

# ChEMBL документы с данными по Vaccinia virus селективности / цитотоксичности
CHEMBL_DOCS = [
    "CHEMBL1139008",   # [Selectivity index] CC50 in HFF / EC50 against vaccinia
    "CHEMBL1629620",   # CC50 (HFF cytotoxicity) + EC50 (VACV in HFF) + IC50
    "CHEMBL1145474",   # Cytotoxicity against HFF cells
    "CHEMBL1146075",   # Antiviral activity against vaccinia virus in E6SM cells
]


def http_get_json(url, timeout=25, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception:
            time.sleep(0.5 * (i + 1))
    return None


def http_get_text(url, timeout=25, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception:
            time.sleep(0.5 * (i + 1))
    return None


def fetch_cids_smiles(cids):
    """Забирает IsomericSMILES для списка CID пачками по 80."""
    cids = sorted({int(c) for c in cids if str(c).isdigit() and int(c) > 0})
    out = {}
    for i in range(0, len(cids), 80):
        chunk = ",".join(map(str, cids[i:i + 80]))
        d = http_get_json(f"{PUBCHEM_PUG_BASE}/compound/cid/{chunk}/property/IsomericSMILES,CanonicalSMILES/json")
        if d and "PropertyTable" in d:
            for p in d["PropertyTable"].get("Properties", []):
                cid = int(p["CID"])
                smi = (p.get("IsomericSMILES") or p.get("CanonicalSMILES")
                       or p.get("SMILES") or p.get("ConnectivitySMILES"))
                if smi:
                    out[cid] = smi
        time.sleep(0.2)
    return out


def _to_num(val):
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(str(val).replace(">", "").replace("<", "").replace("~", "").strip())
    except (TypeError, ValueError):
        return None


def _assay_name(d):
    try:
        if "PC_AssayContainer" in d:
            return d["PC_AssayContainer"][0]["assay"]["descr"].get("name", "")
        if "PC_AssayDescription" in d:
            return d["PC_AssayDescription"].get("description", "")
    except Exception:
        pass
    return ""


def download_pubchem():
    """Скачивает все анализы из диапазонов AID, правильно читает Standard Value."""
    all_aids = [aid for rng in AID_RANGES for aid in rng]
    records = []

    for aid in all_aids:
        d = http_get_json(f"{PUBCHEM_PUG_BASE}/assay/aid/{aid}/description/json")
        if not d:
            continue
        name = _assay_name(d)

        csv_txt = http_get_text(f"{PUBCHEM_PUG_BASE}/assay/aid/{aid}/csv")
        if not csv_txt:
            continue
        try:
            df = pd.read_csv(io.StringIO(csv_txt), dtype=str)
        except Exception:
            continue
        if "PUBCHEM_CID" not in df.columns:
            continue

        data_df = df[df["PUBCHEM_CID"].str.isdigit().fillna(False)].copy()
        if data_df.empty:
            continue

        # Определяем колонку со значением (приоритет: Standard Value -> PubChem Standard Value -> IC50/EC50/CC50)
        val_col = None
        for cand in ["Standard Value", "PubChem Standard Value", "IC50", "EC50", "CC50", "Activity Value"]:
            if cand in data_df.columns:
                val_col = cand
                break
        if val_col is None:
            continue

        for _, r in data_df.iterrows():
            cid = int(r["PUBCHEM_CID"])
            stype = str(r.get("Standard Type") or r.get("Activity Type") or "").strip()
            val = _to_num(r.get(val_col))
            if val is None:
                continue
            unit = str(r.get("Standard Units") or r.get("Activity Units") or "").strip()
            records.append({
                "cid": cid,
                "stype": stype,
                "val": val,
                "unit": unit,
                "assay_name": name,
                "aid": aid,
                "source": f"PubChem AID {aid}",
                "doi": PUBMED_DOI_URL,
            })

        time.sleep(0.15)  # вежливая пауза между запросами

    return pd.DataFrame(records)


def download_chembl():
    """Скачивает активности из документов ChEMBL, распознает [Activity] и [Selectivity index]."""
    records = []
    for doc in CHEMBL_DOCS:
        # activity по документу
        url = f"{CHEMBL_API_BASE}/activity.json?document_chembl_id={doc}&limit=1000"
        d = http_get_json(url)
        if not d or "activities" not in d:
            continue
        for act in d["activities"]:
            smi = act.get("canonical_smiles")
            if not smi:
                continue
            # стандартное значение + единицы
            val = _to_num(act.get("standard_value"))
            unit = str(act.get("standard_units") or "").strip()
            mtype = str(act.get("standard_type") or "").strip()
            relation = str(act.get("standard_relation") or "").strip()
            cid = act.get("molecule_chembl_id")
            records.append({
                "cid": str(cid) if cid else None,
                "stype": mtype,
                "val": val,
                "unit": unit,
                "assay_name": f"ChEMBL {doc} / {act.get('assay_description','')[:60]}",
                "aid": None,
                "source": f"ChEMBL {doc}",
                "doi": f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/" if cid else "",
                "smiles": smi,
            })
        time.sleep(0.3)

    return pd.DataFrame(records)


def build_si_table(df_pc, df_chembl):
    """Из сырых измерений строит таблицу smiles -> SI, IC50, CC50."""
    rows = []

    # ---------- PubChem part ----------
    if len(df_pc):
        cids = df_pc["cid"].unique().tolist()
        cid_smi = fetch_cids_smiles(cids)
        df_pc["smiles"] = df_pc["cid"].map(cid_smi)
        df_pc = df_pc[df_pc["smiles"].notna()].copy()

        # Группируем по CID: собираем IC50, CC50, Ratio
        for cid, g in df_pc.groupby("cid"):
            smi = g["smiles"].iloc[0]
            ic50_vals = g[g["stype"].str.contains("IC50|EC50", case=False, na=False)]["val"].tolist()
            cc50_vals = g[g["stype"].str.contains("CC50", case=False, na=False)]["val"].tolist()
            ratio_vals = g[g["stype"].str.contains("Ratio", case=False, na=False)]["val"].tolist()

            si = None
            ic50 = np.median(ic50_vals) if ic50_vals else None
            cc50 = np.median(cc50_vals) if cc50_vals else None

            # Прямой Ratio CC50/IC50 — приоритет (это уже готовое SI)
            if ratio_vals:
                si = float(np.median(ratio_vals))
            # Иначе считаем SI = CC50 / IC50 (в одинаковых единицах)
            elif ic50 and cc50:
                si = float(cc50 / ic50)

            if si is None:
                continue

            rows.append({
                "smiles": smi,
                "SI": si,
                "IC50_uM": (ic50 / 1000.0) if ic50 else None,
                "CC50_uM": (cc50 / 1000.0) if cc50 else None,
                "source": "; ".join(g["source"].unique())[:120],
                "doi": g["doi"].iloc[0] if len(g) else "",
            })

    # ---------- ChEMBL part ----------
    if len(df_chembl):
        for _, r in df_chembl.iterrows():
            smi = r["smiles"]
            stype = r["stype"].lower()
            val = r["val"]

            # Ищем IC50/EC50 и CC50 для同じ молекулы (просто добавляем как есть, если есть прямое SI)
            if "selectivity" in stype or "ratio" in stype:
                si = float(val)
                rows.append({
                    "smiles": smi,
                    "SI": si,
                    "IC50_uM": None,
                    "CC50_uM": None,
                    "source": r["source"],
                    "doi": r["doi"],
                })

    if not rows:
        return pd.DataFrame(columns=["smiles", "SI", "IC50_uM", "CC50_uM", "source", "doi"])

    df = pd.DataFrame(rows)
    # Дедупликация: медиана SI по SMILES
    df = df.dropna(subset=["smiles", "SI"])
    df = df[(df["SI"] >= 0) & (df["SI"] <= 40000)]
    g = df.groupby("smiles")
    out = pd.DataFrame({
        "smiles": g["smiles"].first(),
        "SI": g["SI"].median(),
        "IC50_uM": g["IC50_uM"].median(),
        "CC50_uM": g["CC50_uM"].median(),
        "source": g["source"].first(),
        "doi": g["doi"].first(),
    }).reset_index(drop=True)
    return out


def main():
    out_dir = Path("external_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "open_database_vacv.csv"

    print("=" * 70)
    print("1. СКАЧИВАНИЕ PUBChem AID (серии 2221740–2221770 и 1996820–1996845)")
    print("=" * 70)
    df_pc = download_pubchem()
    print(f"  Записей из PubChem: {len(df_pc)}")

    print("\n" + "=" * 70)
    print("2. СКАЧИВАНИЕ ChEMBL документов")
    print("=" * 70)
    df_chembl = download_chembl()
    print(f"  Записей из ChEMBL: {len(df_chembl)}")

    print("\n" + "=" * 70)
    print("3. СБОРКА ТАБЛИЦЫ SI (CC50 / IC50, медиана по молекуле)")
    print("=" * 70)
    out = build_si_table(df_pc, df_chembl)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Итого уникальных молекул с SI: {len(out)}")
    print(f"  Сохранено: {out_path}")
    print("\nПример первых 10 строк:")
    print(out.head(10).to_string(index=False))
    print("\n[OK] download_external.py завершён.")


if __name__ == "__main__":
    main()
