"""
fast_xray.py
Мгновенный рентген данных и предсказаний (работает 2 секунды):
1. Показывает все молекулы-монстры в train_v2.csv (SI >= 100).
2. Показывает, какие большие значения попали в 54 строки Train Lookup на тесте.
3. Показывает максимальные предсказания модели на оставшихся 543 строках теста.
4. Находит в test.csv (среди 543 строк вне lookup) ближайших структурных и серийных родственников монстров из train!
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from lookup import fit_lookup, query_hierarchical_lookup, query_lookup
from data import TARGET, load_test, load_train


def tanimoto(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def main():
    with open("cache/prepared.pkl", "rb") as f:
        C = pickle.load(f)
    tr, ext, te = C["tr"], C["ext"], C["te"]

    tr_raw = load_train("train_v2.csv")
    te_raw = load_test("test.csv")

    y_tr = np.asarray(tr["y"], float)

    print("=" * 75)
    print("1. ТОП-25 САМЫХ АКТИВНЫХ МОЛЕКУЛ В TRAIN_V2.CSV (НАШИ МОНСТРЫ)")
    print("=" * 75)
    cols_tr = [c for c in ["Title", "culture", "strain", TARGET, "SMILES"] if c in tr_raw.columns]
    top_tr = tr_raw.sort_values(TARGET, ascending=False)[cols_tr].head(25).copy()
    top_tr["SMILES"] = top_tr["SMILES"].str.slice(0, 38)
    top_tr["strain"] = top_tr["strain"].fillna("").str.slice(0, 28)
    print(top_tr.to_string(index=False))

    # Проверяем 54 совпадения Lookup на тесте
    te_iso = np.array(te["key_iso"], dtype=object)
    te_flat = np.array(te["key_flat"], dtype=object)
    te_cult = np.array(te["culture"], dtype=object)

    lk_flat = query_lookup(fit_lookup(tr["key_flat"], tr["culture"], y_tr), te_flat, te_cult)
    lk_hier = query_hierarchical_lookup(tr["key_iso"], tr["key_flat"], tr["culture"], y_tr, te_iso, te_flat, te_cult)

    has_lk = ~np.isnan(lk_flat)
    print("\n" + "=" * 75)
    print(f"2. ЧТО ПОПАЛО В 54 СТРОКИ TRAIN LOOKUP В TEST.CSV (ТОП-15 ПО SI)")
    print("=" * 75)
    lk_df = pd.DataFrame({
        "test_idx": te["index"][has_lk],
        "te_Title": te_raw.loc[has_lk, "Title"].values,
        "te_cult": te_cult[has_lk],
        "exact_iso_match": te_iso[has_lk] == np.array([
            tr["key_iso"][np.where(np.array(tr["key_flat"]) == kf)[0][0]]
            for kf in te_flat[has_lk]
        ]),
        "SI_flat": np.round(lk_flat[has_lk], 2),
        "SI_hier": np.round(lk_hier[has_lk], 2),
    }).sort_values("SI_flat", ascending=False)
    print(lk_df.head(15).to_string(index=False))

    # Проверяем, что предсказывает модель на остальных 543 строках в submission_attempt1.csv
    sub = pd.read_csv("submission_attempt1.csv")["SI VACV"].to_numpy()
    sub_ml = sub[~has_lk]
    print("\n" + "=" * 75)
    print("3. ЧТО ПРЕДСКАЗАЛА МОДЕЛЬ НА ОСТАЛЬНЫХ 543 СТРОКАХ ТЕСТА (ВНЕ LOOKUP)")
    print("=" * 75)
    print(f" - Минимум на 543 строках  : {sub_ml.min():.2f}")
    print(f" - Медиана на 543 строках  : {np.median(sub_ml):.2f}")
    print(f" - 90-й перцентиль         : {np.percentile(sub_ml, 90):.2f}")
    print(f" - 99-й перцентиль         : {np.percentile(sub_ml, 99):.2f}")
    print(f" - МАКСИМУМ на 543 строках : {sub_ml.max():.2f}  <-- СРАВНИ С МОНСТРАМИ В TRAIN (500..36000)!")

    # 4. Ищем среди 543 строк теста ближайших родственников монстров из train (SI >= 100)
    print("\n" + "=" * 75)
    print("4. РОДСТВЕННИКИ МОНСТРОВ (SI >= 100) СРЕДИ 543 НЕИЗВЕСТНЫХ СТРОК ТЕСТА")
    print("=" * 75)
    fp_tr = np.hstack([tr["fp"].astype(np.float32), tr["maccs"].astype(np.float32)])
    fp_te = np.hstack([te["fp"].astype(np.float32), te["maccs"].astype(np.float32)])
    S = tanimoto(fp_te, fp_tr)

    monster_idx = np.where(y_tr >= 100.0)[0]
    rows_rel = []
    for i in range(len(te["index"])):
        if has_lk[i]:
            continue
        sims_m = S[i, monster_idx]
        best_pos = int(np.argmax(sims_m))
        best_sim_m = float(sims_m[best_pos])
        tr_j = int(monster_idx[best_pos])

        # Также найдем самого близкого соседа вообще во всём train
        best_any_j = int(np.argmax(S[i]))
        best_sim_any = float(S[i, best_any_j])

        if best_sim_m >= 0.55:
            rows_rel.append({
                "te_idx": int(te["index"][i]),
                "te_Title": str(te_raw["Title"].iloc[i])[:18],
                "te_cult": te_cult[i],
                "cur_pred": round(float(sub[i]), 1),
                "sim_to_monster": round(best_sim_m, 3),
                "monster_SI": round(float(y_tr[tr_j]), 1),
                "monster_cult": tr["culture"][tr_j],
                "max_sim_any": round(best_sim_any, 3),
                "nearest_any_SI": round(float(y_tr[best_any_j]), 1),
            })

    df_rel = pd.DataFrame(rows_rel).sort_values(["sim_to_monster", "monster_SI"], ascending=False)
    print(f"Всего найдено близких родственников монстров (sim >= 0.55): {len(df_rel)}")
    print(df_rel.head(30).to_string(index=False))


if __name__ == "__main__":
    main()