"""
Шаг 3. Честная кросс-валидация: плоский InChIKey-14 Lookup + калибровка под MSE.
Запуск:
  python cv.py --repeats 3 --trees 300 --ext-train 1
"""
from __future__ import annotations
import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold

from lookup import apply_lookup, fit_lookup, query_lookup
from models import ENSEMBLE, MODELS, fit_predict
from prepare import assemble

warnings.filterwarnings("ignore")

MODES = ["group", "random"]
VARIANTS = ["none", "train", "train+ext"]
MODEL_NAMES = list(MODELS) + ["ens"]


def make_splits(mode, y, groups, folds, seed):
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X0 = np.zeros(len(y))
    if mode == "group":
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(cv.split(X0, bins, groups))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(cv.split(X0, bins))


def fit_calibration(P, y):
    """Линейная оптимизация под MSE: y ≈ a + b*P."""
    p = P.ravel()
    if np.std(p) < 1e-9:
        return 0.0, 1.0, False
    yy = np.tile(y, P.shape[0])
    X = np.c_[np.ones_like(p), p]
    a, b = np.linalg.lstsq(X, yy, rcond=None)[0]
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0 or b > 5.0:
        return 0.0, 1.0, False
    return float(a), float(b), True


def honest_calibrate(P, y, seed, k=5):
    out = np.empty_like(P)
    for r in range(P.shape[0]):
        for tr_i, te_i in KFold(k, shuffle=True, random_state=seed + r).split(P[r]):
            a, b, ok = fit_calibration(P[r, tr_i][None, :], y[tr_i])
            if ok:
                out[r, te_i] = np.clip(a + b * P[r, te_i], 0, None)
            else:
                out[r, te_i] = P[r, te_i]
    return out


def mse_rows(P, y):
    return ((P - y[None, :]) ** 2).mean(axis=1)


def build_arrays(tr, ext, morgan):
    X_all = np.vstack([assemble(tr, morgan), assemble(ext, morgan)])
    y_all = np.concatenate([tr["y"], ext["y"]]).astype(float)
    keys_all = np.array(list(tr["key_flat"]) + list(ext["key_flat"]), dtype=object)
    cult_all = np.array(list(tr["culture"]) + list(ext["culture"]), dtype=object)
    return X_all, y_all, keys_all, cult_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/prepared.pkl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--morgan", type=int, default=0)
    ap.add_argument("--ext-train", type=int, default=1)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R, F = args.repeats, args.folds

    with open(args.cache, "rb") as f:
        C = pickle.load(f)
    tr, ext = C["tr"], C["ext"]
    n_tr, n_ext = len(tr["y"]), len(ext["y"])
    X_all, y_all, keys_all, cult_all = build_arrays(tr, ext, args.morgan)
    y_tr = y_all[:n_tr]
    ext_idx = np.arange(n_tr, n_tr + n_ext)

    print(f"[cv] train: {n_tr}, external: {n_ext}, признаков: {X_all.shape[1]}, ext_train={args.ext_train}")

    RAW = {m: {n: np.full((R, n_tr), np.nan) for n in MODELS} for m in MODES}
    LKT = {m: np.full((R, n_tr), np.nan) for m in MODES}
    LKE = {m: np.full((R, n_tr), np.nan) for m in MODES}

    t0 = time.time()
    for mode in MODES:
        for r in range(R):
            splits = make_splits(mode, y_tr, tr["groups"], F, args.seed + 100 * r)
            for f, (tr_rows, va_rows) in enumerate(splits):
                va_keys = {k for k in keys_all[va_rows] if k is not None}
                keep = np.array([keys_all[i] not in va_keys for i in ext_idx], dtype=bool)
                ext_keep = ext_idx[keep]

                lk_tr = fit_lookup(keys_all[tr_rows], cult_all[tr_rows], y_all[tr_rows])
                lk_ext = fit_lookup(keys_all[ext_idx], cult_all[ext_idx], y_all[ext_idx])
                LKT[mode][r, va_rows] = query_lookup(lk_tr, keys_all[va_rows], cult_all[va_rows])
                LKE[mode][r, va_rows] = query_lookup(lk_ext, keys_all[va_rows], cult_all[va_rows])

                fit_idx = np.r_[tr_rows, ext_keep] if args.ext_train else tr_rows
                fit_idx = fit_idx.astype(int)
                for name in MODELS:
                    RAW[mode][name][r, va_rows] = fit_predict(
                        name, X_all[fit_idx], y_all[fit_idx], X_all[va_rows],
                        seed=args.seed + 100 * r + f, n_trees=args.trees
                    )
        print(f"[cv] {mode}: готов ({time.time() - t0:.0f} с)")

    for mode in MODES:
        RAW[mode]["ens"] = np.mean([RAW[mode][n] for n in ENSEMBLE], axis=0)

    CAL = {}
    for name in MODEL_NAMES:
        if name == "mean":
            CAL[name] = {"a": 0.0, "b": 1.0, "used": False}
            continue
        a, b, ok = fit_calibration(RAW["group"][name], y_tr)
        CAL[name] = {"a": a, "b": b, "used": ok}

    rows = []
    for mode in MODES:
        for name in MODEL_NAMES:
            calibs = [0, 1] if CAL[name]["used"] else [0]
            for calib in calibs:
                P = RAW[mode][name]
                if calib:
                    P = honest_calibrate(P, y_tr, seed=args.seed)
                for v in VARIANTS:
                    Q = apply_lookup(P, LKT[mode], LKE[mode], v)
                    m = mse_rows(Q, y_tr)
                    rows.append({
                        "mode": mode, "model": name, "calib": calib, "variant": v,
                        "mse_mean": m.mean(),
                    })

    df = pd.DataFrame(rows)
    key = ["model", "calib", "variant"]
    wide = df.pivot_table(index=key, columns="mode", values="mse_mean")
    wide["score"] = wide[MODES].mean(axis=1)
    wide = wide.sort_values("score")
    df.to_csv(out / "cv_results.csv", index=False)

    print("\n=== Топ-10 по CV score (MSE, меньше лучше) ===")
    print(wide.reset_index().round(1).head(10).to_string(index=False))

    best = wide.index[0]
    model, calib, variant = best[0], bool(best[1]), best[2]
    cfg = {
        "model": model, "calib": calib, "variant": variant,
        "a": CAL[model]["a"], "b": CAL[model]["b"],
        "ext_train": bool(args.ext_train), "morgan": bool(args.morgan),
        "trees": args.trees, "seed": args.seed,
        "folds": F, "repeats": R,
        "cv_score_mse": float(wide.iloc[0]["score"]),
    }
    (out / "best_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ЛУЧШИЙ КОНФИГ]: {cfg}")


if __name__ == "__main__":
    main()
