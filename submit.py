"""
Шаг 4. Финальное обучение и генерация submission.csv.
Авторитетная версия: ПЛОСКИЙ InChIKey-14 Lookup (не строгий изомерный!).
Lookup в тест делается только из train (external шум и ухудшил скор на 5.5k в старой базе,
но с РЕМОНТИРОВАННОЙ внешней базой — реальными SI из PubChem AID — external lookup даёт
точные значения для молекул теста, которых нет в train).

Запуск:
  python submit.py --config reports/best_config.json --out submission.csv
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lookup import apply_lookup, fit_lookup, query_lookup
from models import ENSEMBLE, fit_predict
from prepare import assemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/prepared.pkl")
    ap.add_argument("--config", default="reports/best_config.json")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--trees", type=int, default=None)
    ap.add_argument("--variant", default=None, help="none | train | train+ext (переопределяет config)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print("[Конфигурация сабмита]:", cfg)

    with open(args.cache, "rb") as f:
        C = pickle.load(f)
    tr, ext, te = C["tr"], C["ext"], C["te"]

    n_te = len(te["index"])
    trees = args.trees or cfg["trees"]
    morgan = bool(cfg["morgan"])
    variant = args.variant or cfg.get("variant", "train")

    X_tr = assemble(tr, morgan)
    X_ext = assemble(ext, morgan)
    X_te = assemble(te, morgan)

    if cfg.get("ext_train", True):
        X_fit = np.vstack([X_tr, X_ext])
        y_fit = np.concatenate([np.asarray(tr["y"], float), np.asarray(ext["y"], float)])
    else:
        X_fit = X_tr
        y_fit = np.asarray(tr["y"], float)

    names = ENSEMBLE if cfg["model"] == "ens" else [cfg["model"]]
    preds = [fit_predict(m, X_fit, y_fit, X_te, seed=cfg["seed"], n_trees=trees) for m in names]
    raw = np.mean(preds, axis=0)

    if cfg.get("calib", False):
        raw = np.clip(cfg["a"] + cfg["b"] * raw, 0, None)

    # ВАЖНО: key_flat — плоский InChIKey-14 (без стереохимии), это доказало себя экспериментально
    te_keys = np.array(te["key_flat"], dtype=object)
    te_cult = np.array(te["culture"], dtype=object)

    lk_tr = query_lookup(fit_lookup(tr["key_flat"], tr["culture"], tr["y"]), te_keys, te_cult)
    lk_ext = query_lookup(fit_lookup(ext["key_flat"], ext["culture"], ext["y"]), te_keys, te_cult)

    pred = np.clip(apply_lookup(raw, lk_tr, lk_ext, variant), 0, None)

    valid = np.asarray(te["valid"], dtype=bool)
    pred[~valid] = float(np.median(y_fit))

    has_tr = ~np.isnan(lk_tr)
    has_ext = ~np.isnan(lk_ext)
    use_tr = has_tr if variant != "none" else np.zeros(n_te, dtype=bool)
    use_ext = (~has_tr & has_ext) if variant == "train+ext" else np.zeros(n_te, dtype=bool)
    use_model = ~(use_tr | use_ext)

    sub = pd.DataFrame({"index": te["index"], "SI VACV": pred})
    assert len(sub) == n_te
    assert sub["index"].is_unique
    assert np.isfinite(pred).all()

    sub.to_csv(args.out, index=False, float_format="%.6f")

    print("\n" + "=" * 60)
    print(f"[ГОТОВО!] Создан файл: {args.out}")
    print(f" - Variant: {variant}")
    print(f" - Взято из Train Lookup (плоский InChIKey-14): {int(use_tr.sum())}")
    print(f" - Взято из External Lookup:                     {int(use_ext.sum())}")
    print(f" - Предсказано моделью ML:                       {int(use_model.sum())}")
    print(f" - Медиана предсказаний:                          {np.median(pred):.2f}")
    print(f" - Среднее предсказаний:                          {pred.mean():.2f}")
    print(f" - Максимум предсказаний:                         {pred.max():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
