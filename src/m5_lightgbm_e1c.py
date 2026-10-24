#!/usr/bin/env python3
"""
E1c: hierarchical-ensemble LightGBM for M5 (the per-group modelling idea of the M5 winner and of Yang et al.).

Same data table as E1b L3 (52 weekly origins, 6 sampled horizons per series and origin, the 34 base features
plus price, event and context features, all computed from data before the origin, no recursion, no holdout
leakage). The difference is that models are trained per group instead of one global model:

  store         10 models, one per store
  store_cat     30 models, one per store x category
  store_dept    70 models, one per store x department

Each model is early-stopped on a validation origin (28 days before the end of the training window). The three
levels each give a full forecast for all 30,490 series, and the final E1c forecast is the equal-weight average of
the three levels (as in the M5 winner, who averaged equally). Nothing is selected on the holdout.

Settings fixed in advance (not tuned): tweedie 1.1, learning rate 0.05, early stopping 100, at most 2000 rounds,
leaves / min rows per leaf: store 127 / 200, store_cat 63 / 100, store_dept 31 / 100. Features are the E1b L3 set.
Rows are unweighted. Note for the write-up: the choice of L3 features over the pre-specified L4 (dollar weights)
came from the E1b ablation and is disclosed as a post hoc decision.

Calibration variant (reported separately): a single global multiplier = total actual / total forecast on the
validation window, applied to the holdout forecast. It uses validation data only. Whether the calibrated or the
plain forecast goes into later ensembles is decided by validation WRMSSE only.

Validation-window forecasts (predval_*) come from the same models that were early-stopped on that window, so only
the number of boosting rounds per model has seen it. This is a small optimism for weight fitting and should be
mentioned in the limitations.

Outputs (in --out_dir)
  pred_E1c_store.npy, pred_E1c_store_cat.npy, pred_E1c_store_dept.npy   holdout forecasts of each level
  pred_E1c_lightgbm.npy                                                 equal-weight average of the three levels
  pred_E1c_lightgbm_cal.npy                                             same, times the validation calibration factor
  predval_E1c_<same names>.npy                                          validation-window forecasts (d_1886 to d_1913)
  results_E1c_lightgbm.txt                                              plain-text log
  e1c_cache/                                                            per-group results; a rerun skips finished groups

Usage
  python m5_lightgbm_e1c.py              # full run (resumes from e1c_cache if interrupted)
  python m5_lightgbm_e1c.py --quick      # smaller table and fewer rounds, for timing
  python m5_lightgbm_e1c.py --selftest   # synthetic data, no download

Requires m5_data_and_wrmsse.py, m5_lightgbm_e1.py and m5_lightgbm_e1b.py in the same folder.
"""
import argparse
import gc
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import (  # noqa: E402
    HORIZON,
    WRMSSEEvaluator,
    find_file,
    format_table,
    load_m5,
    load_calendar,
    make_synthetic,
    seasonal_naive_forecast,
)
from m5_lightgbm_e1 import CAT_FEATURES, augment_calendar  # noqa: E402
from m5_lightgbm_e1b import (  # noqa: E402
    CONTEXT_NAMES,
    EVENT_NAMES,
    PRICE_NAMES,
    FeatureBuilderB,
)

import lightgbm as lgb  # noqa: E402

LEVEL_PARAMS = {
    "store": dict(num_leaves=127, min_data_in_leaf=200),
    "store_cat": dict(num_leaves=63, min_data_in_leaf=100),
    "store_dept": dict(num_leaves=31, min_data_in_leaf=100),
}


def series_groups(meta, fb):
    return {
        "store": fb.store_code,
        "store_cat": pd.factorize(meta["store_id"] + "|" + meta["cat_id"], sort=True)[0],
        "store_dept": fb.gid_sd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default="m5_outputs")
    ap.add_argument("--origins", type=int, default=52)
    ap.add_argument("--n_h", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    base = dict(
        objective="tweedie", tweedie_variance_power=1.1, metric="rmse", learning_rate=0.05,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=0.1, max_bin=255,
        num_threads=args.threads, seed=args.seed, verbose=-1,
    )
    level_params = {k: dict(v) for k, v in LEVEL_PARAMS.items()}
    if args.quick:
        args.origins, args.n_h, args.rounds = 12, 4, 400
    if args.selftest:
        args.origins, args.n_h, args.rounds = 8, 4, 40
        for v in level_params.values():
            v.update(num_leaves=7, min_data_in_leaf=5)
        tmp = tempfile.mkdtemp()
        make_synthetic(tmp, n_days=240)
        augment_calendar(tmp, 240)
        data_dir = tmp
    elif args.data_dir:
        data_dir = args.data_dir
    elif os.path.isdir("m5_data"):
        data_dir = "m5_data"
    else:
        import kagglehub
        data_dir = kagglehub.competition_download("m5-forecasting-accuracy")
    print(f"Data folder: {data_dir}", flush=True)

    meta, values, dcols, calendar, prices = load_m5(data_dir)
    cal_full = load_calendar(data_dir)
    n, D = values.shape
    T = D - HORIZON
    print(f"Series: {n} | days: {D} | train days: {T}", flush=True)

    ev_hold = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    ev_val = WRMSSEEvaluator(meta, values[:, :T], dcols[:T], calendar, prices)
    fb = FeatureBuilderB(meta, values, dcols, cal_full, prices, T)
    del prices
    gc.collect()
    rng = np.random.default_rng(args.seed)
    groups = series_groups(meta, fb)

    val_o = T - HORIZON
    last_train_o = val_o - HORIZON
    origins = [o for o in (last_train_o - 7 * k for k in range(args.origins)) if o >= 28]
    print(f"Training origins: {len(origins)} (from {origins[-1]} to {origins[0]}) | validation origin: {val_o}", flush=True)

    Xs, ys = [], []
    for o in origins:
        H = rng.random((n, HORIZON)).argsort(axis=1)[:, : args.n_h] + 1
        Xs.append(fb.build(o, H))
        ys.append(fb.targets(o, H))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    gc.collect()
    sid = np.tile(np.repeat(np.arange(n, dtype=np.int32), args.n_h), len(origins))
    print(f"Training table: {X.shape[0]:,} rows x {X.shape[1]} features ({X.nbytes / 1e9:.2f} GB)", flush=True)

    Hv = np.tile(np.arange(1, HORIZON + 1), (n, 1))
    Xv = fb.build(val_o, Hv)
    yv = fb.targets(val_o, Hv)
    Xp = fb.build(T, Hv)

    names = fb.names + fb.extra_names
    idx = {nm: i for i, nm in enumerate(names)}
    cols = [idx[nm] for nm in fb.names + PRICE_NAMES + EVENT_NAMES + CONTEXT_NAMES]
    sub_names = [names[i] for i in cols]
    cats = [c for c in CAT_FEATURES if c in sub_names]
    X, Xv, Xp = X[:, cols], Xv[:, cols], Xp[:, cols]

    cache = os.path.join(args.out_dir, "e1c_cache")
    os.makedirs(cache, exist_ok=True)
    pred_h = {lv: np.zeros((n, HORIZON), dtype=np.float32) for lv in groups}
    pred_v = {lv: np.zeros((n, HORIZON), dtype=np.float32) for lv in groups}
    stats = {lv: [] for lv in groups}  # (best_iter, minutes)

    for lv, gser in groups.items():
        gid_row = gser[sid]
        order = np.argsort(gid_row, kind="stable")
        sorted_g = gid_row[order]
        n_groups = int(gser.max()) + 1
        bounds = np.searchsorted(sorted_g, np.arange(n_groups + 1))
        print(f"\n=== Level {lv}: {n_groups} models ===", flush=True)
        for g in range(n_groups):
            cfile = os.path.join(cache, f"{lv}_{g:03d}.npz")
            ser = np.where(gser == g)[0]
            if os.path.exists(cfile):
                z = np.load(cfile)
                pred_h[lv][ser] = z["hold"]
                pred_v[lv][ser] = z["val"]
                stats[lv].append((int(z["best_it"]), float(z["mins"])))
                continue
            t1 = time.time()
            rows = order[bounds[g]:bounds[g + 1]]
            vrows = (ser[:, None] * HORIZON + np.arange(HORIZON)[None, :]).ravel()
            dtr = lgb.Dataset(X[rows], label=y[rows], feature_name=sub_names,
                              categorical_feature=cats, free_raw_data=True)
            dva = lgb.Dataset(Xv[vrows], label=yv[vrows], reference=dtr)
            params = dict(base, **level_params[lv])
            model = lgb.train(params, dtr, num_boost_round=args.rounds, valid_sets=[dva], valid_names=["val"],
                              callbacks=[lgb.early_stopping(100, verbose=False)])
            ph = np.clip(model.predict(Xp[vrows], num_iteration=model.best_iteration), 0, None).reshape(len(ser), HORIZON)
            pv = np.clip(model.predict(Xv[vrows], num_iteration=model.best_iteration), 0, None).reshape(len(ser), HORIZON)
            mins = (time.time() - t1) / 60
            np.savez(cfile, hold=ph.astype(np.float32), val=pv.astype(np.float32),
                     best_it=model.best_iteration, mins=mins)
            pred_h[lv][ser] = ph
            pred_v[lv][ser] = pv
            stats[lv].append((model.best_iteration, mins))
            del dtr, dva, model
            gc.collect()
            if g % max(1, n_groups // 10) == 0 or g == n_groups - 1:
                print(f"  {lv} {g + 1}/{n_groups} | best iteration {stats[lv][-1][0]} | {mins:.1f} min | "
                      f"elapsed {(time.time() - t0) / 60:.0f} min", flush=True)

    # ---- combine, score, calibrate
    avg_h = np.mean([pred_h[lv] for lv in groups], axis=0)
    avg_v = np.mean([pred_v[lv] for lv in groups], axis=0)
    cal = float(ev_val.y_test.sum() / max(avg_v.sum(), 1e-9))
    cal_h = avg_h * cal
    cal_v = avg_v * cal

    results_h = {lv: ev_hold.score(pred_h[lv]) for lv in groups}
    results_v = {lv: ev_val.score(pred_v[lv]) for lv in groups}
    results_h["average"] = ev_hold.score(avg_h)
    results_v["average"] = ev_val.score(avg_v)
    results_h["average_cal"] = ev_hold.score(cal_h)
    results_v["average_cal"] = ev_val.score(cal_v)

    os.makedirs(args.out_dir, exist_ok=True)
    for lv in groups:
        np.save(os.path.join(args.out_dir, f"pred_E1c_{lv}.npy"), pred_h[lv])
        np.save(os.path.join(args.out_dir, f"predval_E1c_{lv}.npy"), pred_v[lv])
    np.save(os.path.join(args.out_dir, "pred_E1c_lightgbm.npy"), avg_h.astype(np.float32))
    np.save(os.path.join(args.out_dir, "predval_E1c_lightgbm.npy"), avg_v.astype(np.float32))
    np.save(os.path.join(args.out_dir, "pred_E1c_lightgbm_cal.npy"), cal_h.astype(np.float32))
    np.save(os.path.join(args.out_dir, "predval_E1c_lightgbm_cal.npy"), cal_v.astype(np.float32))

    tot_h, tot_v = ev_hold.y_test.sum(), ev_val.y_test.sum()
    lines = [f"{'Model':<14}{'Models':>7}{'Holdout':>10}{'Validation':>12}{'HoldBias':>10}{'ValBias':>9}{'AvgIter':>9}{'Min':>7}"]
    order_names = list(groups) + ["average", "average_cal"]
    for nm in order_names:
        ph = avg_h if nm == "average" else cal_h if nm == "average_cal" else pred_h[nm]
        pv = avg_v if nm == "average" else cal_v if nm == "average_cal" else pred_v[nm]
        if nm in groups:
            nm_models = len(stats[nm])
            it = f"{np.mean([s[0] for s in stats[nm]]):.0f}"
            mn = f"{sum(s[1] for s in stats[nm]):.1f}"
        else:
            nm_models, it, mn = "", "", ""
        lines.append(f"{nm:<14}{nm_models!s:>7}{results_h[nm][0]:>10.4f}{results_v[nm][0]:>12.4f}"
                     f"{ph.sum() / tot_h:>10.3f}{pv.sum() / tot_v:>9.3f}{it:>9}{mn:>7}")

    table = {"SeasonalNaive": ev_hold.score(seasonal_naive_forecast(ev_hold.X_train))}
    for lv in groups:
        table[lv] = results_h[lv]
    table["average"] = results_h["average"]
    table["average_cal"] = results_h["average_cal"]

    use_cal = results_v["average_cal"][0] < results_v["average"][0]
    report = [
        "E1c HIERARCHICAL-ENSEMBLE LIGHTGBM RESULTS",
        f"Data folder: {data_dir}",
        f"Training window: d_1 to d_{T} | Holdout (28 days): d_{T + 1} to d_{D} | validation window: d_{val_o + 1} to d_{T}",
        f"Training origins: {len(origins)} weekly steps | horizons sampled per series per origin: {args.n_h}",
        f"Features: {len(sub_names)} (E1b L3 set) | rows unweighted | objective tweedie 1.1",
        "Models per level: store 10, store_cat 30, store_dept 70. Final forecast = equal-weight average of the three levels.",
        "Each model is early-stopped on the validation origin. Nothing is selected on the holdout.",
        f"Calibration factor (validation actual / forecast total): {cal:.4f}",
        f"Rule: ensembles use the calibrated forecast only if its validation WRMSSE is lower. Result: "
        f"{'use calibrated' if use_cal else 'use plain average'}.",
        "Bias = total forecast / total actual (1.000 is unbiased).",
        "",
        "RESULTS (WRMSSE, lower is better)",
        *lines,
        "",
        "WRMSSE BY LEVEL (holdout)",
        format_table(ev_hold, table),
        "",
        "REFERENCE VALUES (published, window to be confirmed)",
        "  Original M5 winner (hidden test window): 0.520",
        "  Yang et al. Kaggle Top-1 reproduction: 0.5230",
        "  Yang et al. LightGBM + PatchTST: 0.4989",
        "  Our E1b global LightGBM, L3 (unweighted, all features): 0.5422",
        f"  This run, E1c average: {results_h['average'][0]:.4f} | calibrated: {results_h['average_cal'][0]:.4f}",
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min",
    ]
    text = "\n".join(report)
    out_path = os.path.join(args.out_dir, "results_E1c_lightgbm.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
