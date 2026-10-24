#!/usr/bin/env python3
"""
E1d: winner-style non-recursive per-group LightGBM for M5 (a faithful port of the non-recursive half of the
1st place solution, M5-methods folder A1), written to run on a 16 GB machine and to score on our fixed windows.

How it follows the winner
  - one row per series and target day, from d_710 up to the end of the training window (about 3.7 million rows per store)
  - features for target day t use sales only up to day t-28: lags 28 to 42, rolling mean and std (windows 7, 14, 30, 60,
    180) shifted by 28 days, so one model serves all 28 horizons without recursion
  - price features (11), calendar features (14), item_id / dept / cat / event categoricals, mean encodings from the
    training days only (store x department, and item x state for store models or item x store for group models)
  - rows before an item's first price week are dropped, as in the winner's grid
  - settings: tweedie 1.1, learning rate 0.015, 3000 trees, 255 leaves, min 255 rows per leaf, subsample 0.5,
    feature fraction 0.5, max_bin 100, boost_from_average off, NO early stopping
  - levels: store (10 models), store x category (30), store x department (70); equal-weight average of the levels run

How it differs (state these in the write-up)
  - only the non-recursive half of the winner's six model sets is implemented, not the recursive models
  - training window ends at d_1913, so the target window is d_1914 to d_1941, not the hidden final window
  - the price aggregates (max, min, std, mean, number of price levels) use the full price file as in the winner's code;
    prices are planned inputs published ahead of time, not sales
  - tm_y, tm_m and similar calendar variables follow the winner

Two windows are produced, each from its own model trained only on days before the window:
  holdout     trained to d_1913, forecasts d_1914 to d_1941
  validation  trained to d_1885, forecasts d_1886 to d_1913 (clean, because there is no early stopping)
Nothing is selected on the holdout. As in E1c, a single global calibration factor is fitted on the validation window
only, and the calibrated forecast is used in ensembles only if its validation WRMSSE is lower.

Outputs (in --out_dir)
  pred_E1d_<level>.npy, predval_E1d_<level>.npy   per level (30490 x 28)
  pred_E1d_lightgbm.npy, predval_E1d_lightgbm.npy plain average of the levels run
  pred_E1d_lightgbm_cal.npy, predval_E1d_lightgbm_cal.npy   same, times the validation calibration factor
  results_E1d_lightgbm.txt                        plain-text log
  e1d_cache/                                      one file per finished model; a rerun skips finished models

Usage
  python m5_lightgbm_e1d.py --quick                 # one store, 60 trees, prints a time estimate for the full run
  python m5_lightgbm_e1d.py                         # store level only (10 models x 2 windows)
  python m5_lightgbm_e1d.py --levels store,store_cat,store_dept
  python m5_lightgbm_e1d.py --selftest              # synthetic data, no download

Requires m5_data_and_wrmsse.py in the same folder. CPU only. A full-size store model with 3000 trees takes a while
(run --quick first to see the estimate).
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

import lightgbm as lgb  # noqa: E402

PAD = 250
LAGS = list(range(28, 43))
ROLLS = [7, 14, 30, 60, 180]
PRICE_COLS = ["sell_price", "price_max", "price_min", "price_std", "price_mean", "price_norm",
              "price_nunique", "item_nunique", "price_momentum", "price_momentum_m", "price_momentum_y"]
CAL_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI",
            "tm_d", "tm_w", "tm_m", "tm_y", "tm_wm", "tm_dw", "tm_w_end"]
LAG_COLS = [f"sales_lag_{k}" for k in LAGS]
ROLL_COLS = [f"rolling_{s}_{w}" for w in ROLLS for s in ("mean", "std")]
ENC_COLS = ["enc_store_dept_mean", "enc_store_dept_std", "enc_item_grp_mean", "enc_item_grp_std"]
CAT_COLS = ["item_id", "dept_id", "cat_id", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
ALL_COLS = ["item_id", "dept_id", "cat_id"] + PRICE_COLS + CAL_COLS + LAG_COLS + ROLL_COLS + ENC_COLS
COL = {c: i for i, c in enumerate(ALL_COLS)}


def augment_calendar(folder, n_days, seed=1):
    rng = np.random.default_rng(seed)
    cal = pd.read_csv(os.path.join(folder, "calendar.csv"))
    dates = pd.date_range("2015-01-01", periods=n_days)
    cal["date"] = dates.strftime("%Y-%m-%d")
    cal["wday"] = dates.dayofweek + 1
    cal["month"] = dates.month
    ev = rng.random(n_days) < 0.05
    cal["event_name_1"] = np.where(ev, "EventX", None)
    cal["event_type_1"] = np.where(ev, "Cultural", None)
    for st in ["CA", "TX", "WI"]:
        cal[f"snap_{st}"] = (rng.random(n_days) < 0.3).astype(int)
    cal.to_csv(os.path.join(folder, "calendar.csv"), index=False)


# ----------------------------------------------------------------------------
# Global preparation
# ----------------------------------------------------------------------------
def prepare_calendar(cal_full, D):
    cal = cal_full.set_index("d")
    cal = cal.iloc[:D] if len(cal) >= D else cal
    dt = pd.to_datetime(cal["date"])
    out = pd.DataFrame(index=np.arange(D))
    for c in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if c in cal.columns:
            codes = pd.factorize(cal[c])[0].astype(np.float32)
            codes[codes < 0] = np.nan
        else:
            codes = np.full(D, np.nan, dtype=np.float32)
        out[c] = codes[:D]
    for st in ["CA", "TX", "WI"]:
        col = f"snap_{st}"
        out[col] = cal[col].to_numpy()[:D].astype(np.float32) if col in cal.columns else 0.0
    day = dt.dt.day.to_numpy()[:D]
    out["tm_d"] = day
    out["tm_w"] = dt.dt.isocalendar().week.to_numpy().astype(np.int16)[:D]
    out["tm_m"] = dt.dt.month.to_numpy()[:D]
    out["tm_y"] = dt.dt.year.to_numpy()[:D] - dt.dt.year.min()
    out["tm_wm"] = np.ceil(day / 7).astype(np.int8)
    out["tm_dw"] = dt.dt.dayofweek.to_numpy()[:D]
    out["tm_w_end"] = (out["tm_dw"] >= 5).astype(np.int8)
    return out[CAL_COLS].to_numpy(dtype=np.float32)


def price_table(prices, cal_full, meta):
    """Weekly price features per (store, item), computed as in the winner's preprocessing."""
    store_cat = pd.Categorical(meta["store_id"]).categories
    item_cat = pd.Categorical(meta["item_id"]).categories
    p = prices.copy()
    p["sid"] = pd.Categorical(p["store_id"], categories=store_cat).codes
    p["iid"] = pd.Categorical(p["item_id"], categories=item_cat).codes
    p = p[(p["sid"] >= 0) & (p["iid"] >= 0)].sort_values(["sid", "iid", "wm_yr_wk"]).reset_index(drop=True)
    p["sell_price"] = p["sell_price"].astype(np.float32)
    g = p.groupby(["sid", "iid"])["sell_price"]
    p["price_max"] = g.transform("max")
    p["price_min"] = g.transform("min")
    p["price_std"] = g.transform("std")
    p["price_mean"] = g.transform("mean")
    p["price_norm"] = p["sell_price"] / p["price_max"]
    p["price_nunique"] = g.transform("nunique").astype(np.float32)
    p["item_nunique"] = p.groupby(["sid", "sell_price"])["iid"].transform("nunique").astype(np.float32)
    p["price_momentum"] = p["sell_price"] / g.shift(1)
    dt = pd.to_datetime(cal_full["date"])
    wk = pd.DataFrame({"wm_yr_wk": cal_full["wm_yr_wk"], "month": dt.dt.month, "year": dt.dt.year}).drop_duplicates("wm_yr_wk")
    p = p.merge(wk, on="wm_yr_wk", how="left")
    p["price_momentum_m"] = p["sell_price"] / p.groupby(["sid", "iid", "month"])["sell_price"].transform("mean")
    p["price_momentum_y"] = p["sell_price"] / p.groupby(["sid", "iid", "year"])["sell_price"].transform("mean")
    keep = ["sid", "iid", "wm_yr_wk"] + PRICE_COLS
    for c in PRICE_COLS:
        p[c] = p[c].astype(np.float32)
    return p[keep], store_cat, item_cat


def series_triples(values, valid, te):
    """Per-series sum, sum of squares, count over valid days before te."""
    n = values.shape[0]
    s = np.zeros(n)
    s2 = np.zeros(n)
    c = np.zeros(n)
    for a in range(0, n, 4000):
        b = min(a + 4000, n)
        v = valid[a:b, :te]
        x = np.where(v, values[a:b, :te], 0.0)
        s[a:b] = x.sum(axis=1)
        s2[a:b] = (x * x).sum(axis=1)
        c[a:b] = v.sum(axis=1)
    return s, s2, c


def group_stats(key, s, s2, c):
    k = int(key.max()) + 1
    gs = np.bincount(key, weights=s, minlength=k)
    gs2 = np.bincount(key, weights=s2, minlength=k)
    gc_ = np.bincount(key, weights=c, minlength=k)
    with np.errstate(all="ignore"):
        mean = gs / gc_
        var = (gs2 - gs * gs / gc_) / (gc_ - 1)
    std = np.sqrt(np.maximum(var, 0))
    mean = np.where(gc_ > 0, mean, np.nan)
    std = np.where(gc_ > 1, std, np.nan)
    return mean[key], std[key]


def encodings(values, valid, te, meta_codes):
    """Mean encodings from days before te only. Returns dict of per-series arrays."""
    s, s2, c = series_triples(values, valid, te)
    sd_key = meta_codes["store"] * (meta_codes["dept"].max() + 1) + meta_codes["dept"]
    is_key = meta_codes["item"] * (meta_codes["state"].max() + 1) + meta_codes["state"]
    sd_m, sd_s = group_stats(sd_key, s, s2, c)
    is_m, is_s = group_stats(is_key, s, s2, c)
    with np.errstate(all="ignore"):
        it_m = s / c
        it_s = np.sqrt(np.maximum((s2 - s * s / c) / (c - 1), 0))
    it_m = np.where(c > 0, it_m, np.nan)
    it_s = np.where(c > 1, it_s, np.nan)
    return {"sd": (sd_m, sd_s), "item_state": (is_m, is_s), "item_store": (it_m, it_s)}


# ----------------------------------------------------------------------------
# Per-store feature matrix
# ----------------------------------------------------------------------------
def build_store(ser, meta_codes, values, valid, wk_idx, ptab, weeks, cal_arr, first_idx, D):
    """X3: (n_s, n_days, F) float32 for target days first_idx..D-1 of the series in `ser`."""
    n_s = len(ser)
    n_days = D - first_idx
    t_all = np.arange(first_idx, D)
    X3 = np.full((n_s, n_days, len(ALL_COLS)), np.nan, dtype=np.float32)

    X3[:, :, COL["item_id"]] = meta_codes["item"][ser][:, None]
    X3[:, :, COL["dept_id"]] = meta_codes["dept"][ser][:, None]
    X3[:, :, COL["cat_id"]] = meta_codes["cat"][ser][:, None]

    # prices: dense (n_s, n_weeks) per feature, expanded to days
    store_code = meta_codes["store"][ser[0]]
    sub = ptab[ptab["sid"] == store_code]
    pos = {int(it): j for j, it in enumerate(meta_codes["item"][ser])}
    r = np.array([pos.get(int(i), -1) for i in sub["iid"].to_numpy()])
    ok = r >= 0
    cidx = np.searchsorted(weeks, sub["wm_yr_wk"].to_numpy())
    day_week = wk_idx[t_all]
    for cname in PRICE_COLS:
        M = np.full((n_s, len(weeks)), np.nan, dtype=np.float32)
        M[r[ok], cidx[ok]] = sub[cname].to_numpy()[ok]
        X3[:, :, COL[cname]] = M[:, day_week]
    del sub

    # calendar
    for k, cname in enumerate(CAL_COLS):
        X3[:, :, COL[cname]] = cal_arr[t_all, k][None, :]

    # sales history with pre-release days as missing
    S = np.where(valid[ser], values[ser], np.nan).astype(np.float32)
    Sp = np.concatenate([np.full((n_s, PAD), np.nan, dtype=np.float32), S], axis=1)
    for L, cname in zip(LAGS, LAG_COLS):
        X3[:, :, COL[cname]] = Sp[:, PAD + first_idx - L: PAD + D - L]
    Z = np.nan_to_num(Sp).astype(np.float64)
    V = (~np.isnan(Sp)).astype(np.float64)
    cs = np.concatenate([np.zeros((n_s, 1)), np.cumsum(Z, axis=1)], axis=1)
    cs2 = np.concatenate([np.zeros((n_s, 1)), np.cumsum(Z * Z, axis=1)], axis=1)
    cv = np.concatenate([np.zeros((n_s, 1)), np.cumsum(V, axis=1)], axis=1)
    del Z, V, Sp
    end = PAD + t_all - 28 + 1
    for w in ROLLS:
        a = end - w
        with np.errstate(all="ignore"):
            sm = cs[:, end] - cs[:, a]
            sm2 = cs2[:, end] - cs2[:, a]
            cnt = cv[:, end] - cv[:, a]
            mean = sm / w
            var = (sm2 - w * mean * mean) / (w - 1)
        full = cnt == w
        X3[:, :, COL[f"rolling_mean_{w}"]] = np.where(full, mean, np.nan)
        X3[:, :, COL[f"rolling_std_{w}"]] = np.where(full, np.sqrt(np.maximum(var, 0)), np.nan)
    del cs, cs2, cv
    return X3, S[:, first_idx:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default="m5_outputs")
    ap.add_argument("--levels", default="store", help="comma list: store,store_cat,store_dept")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=1995)
    ap.add_argument("--first_day", type=int, default=710)
    ap.add_argument("--quick", action="store_true", help="one store, 60 trees, prints a time estimate")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    params = dict(
        objective="tweedie", tweedie_variance_power=1.1, metric="rmse", boosting_type="gbdt",
        subsample=0.5, subsample_freq=1, learning_rate=args.lr, num_leaves=2 ** 8 - 1,
        min_data_in_leaf=2 ** 8 - 1, feature_fraction=0.5, max_bin=100, boost_from_average=False,
        verbose=-1, seed=args.seed, num_threads=args.threads,
    )
    first_idx = args.first_day - 1
    if args.selftest:
        args.rounds, params["learning_rate"] = 30, 0.1
        params.update(num_leaves=7, min_data_in_leaf=5)
        first_idx = 60
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
    log(f"Data folder: {data_dir}")

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    for lv in levels:
        if lv not in ("store", "store_cat", "store_dept"):
            raise SystemExit(f"Unknown level {lv}")

    meta, values, dcols, calendar, prices = load_m5(data_dir)
    cal_full = load_calendar(data_dir)
    n, D = values.shape
    T = D - HORIZON
    val_o = T - HORIZON
    log(f"Series: {n} | days: {D} | holdout origin: {T} | validation origin: {val_o} | first training day: d_{first_idx + 1}")

    wk_of_day = cal_full.set_index("d").loc[dcols, "wm_yr_wk"].to_numpy()
    weeks = np.unique(cal_full["wm_yr_wk"].to_numpy())
    wk_idx = np.searchsorted(weeks, wk_of_day)
    cal_arr = prepare_calendar(cal_full.assign(date=cal_full["date"]).set_index("d").loc[dcols].reset_index(), D)

    log("Preparing price features ...")
    ptab, store_cat_levels, item_cat_levels = price_table(prices, cal_full, meta)
    meta_codes = {
        "store": pd.Categorical(meta["store_id"], categories=store_cat_levels).codes.astype(np.int64),
        "item": pd.Categorical(meta["item_id"], categories=item_cat_levels).codes.astype(np.int64),
        "dept": pd.factorize(meta["dept_id"], sort=True)[0].astype(np.int64),
        "cat": pd.factorize(meta["cat_id"], sort=True)[0].astype(np.int64),
        "state": pd.factorize(meta["state_id"], sort=True)[0].astype(np.int64),
    }
    first_week = ptab.groupby(["sid", "iid"])["wm_yr_wk"].min()
    rel = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    key = pd.MultiIndex.from_arrays([meta_codes["store"], meta_codes["item"]])
    ridx = first_week.index.get_indexer(key)
    rel_ok = ridx >= 0
    rel[rel_ok] = first_week.to_numpy()[ridx[rel_ok]]
    valid = wk_of_day[None, :] >= rel[:, None]
    values32 = values.astype(np.float32)

    ev_hold = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    ev_val = WRMSSEEvaluator(meta, values[:, :T], dcols[:T], calendar, prices)
    del prices
    gc.collect()

    windows = {"hold": T, "val": val_o}
    enc = {w: encodings(values, valid, te, meta_codes) for w, te in windows.items()}

    cache = os.path.join(args.out_dir, "e1d_cache")
    os.makedirs(cache, exist_ok=True)
    pred = {w: {lv: np.zeros((n, HORIZON), dtype=np.float32) for lv in levels} for w in windows}
    stats = {lv: [] for lv in levels}  # minutes per model

    store_ids = list(store_cat_levels)
    if args.quick:
        store_ids = store_ids[:1]
        args.rounds_full = args.rounds
        args.rounds = 60
    n_total_models = 0

    for st_i, store_name in enumerate(store_ids):
        scode = list(store_cat_levels).index(store_name)
        ser = np.where(meta_codes["store"] == scode)[0]
        groups = {}
        for lv in levels:
            if lv == "store":
                groups[lv] = {"all": np.arange(len(ser))}
            else:
                key_arr = meta_codes["cat" if lv == "store_cat" else "dept"][ser]
                groups[lv] = {int(k): np.where(key_arr == k)[0] for k in np.unique(key_arr)}

        def cfile(lv, g, w):
            return os.path.join(cache, f"{lv}_{store_name}_{g}_{w}.npz")

        X3 = None
        t_build = 0.0
        for lv in levels:
            for g, jmask in groups[lv].items():
                for w, te in windows.items():
                    cf = cfile(lv, g, w)
                    gser = ser[jmask]
                    if os.path.exists(cf) and not args.quick:
                        z = np.load(cf)
                        pred[w][lv][gser] = z["pred"]
                        stats[lv].append(float(z["mins"]))
                        continue
                    if X3 is None:
                        tb = time.time()
                        X3, Yst = build_store(ser, meta_codes, values32, valid, wk_idx, ptab, weeks, cal_arr, first_idx, D)
                        t_build = (time.time() - tb) / 60
                        log(f"{store_name}: feature matrix {X3.shape[0]} x {X3.shape[1]} x {X3.shape[2]} "
                            f"({X3.nbytes / 1e9:.2f} GB) built in {t_build:.1f} min")
                    t1 = time.time()
                    e = enc[w]
                    X3[:, :, COL["enc_store_dept_mean"]] = e["sd"][0][ser][:, None]
                    X3[:, :, COL["enc_store_dept_std"]] = e["sd"][1][ser][:, None]
                    grp = e["item_state"] if lv == "store" else e["item_store"]
                    X3[:, :, COL["enc_item_grp_mean"]] = grp[0][ser][:, None]
                    X3[:, :, COL["enc_item_grp_std"]] = grp[1][ser][:, None]

                    ntr = te - first_idx
                    vtr = valid[ser][jmask][:, first_idx:te]
                    Xtr = X3[jmask][:, :ntr, :][vtr]
                    ytr = Yst[jmask][:, :ntr][vtr]
                    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=ALL_COLS, categorical_feature=CAT_COLS,
                                      free_raw_data=True)
                    del Xtr

                    def cb(env, tag=f"{store_name} {lv} {g} {w}", t_start=t1):
                        if (env.iteration + 1) % 500 == 0:
                            log(f"    {tag}: {env.iteration + 1} trees | {(time.time() - t_start) / 60:.1f} min")

                    model = lgb.train(params, dtr, num_boost_round=args.rounds, callbacks=[cb])
                    Xpr = X3[jmask][:, ntr:ntr + HORIZON, :].reshape(-1, X3.shape[2])
                    ph = np.clip(model.predict(Xpr), 0, None).reshape(len(gser), HORIZON)
                    on_sale = valid[gser][:, te:te + HORIZON]
                    ph = np.where(on_sale, ph, 0.0)
                    mins = (time.time() - t1) / 60
                    del dtr, model, ytr, vtr
                    gc.collect()
                    if args.quick:
                        full_models = len(store_cat_levels) * len(windows) * len(levels)
                        per_tree = mins * 60 / args.rounds
                        est = (t_build * len(store_cat_levels) + mins / args.rounds * args.rounds_full * full_models) / 60
                        log(f"\nQUICK RESULT: build {t_build:.1f} min, {args.rounds} trees in {mins:.1f} min "
                            f"({per_tree:.2f} s per tree incl. dataset construction)")
                        log(f"Rough estimate for this setup ({', '.join(levels)} | {len(windows)} windows | "
                            f"{args.rounds_full} trees): {est:.1f} hours "
                            f"(the first trees include fixed setup cost, so treat this as an upper-leaning guide)")
                        log(f"NaN in predictions: {int(np.isnan(ph).sum())}")
                        return
                    np.savez(cf, pred=ph.astype(np.float32), mins=mins)
                    pred[w][lv][gser] = ph
                    stats[lv].append(mins)
                    n_total_models += 1
                    log(f"  done {store_name} | {lv} {g} | {w} | {mins:.1f} min | elapsed {(time.time() - t0) / 60:.0f} min")
        del X3
        gc.collect()

    # ---- combine, score, calibrate
    avg = {w: np.mean([pred[w][lv] for lv in levels], axis=0) for w in windows}
    cal = float(ev_val.y_test.sum() / max(avg["val"].sum(), 1e-9))
    res_h, res_v = {}, {}
    for lv in levels:
        res_h[lv] = ev_hold.score(pred["hold"][lv])
        res_v[lv] = ev_val.score(pred["val"][lv])
    res_h["average"], res_v["average"] = ev_hold.score(avg["hold"]), ev_val.score(avg["val"])
    res_h["average_cal"], res_v["average_cal"] = ev_hold.score(avg["hold"] * cal), ev_val.score(avg["val"] * cal)

    os.makedirs(args.out_dir, exist_ok=True)
    for lv in levels:
        np.save(os.path.join(args.out_dir, f"pred_E1d_{lv}.npy"), pred["hold"][lv])
        np.save(os.path.join(args.out_dir, f"predval_E1d_{lv}.npy"), pred["val"][lv])
    np.save(os.path.join(args.out_dir, "pred_E1d_lightgbm.npy"), avg["hold"].astype(np.float32))
    np.save(os.path.join(args.out_dir, "predval_E1d_lightgbm.npy"), avg["val"].astype(np.float32))
    np.save(os.path.join(args.out_dir, "pred_E1d_lightgbm_cal.npy"), (avg["hold"] * cal).astype(np.float32))
    np.save(os.path.join(args.out_dir, "predval_E1d_lightgbm_cal.npy"), (avg["val"] * cal).astype(np.float32))

    tot_h, tot_v = ev_hold.y_test.sum(), ev_val.y_test.sum()
    lines = [f"{'Model':<14}{'Models':>7}{'Holdout':>10}{'Validation':>12}{'HoldBias':>10}{'ValBias':>9}{'Minutes':>9}"]
    for nm in levels + ["average", "average_cal"]:
        if nm in levels:
            ph, pv = pred["hold"][nm], pred["val"][nm]
            mods, mn = str(len(stats[nm]) // len(windows)), f"{sum(stats[nm]):.0f}"
        else:
            f = cal if nm == "average_cal" else 1.0
            ph, pv, mods, mn = avg["hold"] * f, avg["val"] * f, "", ""
        lines.append(f"{nm:<14}{mods:>7}{res_h[nm][0]:>10.4f}{res_v[nm][0]:>12.4f}"
                     f"{ph.sum() / tot_h:>10.3f}{pv.sum() / tot_v:>9.3f}{mn:>9}")
    table = {"SeasonalNaive": ev_hold.score(seasonal_naive_forecast(ev_hold.X_train))}
    for lv in levels:
        table[lv] = res_h[lv]
    table["average"], table["average_cal"] = res_h["average"], res_h["average_cal"]
    use_cal = res_v["average_cal"][0] < res_v["average"][0]

    report = [
        "E1d WINNER-STYLE NON-RECURSIVE PER-GROUP LIGHTGBM RESULTS",
        f"Data folder: {data_dir}",
        f"Holdout: trained on days up to d_{T}, forecasts d_{T + 1} to d_{D}. "
        f"Validation: trained up to d_{val_o}, forecasts d_{val_o + 1} to d_{T}.",
        f"Levels run: {', '.join(levels)} | rounds: {args.rounds} | learning rate: {args.lr} | no early stopping",
        f"Training rows start at d_{first_idx + 1}; features use sales only up to 28 days before each target day.",
        "Only the non-recursive half of the winner's design is implemented (no recursive models).",
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
        "  Our E1b global LightGBM, L3: 0.5422 | E1c per-group (sampled rows): 0.6137, calibrated 0.5935",
        f"  This run, E1d average: {res_h['average'][0]:.4f} | calibrated: {res_h['average_cal'][0]:.4f}",
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min",
    ]
    text = "\n".join(report)
    out_path = os.path.join(args.out_dir, "results_E1d_lightgbm.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
