#!/usr/bin/env python3
"""
E1: global LightGBM baseline for M5 (memory-aware, single script).

Design
  - One global LightGBM model with a horizon feature h (1..28), so no recursive
    forecasting is needed.
  - Training rows are (series, forecast origin, horizon). Features are computed
    only from data observed before the origin, plus known future inputs for the
    target day (calendar, SNAP, events, price).
  - Origins step back every 7 days. For each origin, n_h random horizons per
    series are sampled, which keeps the training table small enough for 16 GB RAM.
  - Validation: the origin 28 days before the training end. Its targets lie after
    every training target, and the 28-day test holdout is never touched.
  - Objective: Tweedie (variance power 1.1), as commonly used on M5.

Outputs (in --out_dir)
  pred_E1_lightgbm.npy        forecasts, shape (30490, 28), same row order as the sales file
  results_E1_lightgbm.txt     plain-text log with 12-level WRMSSE table and settings

Usage
  python m5_lightgbm_e1.py                  # full run
  python m5_lightgbm_e1.py --quick          # smaller, faster first run
  python m5_lightgbm_e1.py --selftest       # synthetic data, no download

Requires m5_data_and_wrmsse.py in the same folder.
"""
import argparse
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
    make_synthetic,
    seasonal_naive_forecast,
)

import lightgbm as lgb  # noqa: E402

CAT_FEATURES = ["dept", "cat", "store", "state", "event1", "event_type1", "wday"]


# ----------------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------------
def build_price_matrix(meta, calendar, prices, dcols):
    wk = calendar.set_index("d")["wm_yr_wk"].loc[dcols].to_numpy()
    weeks = np.unique(wk)
    widx = np.searchsorted(weeks, wk)
    p = prices[prices["wm_yr_wk"].isin(weeks)]
    key = pd.MultiIndex.from_frame(meta[["item_id", "store_id"]])
    r = key.get_indexer(pd.MultiIndex.from_frame(p[["item_id", "store_id"]]))
    c = np.searchsorted(weeks, p["wm_yr_wk"].to_numpy())
    ok = r >= 0
    Pw = np.full((len(meta), len(weeks)), np.nan, dtype=np.float32)
    Pw[r[ok], c[ok]] = p["sell_price"].to_numpy(dtype=np.float32)[ok]
    return Pw[:, widx]


class FeatureBuilder:
    """Builds feature rows for (origin o, horizons H). Uses only columns < T of the sales matrix."""

    def __init__(self, meta, values, dcols, cal_full, prices, T):
        self.n = values.shape[0]
        self.T = T
        S = values[:, :T]
        self.S32 = S.astype(np.float32)
        self.cs = np.zeros((self.n, T + 1))
        self.cs[:, 1:] = np.cumsum(S, axis=1)
        self.cs2 = np.zeros((self.n, T + 1))
        self.cs2[:, 1:] = np.cumsum(S * S, axis=1)
        self.cn = np.zeros((self.n, T + 1), dtype=np.int32)
        self.cn[:, 1:] = np.cumsum(S > 0, axis=1)
        idx = np.where(S > 0, np.arange(T)[None, :], -1).astype(np.int32)
        self.lastnz = np.maximum.accumulate(idx, axis=1)
        del idx

        self.P = build_price_matrix(meta, cal_full, prices, dcols)

        cal = cal_full.set_index("d").loc[dcols]
        self.wday = cal["wday"].to_numpy()
        self.month = cal["month"].to_numpy()
        self.mday = pd.to_datetime(cal["date"]).dt.day.to_numpy()
        self.ev1 = pd.factorize(cal["event_name_1"].fillna("none"))[0]
        self.evt1 = pd.factorize(cal["event_type_1"].fillna("none"))[0]
        states = sorted(meta["state_id"].unique())
        self.snap = np.stack([cal[f"snap_{s}"].to_numpy() for s in states])
        self.state_code = pd.factorize(meta["state_id"], sort=True)[0]
        self.dept_code = pd.factorize(meta["dept_id"], sort=True)[0]
        self.cat_code = pd.factorize(meta["cat_id"], sort=True)[0]
        self.store_code = pd.factorize(meta["store_id"], sort=True)[0]
        self.names = None

    def _win(self, o, w):
        a = max(o - w, 0)
        m = o - a
        s = self.cs[:, o] - self.cs[:, a]
        s2 = self.cs2[:, o] - self.cs2[:, a]
        c = self.cn[:, o] - self.cn[:, a]
        mean = s / m
        var = np.maximum(s2 / m - mean ** 2, 0)
        mean_nz = np.where(c > 0, s / np.maximum(c, 1), 0.0)
        return mean, np.sqrt(var), c / m, mean_nz

    def _gather(self, rows, idx):
        ok = idx >= 0
        v = self.S32[rows, np.clip(idx, 0, self.T - 1)]
        return np.where(ok, v, np.nan)

    def build(self, o, H):
        n, k = H.shape
        rows = np.arange(n)[:, None]
        tc = o + H - 1  # target column index
        cols, names = [], []

        def add(name, arr):
            names.append(name)
            cols.append(np.broadcast_to(np.asarray(arr, dtype=np.float32), (n, k)))

        with np.errstate(all="ignore"):
            # origin-level history features
            m7, _, _, _ = self._win(o, 7)
            m14, _, _, _ = self._win(o, 14)
            m28, sd28, nz28, mnz28 = self._win(o, 28)
            m56, _, _, _ = self._win(o, 56)
            m112, sd112, nz112, _ = self._win(o, 112)
            add("mean7", m7[:, None])
            add("mean14", m14[:, None])
            add("mean28", m28[:, None])
            add("mean56", m56[:, None])
            add("mean112", m112[:, None])
            add("std28", sd28[:, None])
            add("std112", sd112[:, None])
            add("nzfrac28", nz28[:, None])
            add("nzfrac112", nz112[:, None])
            add("meannz28", mnz28[:, None])
            add("ratio7_28", (m7 / (m28 + 0.01))[:, None])
            add("ratio28_112", (m28 / (m112 + 0.01))[:, None])
            dsl = np.where(self.lastnz[:, o - 1] >= 0, o - 1 - self.lastnz[:, o - 1], o)
            add("days_since_sale", dsl[:, None])
            add("last1", self.S32[:, o - 1][:, None])

            # same-weekday lags aligned to the target day
            kmin = -(-H // 7)
            j1 = tc - 7 * kmin
            sw = [self._gather(rows, j1 - 7 * q) for q in range(4)]
            for q, arr in enumerate(sw, start=1):
                add(f"sw_lag{q}", arr)
            stack = np.stack(sw)
            cnt = np.sum(~np.isnan(stack), axis=0)
            add("sw_mean", np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan))
            add("lag364", self._gather(rows, tc - 364))

            # price (known in advance)
            Pt = self.P[rows, tc]
            Po = self.P[:, o - 1][:, None]
            Po28 = self.P[:, max(o - 29, 0)][:, None]
            add("price", Pt)
            add("price_ratio_origin", Pt / Po)
            add("price_chg28", Po / Po28)

            # calendar (known in advance)
            add("h", H)
            add("wday", self.wday[tc])
            add("month", self.month[tc])
            add("mday", self.mday[tc])
            add("snap", self.snap[self.state_code[:, None], tc])
            add("event1", self.ev1[tc])
            add("event_type1", self.evt1[tc])

            # static
            add("dept", self.dept_code[:, None])
            add("cat", self.cat_code[:, None])
            add("store", self.store_code[:, None])
            add("state", self.state_code[:, None])

        X = np.stack(cols, axis=-1).reshape(n * k, len(cols))
        if self.names is None:
            self.names = names
        return X

    def targets(self, o, H):
        rows = np.arange(self.n)[:, None]
        tc = o + H - 1
        assert tc.max() <= self.T - 1, "target would touch the holdout"
        return self.S32[rows, tc].reshape(-1)


# ----------------------------------------------------------------------------
# Synthetic calendar for --selftest
# ----------------------------------------------------------------------------
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
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default="m5_outputs")
    ap.add_argument("--origins", type=int, default=52, help="number of training origins (weekly steps)")
    ap.add_argument("--n_h", type=int, default=6, help="horizons sampled per series per origin")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="smaller, faster first run")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    params = dict(
        objective="tweedie", tweedie_variance_power=1.1, metric="rmse",
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=0.1, max_bin=255, num_threads=args.threads, seed=args.seed, verbose=-1,
    )
    if args.quick:
        args.origins, args.n_h, args.rounds = 12, 4, 400
    if args.selftest:
        args.origins, args.n_h, args.rounds = 8, 4, 40
        params.update(num_leaves=15, min_data_in_leaf=10)
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
    print(f"Data folder: {data_dir}")

    meta, values, dcols, calendar, prices = load_m5(data_dir)
    cal_full = pd.read_csv(find_file(data_dir, "calendar.csv"))
    n, D = values.shape
    T = D - HORIZON
    print(f"Series: {n} | days: {D} | train days: {T}")

    ev = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    fb = FeatureBuilder(meta, values, dcols, cal_full, prices, T)
    rng = np.random.default_rng(args.seed)

    val_o = T - HORIZON
    last_train_o = val_o - HORIZON
    origins = [last_train_o - 7 * k for k in range(args.origins)]
    origins = [o for o in origins if o >= 28]
    print(f"Training origins: {len(origins)} (from {origins[-1]} to {origins[0]}) | validation origin: {val_o}")

    Xs, ys = [], []
    for o in origins:
        H = rng.random((n, HORIZON)).argsort(axis=1)[:, : args.n_h] + 1
        Xs.append(fb.build(o, H))
        ys.append(fb.targets(o, H))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    print(f"Training table: {X.shape[0]:,} rows x {X.shape[1]} features ({X.nbytes / 1e9:.2f} GB)")

    Hv = np.tile(np.arange(1, HORIZON + 1), (n, 1))
    Xv = fb.build(val_o, Hv)
    yv = fb.targets(val_o, Hv)

    names = fb.names
    cats = [c for c in CAT_FEATURES if c in names]
    dtrain = lgb.Dataset(X, label=y, feature_name=names, categorical_feature=cats, free_raw_data=True)
    dval = lgb.Dataset(Xv, label=yv, reference=dtrain)
    print("Training LightGBM ...")
    model = lgb.train(
        params, dtrain, num_boost_round=args.rounds, valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )
    best_it = model.best_iteration
    val_rmse = model.best_score["val"]["rmse"]
    del X, y, dtrain

    Hp = np.tile(np.arange(1, HORIZON + 1), (n, 1))
    Xp = fb.build(T, Hp)
    pred = np.clip(model.predict(Xp, num_iteration=best_it), 0, None).reshape(n, HORIZON)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "pred_E1_lightgbm.npy"), pred.astype(np.float32))

    results = {
        "SeasonalNaive": ev.score(seasonal_naive_forecast(ev.X_train)),
        "E1_LightGBM": ev.score(pred),
    }
    overall = results["E1_LightGBM"][0]

    imp = model.feature_importance("gain")
    order = np.argsort(imp)[::-1][:15]
    imp_lines = [f"  {names[i]:<22}{100 * imp[i] / imp.sum():>6.1f}%" for i in order]

    report = [
        "E1 GLOBAL LIGHTGBM RESULTS",
        f"Data folder: {data_dir}",
        f"Training window: d_1 to d_{T} | Holdout (28 days): d_{T + 1} to d_{D}",
        f"Training origins: {len(origins)} weekly steps | horizons sampled per series per origin: {args.n_h}",
        f"Training rows: {len(origins) * n * args.n_h:,} | features: {len(names)}",
        f"Objective: tweedie (variance power 1.1) | best iteration: {best_it} | validation RMSE: {val_rmse:.4f}",
        "Early stopping validation uses an origin whose targets lie before the holdout (no test leakage).",
        "",
        "WRMSSE BY LEVEL",
        format_table(ev, results),
        "",
        "REFERENCE VALUES (published, window to be confirmed)",
        "  Original M5 winner (hidden test window): 0.520",
        "  Yang et al. Kaggle Top-1 reproduction: 0.5230",
        "  Yang et al. LightGBM + PatchTST: 0.4989",
        f"  This run, E1 LightGBM: {overall:.4f}",
        "",
        "TOP FEATURES BY GAIN",
        *imp_lines,
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min",
    ]
    text = "\n".join(report)
    out_path = os.path.join(args.out_dir, "results_E1_lightgbm.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {os.path.join(args.out_dir, 'pred_E1_lightgbm.npy')}")


if __name__ == "__main__":
    main()
