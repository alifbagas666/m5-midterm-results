#!/usr/bin/env python3
"""
E1b: stronger global LightGBM for M5, with a cumulative ablation ladder.

Builds on m5_lightgbm_e1.py (same origins, horizon feature, validation scheme,
no recursion, no holdout leakage) and adds four changes, one step at a time:

  L0_base      the 34 features of E1
  L1_+price    price history: price vs 365-day max and mean, number of price
               changes, days on sale
  L2_+event    event proximity: days to next and since last event, second event
               slot, days to next and since last SNAP day for the state
  L3_+context  context from the same store and department, the same item across
               stores, and the same store: relative scale, group trend, group
               weekday profile
  L4_+weights  all features, training rows weighted by recent dollar sales

Every step is trained, early-stopped on the validation origin, and scored on the
28-day holdout, so the log shows what each change contributes.

Selection note: the final model is L4, fixed in advance. The holdout scores of
L0 to L3 are reported for ablation only and are not used to choose the final model.

Outputs (in --out_dir)
  pred_E1b_lightgbm.npy       forecasts of the final model L4, shape (30490, 28)
  pred_E1b_L0.npy ... L3.npy  forecasts of the intermediate steps
  results_E1b_lightgbm.txt    plain-text log with ladder and 12-level tables

Usage
  python m5_lightgbm_e1b.py                # full run
  python m5_lightgbm_e1b.py --quick        # smaller, faster first run
  python m5_lightgbm_e1b.py --selftest     # synthetic data, no download

Requires m5_data_and_wrmsse.py and m5_lightgbm_e1.py in the same folder.
"""
import argparse
import gc
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

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
from m5_lightgbm_e1 import CAT_FEATURES, FeatureBuilder, augment_calendar  # noqa: E402

import lightgbm as lgb  # noqa: E402

PRICE_NAMES = ["price_max365_ratio", "price_mean365_ratio", "n_price_changes365", "days_on_sale"]
EVENT_NAMES = ["ev_next", "ev_prev", "ev2_flag", "snap_next", "snap_prev"]
CONTEXT_NAMES = ["ctx_rel_scale", "ctx_group_trend", "ctx_group_dow",
                 "ctx_item_mean28", "ctx_item_ratio7_28", "ctx_store_trend"]


def day_distances(flag, cap=30):
    """Days to the next flagged day and since the last flagged day, capped."""
    D = len(flag)
    ar = np.arange(D)
    nxt = np.where(flag, ar, 10 ** 6)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    prv = np.where(flag, ar, -(10 ** 6))
    prv = np.maximum.accumulate(prv)
    return np.minimum(nxt - ar, cap).astype(np.float32), np.minimum(ar - prv, cap).astype(np.float32)


class FeatureBuilderB(FeatureBuilder):
    def __init__(self, meta, values, dcols, cal_full, prices, T):
        super().__init__(meta, values, dcols, cal_full, prices, T)
        D = values.shape[1]
        cal = cal_full.set_index("d").loc[dcols]

        # event proximity (calendar is known in advance)
        self.ev_next, self.ev_prev = day_distances(cal["event_name_1"].notna().to_numpy())
        if "event_name_2" in cal.columns:
            self.ev2 = cal["event_name_2"].notna().to_numpy().astype(np.float32)
        else:
            self.ev2 = np.zeros(D, dtype=np.float32)
        states = sorted(meta["state_id"].unique())
        nx, pv = [], []
        for s in states:
            a, b = day_distances(cal[f"snap_{s}"].to_numpy() > 0)
            nx.append(a)
            pv.append(b)
        self.snap_next = np.stack(nx)
        self.snap_prev = np.stack(pv)

        # first day with a price (for days on sale)
        valid = ~np.isnan(self.P)
        self.first_price = np.where(valid.any(axis=1), valid.argmax(axis=1), D)
        del valid

        # group indicator matrices
        n = self.n
        self.gid_sd = pd.factorize(meta["store_id"] + "|" + meta["dept_id"], sort=True)[0]
        self.gid_st = self.store_code
        self.gid_it = pd.factorize(meta["item_id"], sort=True)[0]

        def indicator(gid):
            g = int(gid.max()) + 1
            M = sparse.csr_matrix((np.ones(n), (gid, np.arange(n))), shape=(g, n))
            cnt = np.asarray(M.sum(axis=1)).ravel()
            return M, cnt

        self.G_sd, self.cnt_sd = indicator(self.gid_sd)
        self.G_st, self.cnt_st = indicator(self.gid_st)
        self.G_it, self.cnt_it = indicator(self.gid_it)
        self.extra_names = None

    def build(self, o, H):
        X = super().build(o, H)
        n, k = H.shape
        rows = np.arange(n)[:, None]
        tc = o + H - 1
        cols, names = [], []

        def add(name, arr):
            names.append(name)
            cols.append(np.broadcast_to(np.asarray(arr, dtype=np.float32), (n, k)))

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ---- price history (uses prices before the origin plus the target price)
            a = max(o - 365, 0)
            Pw = self.P[:, a:o]
            pmax = np.nanmax(Pw, axis=1)
            pmean = np.nanmean(Pw, axis=1)
            d = np.diff(Pw, axis=1)
            nchg = np.sum(np.isfinite(d) & (d != 0), axis=1)
            Pt = self.P[rows, tc]
            add("price_max365_ratio", Pt / pmax[:, None])
            add("price_mean365_ratio", Pt / pmean[:, None])
            add("n_price_changes365", nchg[:, None])
            add("days_on_sale", np.where(self.first_price <= o - 1, o - self.first_price, -1)[:, None])

            # ---- event proximity
            add("ev_next", self.ev_next[tc])
            add("ev_prev", self.ev_prev[tc])
            add("ev2_flag", self.ev2[tc])
            sc = self.state_code[:, None]
            add("snap_next", self.snap_next[sc, tc])
            add("snap_prev", self.snap_prev[sc, tc])

            # ---- context from groups (origin data only)
            m7 = self._win(o, 7)[0]
            m28 = self._win(o, 28)[0]
            gm28_sd = (self.G_sd @ m28) / self.cnt_sd
            add("ctx_rel_scale", (m28 / (gm28_sd[self.gid_sd] + 0.01))[:, None])
            add("ctx_group_trend", ((self.G_sd @ m7) / (self.G_sd @ m28 + 0.01))[self.gid_sd][:, None])

            a56 = max(o - 56, 0)
            tot = np.asarray(self.G_sd @ self.S32[:, a56:o])
            wd = self.wday[a56:o]
            fac = np.ones((tot.shape[0], 7))
            overall = tot.mean(axis=1) + 1e-6
            for w in range(1, 8):
                m = wd == w
                if m.any():
                    fac[:, w - 1] = tot[:, m].mean(axis=1) / overall
            add("ctx_group_dow", fac[self.gid_sd[:, None], self.wday[tc] - 1])

            add("ctx_item_mean28", (((self.G_it @ m28) / self.cnt_it)[self.gid_it])[:, None])
            add("ctx_item_ratio7_28", (((self.G_it @ m7) / (self.G_it @ m28 + 0.01))[self.gid_it])[:, None])
            add("ctx_store_trend", (((self.G_st @ m7) / (self.G_st @ m28 + 0.01))[self.gid_st])[:, None])

        extra = np.stack(cols, axis=-1).reshape(n * k, len(cols))
        if self.extra_names is None:
            self.extra_names = names
        return np.concatenate([X, extra], axis=1)

    def row_weights(self, o, H):
        n, k = H.shape
        a = max(o - 28, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dollars = np.nansum(self.S32[:, a:o] * self.P[:, a:o], axis=1)
        w = np.clip(dollars / (dollars.mean() + 1e-9), 0.05, 20.0).astype(np.float32)
        return np.repeat(w, k)


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
    fb = FeatureBuilderB(meta, values, dcols, cal_full, prices, T)
    del prices
    gc.collect()
    rng = np.random.default_rng(args.seed)

    val_o = T - HORIZON
    last_train_o = val_o - HORIZON
    origins = [o for o in (last_train_o - 7 * k for k in range(args.origins)) if o >= 28]
    print(f"Training origins: {len(origins)} (from {origins[-1]} to {origins[0]}) | validation origin: {val_o}")

    Xs, ys, ws = [], [], []
    for o in origins:
        H = rng.random((n, HORIZON)).argsort(axis=1)[:, : args.n_h] + 1
        Xs.append(fb.build(o, H))
        ys.append(fb.targets(o, H))
        ws.append(fb.row_weights(o, H))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    del Xs, ys, ws
    gc.collect()
    print(f"Training table: {X.shape[0]:,} rows x {X.shape[1]} features ({X.nbytes / 1e9:.2f} GB)")

    Hv = np.tile(np.arange(1, HORIZON + 1), (n, 1))
    Xv = fb.build(val_o, Hv)
    yv = fb.targets(val_o, Hv)
    wv = fb.row_weights(val_o, Hv)
    Xp = fb.build(T, Hv)

    names = fb.names + fb.extra_names
    idx = {nm: i for i, nm in enumerate(names)}
    base_cols = [idx[nm] for nm in fb.names]
    price_cols = [idx[nm] for nm in PRICE_NAMES]
    event_cols = [idx[nm] for nm in EVENT_NAMES]
    ctx_cols = [idx[nm] for nm in CONTEXT_NAMES]
    steps = [
        ("L0_base", base_cols, False),
        ("L1_+price", base_cols + price_cols, False),
        ("L2_+event", base_cols + price_cols + event_cols, False),
        ("L3_+context", base_cols + price_cols + event_cols + ctx_cols, False),
        ("L4_+weights", base_cols + price_cols + event_cols + ctx_cols, True),
    ]

    results, ladder, preds = {}, [], {}
    final_model, final_names = None, None
    total_actual = ev.y_test.sum()
    for label, cols, weighted in steps:
        t1 = time.time()
        sub_names = [names[i] for i in cols]
        cats = [c for c in CAT_FEATURES if c in sub_names]
        dtr = lgb.Dataset(X[:, cols], label=y, weight=w if weighted else None,
                          feature_name=sub_names, categorical_feature=cats, free_raw_data=True)
        dva = lgb.Dataset(Xv[:, cols], label=yv, weight=wv if weighted else None, reference=dtr)
        print(f"\nTraining {label} ({len(cols)} features) ...")
        model = lgb.train(
            params, dtr, num_boost_round=args.rounds, valid_sets=[dva], valid_names=["val"],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
        )
        pred = np.clip(model.predict(Xp[:, cols], num_iteration=model.best_iteration), 0, None).reshape(n, HORIZON)
        score = ev.score(pred)
        results[label] = score
        preds[label] = pred
        ladder.append((label, len(cols), score[0], model.best_iteration,
                       model.best_score["val"]["rmse"], pred.sum() / total_actual, (time.time() - t1) / 60))
        print(f"{label}: WRMSSE {score[0]:.4f} | best iteration {model.best_iteration} | {ladder[-1][-1]:.1f} min")
        if label == "L4_+weights":
            final_model, final_names = model, sub_names
        del dtr, dva
        gc.collect()

    os.makedirs(args.out_dir, exist_ok=True)
    for i, label in enumerate(["L0_base", "L1_+price", "L2_+event", "L3_+context"]):
        np.save(os.path.join(args.out_dir, f"pred_E1b_L{i}.npy"), preds[label].astype(np.float32))
    np.save(os.path.join(args.out_dir, "pred_E1b_lightgbm.npy"), preds["L4_+weights"].astype(np.float32))

    snaive = ev.score(seasonal_naive_forecast(ev.X_train))
    table_results = {"SeasonalNaive": snaive, "L0_base": results["L0_base"], "L4_final": results["L4_+weights"]}

    ladder_lines = [f"{'Step':<14}{'Feats':>6}{'WRMSSE':>9}{'Change':>9}{'BestIt':>8}{'ValRMSE':>9}{'Bias':>8}{'Min':>6}"]
    prev = None
    for label, nf, sc, bi, vr, bias, mins in ladder:
        ch = "" if prev is None else f"{sc - prev:+.4f}"
        ladder_lines.append(f"{label:<14}{nf:>6}{sc:>9.4f}{ch:>9}{bi:>8}{vr:>9.4f}{bias:>8.3f}{mins:>6.1f}")
        prev = sc

    imp = final_model.feature_importance("gain")
    order = np.argsort(imp)[::-1][:20]
    imp_lines = [f"  {final_names[i]:<22}{100 * imp[i] / imp.sum():>6.1f}%" for i in order]
    final = results["L4_+weights"][0]

    report = [
        "E1b GLOBAL LIGHTGBM RESULTS (ablation ladder)",
        f"Data folder: {data_dir}",
        f"Training window: d_1 to d_{T} | Holdout (28 days): d_{T + 1} to d_{D}",
        f"Training origins: {len(origins)} weekly steps | horizons sampled per series per origin: {args.n_h}",
        f"Training rows: {len(origins) * n * args.n_h:,} | total features available: {len(names)}",
        "Objective: tweedie (variance power 1.1). Early stopping on a validation origin before the holdout.",
        "Final model is L4, fixed in advance. L0 to L3 holdout scores are for ablation only.",
        "Bias = total forecast / total actual over the holdout (1.000 means unbiased overall).",
        "",
        "ABLATION LADDER (each step adds to the previous one)",
        *ladder_lines,
        "",
        "WRMSSE BY LEVEL",
        format_table(ev, table_results),
        "",
        "REFERENCE VALUES (published, window to be confirmed)",
        "  Original M5 winner (hidden test window): 0.520",
        "  Yang et al. Kaggle Top-1 reproduction: 0.5230",
        "  Yang et al. LightGBM + PatchTST: 0.4989",
        "  E1 (previous script, full run): 0.6141",
        f"  This run, E1b final (L4): {final:.4f}",
        "",
        "TOP FEATURES BY GAIN (final model)",
        *imp_lines,
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min",
    ]
    text = "\n".join(report)
    out_path = os.path.join(args.out_dir, "results_E1b_lightgbm.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
