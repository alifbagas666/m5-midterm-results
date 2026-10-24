#!/usr/bin/env python3
"""
E4, E5, E6: Chronos-2 on M5 as one ablation ladder (zero-shot, no training on M5).

  A_univariate     E4  sales history only
  B_covariates     E5  + price, SNAP, weekday, event flag (all known in advance)
  C_crosslearn_*   E6  + cross-learning inside groups of related series

Each configuration is run twice with the same settings:
  holdout     context ends at d_1913, forecasts d_1914..d_1941 (the 28-day test window)
  validation  context ends 28 days earlier, forecasts d_1886..d_1913

The validation window gives a clean basis for choosing models and for fitting
ensemble weights later, so nothing is ever chosen on the holdout.

Point forecast: Chronos-2 returns quantiles. WRMSSE is a squared-error metric, so
the primary point forecast is the mean implied by the quantiles (integral of the
quantile function). The median forecast is also scored and saved for transparency.
This choice is fixed in advance.

Data handling: days before an item's first price (not yet on sale) are set to
missing in the context, and missing prices are back-filled with the first price.
Scoring always uses the raw sales.

Outputs (in --out_dir)
  pred_chronos2_<cfg>.npy          holdout forecast, mean point, shape (30490, 28)
  pred_chronos2_<cfg>_median.npy   holdout forecast, median point
  predval_chronos2_<cfg>.npy       validation-window forecast, mean point
  results_chronos2.txt             plain-text log: ladder, 12-level tables, timings

Usage
  python m5_chronos2.py --quick            # 512 series, checks the setup and prints a time estimate
  python m5_chronos2.py                    # full run
  python m5_chronos2.py --groupings store_dept,item,store_cat    # try several cross-learning groupings
  python m5_chronos2.py --selftest         # synthetic data and a mock model, no GPU needed

Requires m5_data_and_wrmsse.py in the same folder. A full run needs chronos-forecasting and a CUDA PyTorch.
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
    load_calendar,
    make_synthetic,
    seasonal_naive_forecast,
)

COVARIATES = ["price", "snap", "wday", "event"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x)


def quantile_mean_weights(levels):
    """Weights w such that sum_j w_j * Q(p_j) approximates the mean (integral of Q over [0, 1])."""
    p = np.concatenate([[0.0], np.asarray(levels, dtype=np.float64), [1.0]])
    dp = np.diff(p)
    w = np.zeros(len(p))
    w[:-1] += dp / 2
    w[1:] += dp / 2
    out = w[1:-1].copy()
    out[0] += w[0]
    out[-1] += w[-1]
    return out


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


class MockPipeline:
    """Stands in for Chronos2Pipeline in --selftest: seasonal-naive quantiles, same interface."""

    quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    def predict(self, inputs, prediction_length=28, batch_size=256, cross_learning=False, **kw):
        out = []
        for item in inputs:
            tgt = item["target"] if isinstance(item, dict) else item
            tgt = np.nan_to_num(np.asarray(tgt, dtype=np.float64))
            base = np.tile(tgt[-7:], 5)[:prediction_length]
            q = np.stack([base * f for f in np.linspace(0.6, 1.6, len(self.quantiles))])
            out.append(q[None, :, :].astype(np.float32))
        return out


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class M5Inputs:
    def __init__(self, meta, values, dcols, cal_full, prices):
        self.n, self.D = values.shape
        cal = cal_full.set_index("d").loc[dcols]
        self.meta = meta

        # price matrix (n x D), NaN before an item is on sale
        wk = cal["wm_yr_wk"].to_numpy()
        weeks = np.unique(wk)
        widx = np.searchsorted(weeks, wk)
        p = prices[prices["wm_yr_wk"].isin(weeks)]
        key = pd.MultiIndex.from_frame(meta[["item_id", "store_id"]])
        r = key.get_indexer(pd.MultiIndex.from_frame(p[["item_id", "store_id"]]))
        c = np.searchsorted(weeks, p["wm_yr_wk"].to_numpy())
        ok = r >= 0
        Pw = np.full((self.n, len(weeks)), np.nan, dtype=np.float32)
        Pw[r[ok], c[ok]] = p["sell_price"].to_numpy(dtype=np.float32)[ok]
        P = Pw[:, widx]
        valid = ~np.isnan(P)
        first = np.where(valid.any(axis=1), valid.argmax(axis=1), 0)
        fill = P[np.arange(self.n), first]
        fill = np.where(np.isnan(fill), 0.0, fill).astype(np.float32)
        self.P = np.where(valid, P, fill[:, None]).astype(np.float32)

        # target with pre-launch days masked as missing
        Y = values.astype(np.float32)
        Y[~valid] = np.nan
        self.Y = Y

        # shared calendar covariates
        states = sorted(meta["state_id"].unique())
        self.state_code = pd.factorize(meta["state_id"], sort=True)[0]
        self.snap = [cal[f"snap_{s}"].to_numpy().astype(np.float32) for s in states]
        wd = cal["wday"].to_numpy().astype(int)
        self.wday = np.array([f"w{x}" for x in wd])
        self.event = cal["event_name_1"].notna().to_numpy().astype(np.float32)

        # groupings for cross-learning
        m = meta
        self.groupings = {
            "store_dept": pd.factorize(m["store_id"] + "|" + m["dept_id"], sort=True)[0],
            "store_cat": pd.factorize(m["store_id"] + "|" + m["cat_id"], sort=True)[0],
            "store": pd.factorize(m["store_id"], sort=True)[0],
            "item": pd.factorize(m["item_id"], sort=True)[0],
            "cat_state": pd.factorize(m["cat_id"] + "|" + m["state_id"], sort=True)[0],
        }

    def build(self, i, origin, ctx_len, with_cov):
        a = max(origin - ctx_len, 0)
        tgt = self.Y[i, a:origin]
        if not with_cov:
            return tgt
        sn = self.snap[self.state_code[i]]
        past = {"price": self.P[i, a:origin], "snap": sn[a:origin],
                "wday": self.wday[a:origin], "event": self.event[a:origin]}
        fut = {"price": self.P[i, origin:origin + HORIZON], "snap": sn[origin:origin + HORIZON],
               "wday": self.wday[origin:origin + HORIZON], "event": self.event[origin:origin + HORIZON]}
        return {"target": tgt, "past_covariates": past, "future_covariates": fut}


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def run_config(pipe, data, idx_series, origin, ctx_len, with_cov, group_ids, batch_size, cl_tasks, weights, quantiles, log):
    """Returns (mean_forecast, median_forecast), each (len(idx_series), HORIZON)."""
    n = len(idx_series)
    q_levels = np.asarray(quantiles)
    med_idx = int(np.argmin(np.abs(q_levels - 0.5)))
    mean_out = np.zeros((n, HORIZON), dtype=np.float32)
    med_out = np.zeros((n, HORIZON), dtype=np.float32)

    def store(pos, preds):
        arr = np.stack([to_np(t)[0] for t in preds])  # (m, n_q, H)
        mean_out[pos] = np.tensordot(arr, weights, axes=([1], [0]))
        med_out[pos] = arr[:, med_idx, :]

    if group_ids is None:
        block = 2048
        for s in range(0, n, block):
            pos = np.arange(s, min(s + block, n))
            inputs = [data.build(idx_series[j], origin, ctx_len, with_cov) for j in pos]
            store(pos, pipe.predict(inputs, prediction_length=HORIZON, batch_size=batch_size, cross_learning=False))
            if (s // block) % 5 == 0:
                log(f"    {min(s + block, n)}/{n} series")
    else:
        nv = 1 + len(COVARIATES) if with_cov else 1
        chunk = max(4, cl_tasks // nv)
        g = group_ids[idx_series]
        order = np.argsort(g, kind="stable")
        chunks = []
        for gid in np.unique(g):
            members = order[g[order] == gid]
            for s in range(0, len(members), chunk):
                chunks.append(members[s:s + chunk])
        for ci, pos in enumerate(chunks):
            inputs = [data.build(idx_series[j], origin, ctx_len, with_cov) for j in pos]
            preds = pipe.predict(inputs, prediction_length=HORIZON,
                                 batch_size=len(pos) * nv + 8, cross_learning=True)
            store(pos, preds)
            if ci % 200 == 0:
                log(f"    chunk {ci + 1}/{len(chunks)}")
    return np.clip(mean_out, 0, None), np.clip(med_out, 0, None)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default="m5_outputs")
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--context_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=256, help="tasks per batch when cross-learning is off")
    ap.add_argument("--cl_tasks", type=int, default=100, help="approx. tasks per cross-learning batch")
    ap.add_argument("--groupings", default="store_dept", help="comma list: store_dept,store_cat,store,item,cat_state")
    ap.add_argument("--no_val", action="store_true", help="skip validation-window forecasts")
    ap.add_argument("--quick", action="store_true", help="512 series, no WRMSSE, prints a time estimate")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    logs = []

    def log(msg):
        print(msg, flush=True)

    if args.selftest:
        tmp = tempfile.mkdtemp()
        make_synthetic(tmp, n_days=240)
        augment_calendar(tmp, 240)
        data_dir = tmp
        args.context_length = 120
    elif args.data_dir:
        data_dir = args.data_dir
    elif os.path.isdir("m5_data"):
        data_dir = "m5_data"
    else:
        import kagglehub
        data_dir = kagglehub.competition_download("m5-forecasting-accuracy")
    log(f"Data folder: {data_dir}")

    meta, values, dcols, calendar, prices = load_m5(data_dir)
    cal_full = load_calendar(data_dir)
    n, D = values.shape
    T = D - HORIZON
    log(f"Series: {n} | days: {D} | holdout origin: {T} | validation origin: {T - HORIZON}")

    data = M5Inputs(meta, values, dcols, cal_full, prices)

    if args.selftest:
        pipe = MockPipeline()
        device_note = "mock model (selftest)"
    else:
        import torch
        from chronos import Chronos2Pipeline
        use_cuda = torch.cuda.is_available()
        device_note = f"cuda: {torch.cuda.get_device_name(0)}" if use_cuda else "CPU (no CUDA found, this will be slow)"
        log(f"Loading {args.model} on {device_note}")
        pipe = Chronos2Pipeline.from_pretrained(args.model, device_map="cuda" if use_cuda else "cpu")
    weights = quantile_mean_weights(pipe.quantiles)

    groupings = [g.strip() for g in args.groupings.split(",") if g.strip()]
    for g in groupings:
        if g not in data.groupings:
            raise SystemExit(f"Unknown grouping {g}. Choose from {list(data.groupings)}")
    configs = [("A_univariate", False, None), ("B_covariates", True, None)]
    configs += [(f"C_crosslearn_{g}", True, g) for g in groupings]

    if args.quick:
        sub = np.random.default_rng(0).choice(n, size=min(512, n), replace=False)
        sub.sort()
        log(f"QUICK MODE: {len(sub)} series, timing only")
        for name, with_cov, grp in configs:
            t1 = time.time()
            gid = data.groupings[grp] if grp else None
            mean_f, med_f = run_config(pipe, data, sub, T, args.context_length, with_cov, gid,
                                       args.batch_size, args.cl_tasks, weights, pipe.quantiles, log)
            dt = time.time() - t1
            est = dt / len(sub) * n / 60
            bad = int(np.isnan(mean_f).sum())
            log(f"{name}: {dt:.1f} s for {len(sub)} series | NaN count {bad} | "
                f"estimated {est:.1f} min per window for all {n} series "
                f"({'x2 with validation' if not args.no_val else 'holdout only'})")
        return

    ev_hold = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    ev_val = None
    if not args.no_val:
        ev_val = WRMSSEEvaluator(meta, values[:, :T], dcols[:T], calendar, prices)

    os.makedirs(args.out_dir, exist_ok=True)
    allidx = np.arange(n)
    results, ladder = {}, []
    for name, with_cov, grp in configs:
        t1 = time.time()
        gid = data.groupings[grp] if grp else None
        log(f"\n{name}: holdout window")
        mean_f, med_f = run_config(pipe, data, allidx, T, args.context_length, with_cov, gid,
                                   args.batch_size, args.cl_tasks, weights, pipe.quantiles, log)
        s_mean, s_med = ev_hold.score(mean_f), ev_hold.score(med_f)
        np.save(os.path.join(args.out_dir, f"pred_chronos2_{name}.npy"), mean_f)
        np.save(os.path.join(args.out_dir, f"pred_chronos2_{name}_median.npy"), med_f)
        val_score = None
        if ev_val is not None:
            log(f"{name}: validation window")
            vmean, _ = run_config(pipe, data, allidx, T - HORIZON, args.context_length, with_cov, gid,
                                  args.batch_size, args.cl_tasks, weights, pipe.quantiles, log)
            val_score = ev_val.score(vmean)
            np.save(os.path.join(args.out_dir, f"predval_chronos2_{name}.npy"), vmean)
        mins = (time.time() - t1) / 60
        results[name] = s_mean
        ladder.append((name, s_mean[0], s_med[0], None if val_score is None else val_score[0], mins))
        log(f"{name}: holdout WRMSSE (mean point) {s_mean[0]:.4f} | median point {s_med[0]:.4f} | {mins:.1f} min")

    snaive = ev_hold.score(seasonal_naive_forecast(ev_hold.X_train))
    table_results = {"SeasonalNaive": snaive}
    for name, _, _ in configs:
        table_results[name.replace("crosslearn", "cl")] = results[name]

    lines = [f"{'Config':<28}{'Holdout mean':>14}{'Holdout median':>16}{'Validation mean':>17}{'Min':>7}"]
    for name, hm, hmed, v, mins in ladder:
        vs = "n/a" if v is None else f"{v:.4f}"
        lines.append(f"{name:<28}{hm:>14.4f}{hmed:>16.4f}{vs:>17}{mins:>7.1f}")

    report = [
        "CHRONOS-2 ON M5 (zero-shot ablation ladder)",
        f"Data folder: {data_dir}",
        f"Device: {device_note} | model: {args.model}",
        f"Context length: {args.context_length} days | cross-learning tasks per batch: about {args.cl_tasks}",
        f"Training window: d_1 to d_{T} | Holdout (28 days): d_{T + 1} to d_{D}",
        "Point forecast: mean implied by the quantiles (primary, fixed in advance); median also reported.",
        "No training on M5 was done. Whether M5 or similar retail data was in the Chronos-2 pretraining corpus",
        "is not checked here and should be addressed in the limitations section.",
        "",
        "LADDER (WRMSSE, lower is better; validation window is d_%d to d_%d)" % (T - HORIZON + 1, T),
        *lines,
        "",
        "WRMSSE BY LEVEL (holdout, mean point)",
        format_table(ev_hold, table_results),
        "",
        "REFERENCE VALUES (published, window to be confirmed)",
        "  Original M5 winner: 0.520 | Yang et al. Top-1 reproduction: 0.5230 | Yang et al. LightGBM + PatchTST: 0.4989",
        "  Our E1b L3 (LightGBM, unweighted, all features): 0.5422",
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min",
    ]
    text = "\n".join(report)
    out_path = os.path.join(args.out_dir, "results_chronos2.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
