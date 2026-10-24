#!/usr/bin/env python3
"""
M5 data preparation + official-style 12-level WRMSSE evaluator (Phase 1 and 2).

What this single script does
  1. Downloads M5 via kagglehub (or reads a local folder with --data_dir).
  2. Defines ONE fixed 28-day holdout: the last 28 observed days of
     sales_train_evaluation.csv (d_1914..d_1941 on the real data).
  3. Builds the 12 hierarchical aggregation levels (42,840 series).
  4. Computes WRMSSE exactly as in the M5 protocol:
       - RMSSE scale = mean squared first difference of the training series,
         starting from its first non-zero observation
       - weights = dollar sales over the last 28 training days, normalised
         within each level
       - overall WRMSSE = mean of the 12 level scores
  5. Runs Naive and Seasonal Naive as pipeline sanity checks.
  6. Writes a plain-text results file (AI-readable) to --out_dir.

Usage
  python m5_data_and_wrmsse.py                      # downloads M5 with kagglehub
  python m5_data_and_wrmsse.py --data_dir /path/to/m5
  python m5_data_and_wrmsse.py --selftest           # synthetic data, no download

Later phases only need: ev = WRMSSEEvaluator(...); ev.score(pred) where pred has
shape (30490, 28) in the same row order as sales_train_evaluation.csv.
"""
import argparse
import os
import tempfile
import time

import numpy as np
import pandas as pd
from scipy import sparse

HORIZON = 28
WEIGHT_WINDOW = 28
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

LEVELS = [
    (1, "Total", []),
    (2, "State", ["state_id"]),
    (3, "Store", ["store_id"]),
    (4, "Category", ["cat_id"]),
    (5, "Department", ["dept_id"]),
    (6, "State x Category", ["state_id", "cat_id"]),
    (7, "State x Department", ["state_id", "dept_id"]),
    (8, "Store x Category", ["store_id", "cat_id"]),
    (9, "Store x Department", ["store_id", "dept_id"]),
    (10, "Item", ["item_id"]),
    (11, "Item x State", ["item_id", "state_id"]),
    (12, "Item x Store", ["item_id", "store_id"]),
]
EXPECTED_SIZES = [1, 3, 10, 3, 7, 9, 21, 30, 70, 3049, 9147, 30490]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def find_file(root, name):
    for dirpath, _, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"{name} not found under {root}")


def load_calendar(data_dir):
    """Full calendar with a 'd' column (rebuilt as d_1, d_2, ... when the full organiser release has none)."""
    calendar = pd.read_csv(find_file(data_dir, "calendar.csv"))
    if "d" not in calendar.columns:
        calendar["d"] = ["d_%d" % (i + 1) for i in range(len(calendar))]
    return calendar


def load_m5(data_dir, with_test=None):
    """Load M5 sales, calendar and prices (Kaggle files or the full organiser release).

    Window: if sales_test_evaluation.csv (actual sales d_1942..d_1969) is found, its 28 days are appended, so the
    last 28 days (the holdout) are the competition's FINAL window d_1942..d_1969 and the 28 days before that
    (d_1914..d_1941) are the validation window. Without that file, or with environment variable M5_WINDOW=public
    (or with_test=False), the holdout is d_1914..d_1941 and the validation window is d_1886..d_1913.
    """
    sales = pd.read_csv(find_file(data_dir, "sales_train_evaluation.csv"))
    calendar = load_calendar(data_dir)[["d", "wm_yr_wk"]]
    prices = pd.read_csv(find_file(data_dir, "sell_prices.csv"))
    # Full organiser release: no 'id' in the sales file. Rebuild it in the Kaggle layout.
    if "id" not in sales.columns:
        sales.insert(0, "id", sales["item_id"] + "_" + sales["store_id"] + "_evaluation")
    dcols = sorted([c for c in sales.columns if c.startswith("d_")], key=lambda c: int(c[2:]))
    meta = sales[["id"] + ID_COLS].copy()
    values = sales[dcols].to_numpy(dtype=np.float64)
    if with_test is None:
        with_test = os.environ.get("M5_WINDOW", "final").lower() != "public"
    if with_test:
        try:
            tpath = find_file(data_dir, "sales_test_evaluation.csv")
        except FileNotFoundError:
            tpath = None
        if tpath is not None:
            tdf = pd.read_csv(tpath)
            tcols = sorted([c for c in tdf.columns if c.startswith("d_")], key=lambda c: int(c[2:]))
            if len(tcols) == HORIZON and int(tcols[0][2:]) == len(dcols) + 1:
                tdf = tdf.set_index(tdf["item_id"] + "|" + tdf["store_id"])
                tv = tdf.reindex(meta["item_id"] + "|" + meta["store_id"])[tcols].to_numpy(dtype=np.float64)
                if not np.isnan(tv).any():
                    values = np.concatenate([values, tv], axis=1)
                    dcols = list(dcols) + tcols
    return meta, values, dcols, calendar, prices


# ----------------------------------------------------------------------------
# Hierarchy
# ----------------------------------------------------------------------------
def build_aggregators(meta):
    """One sparse (n_groups x n_bottom) summing matrix per level."""
    n = len(meta)
    mats = []
    for _, _, keys in LEVELS:
        if keys:
            gid = meta.groupby(keys, sort=True).ngroup().to_numpy()
        else:
            gid = np.zeros(n, dtype=np.int64)
        g = int(gid.max()) + 1
        A = sparse.csr_matrix((np.ones(n), (gid, np.arange(n))), shape=(g, n))
        mats.append(A)
    return mats


# ----------------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------------
def rmsse_scale(block):
    """Mean squared first difference starting at the first non-zero value."""
    m, T = block.shape
    nz = block > 0
    first = np.where(nz.any(axis=1), nz.argmax(axis=1), T)
    d2 = np.diff(block, axis=1) ** 2
    idx = np.arange(1, T)[None, :]
    mask = idx > first[:, None]
    cnt = mask.sum(axis=1)
    total = (d2 * mask).sum(axis=1)
    return np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)


class WRMSSEEvaluator:
    def __init__(self, meta, values, dcols, calendar, prices, horizon=HORIZON):
        self.meta = meta
        self.horizon = horizon
        self.train_cols = dcols[:-horizon]
        self.test_cols = dcols[-horizon:]
        self.X_train = values[:, : -horizon]
        self.y_test = values[:, -horizon:]
        self.A = build_aggregators(meta)
        self.sizes = [A.shape[0] for A in self.A]

        dollars = self.X_train[:, -WEIGHT_WINDOW:] * self._price_matrix(
            calendar, prices, self.train_cols[-WEIGHT_WINDOW:]
        )

        self.scale, self.weight, self.y_agg = [], [], []
        for A in self.A:
            G = A.shape[0]
            sc = np.empty(G)
            for s in range(0, G, 4000):
                blk = A[s : s + 4000] @ self.X_train
                sc[s : s + 4000] = rmsse_scale(blk)
            dol = np.asarray(A @ dollars).sum(axis=1)
            w = dol / dol.sum()
            w = np.where(np.isfinite(sc) & (sc > 0), w, 0.0)
            self.scale.append(sc)
            self.weight.append(w)
            self.y_agg.append(np.asarray(A @ self.y_test))

    def _price_matrix(self, calendar, prices, cols):
        wk_of_day = calendar.set_index("d")["wm_yr_wk"]
        weeks = sorted(set(wk_of_day.loc[cols]))
        keys = pd.MultiIndex.from_frame(self.meta[["item_id", "store_id"]])
        per_week = {}
        for w in weeks:
            p = prices[prices["wm_yr_wk"] == w].set_index(["item_id", "store_id"])["sell_price"]
            per_week[w] = p.reindex(keys).to_numpy(dtype=np.float64)
        out = np.zeros((len(self.meta), len(cols)))
        for j, c in enumerate(cols):
            out[:, j] = np.nan_to_num(per_week[wk_of_day.loc[c]], nan=0.0)
        return out

    def score(self, pred):
        """pred: (n_bottom_series, horizon). Returns (overall, list of level scores)."""
        pred = np.asarray(pred, dtype=np.float64)
        assert pred.shape == self.y_test.shape, f"pred shape {pred.shape} != {self.y_test.shape}"
        out = []
        for A, sc, w, Y in zip(self.A, self.scale, self.weight, self.y_agg):
            P = np.asarray(A @ pred)
            valid = w > 0
            rmsse = np.zeros(len(sc))
            rmsse[valid] = np.sqrt(np.mean((Y[valid] - P[valid]) ** 2, axis=1) / sc[valid])
            out.append(float(np.sum(w * rmsse)))
        return float(np.mean(out)), out


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def naive_forecast(X_train, h=HORIZON):
    return np.repeat(X_train[:, -1:], h, axis=1)


def seasonal_naive_forecast(X_train, h=HORIZON, period=7):
    last = X_train[:, -period:]
    reps = int(np.ceil(h / period))
    return np.tile(last, (1, reps))[:, :h]


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def format_table(ev, results):
    names = list(results.keys())
    lines = []
    head = f"{'Level':<6}{'Name':<22}{'Series':>8}" + "".join(f"{n:>16}" for n in names)
    lines.append(head)
    lines.append("-" * len(head))
    for i, (lv, name, _) in enumerate(LEVELS):
        row = f"{lv:<6}{name:<22}{ev.sizes[i]:>8}"
        row += "".join(f"{results[n][1][i]:>16.4f}" for n in names)
        lines.append(row)
    lines.append("-" * len(head))
    lines.append(f"{'':<6}{'OVERALL WRMSSE':<22}{sum(ev.sizes):>8}" + "".join(f"{results[n][0]:>16.4f}" for n in names))
    return "\n".join(lines)


def run_checks(ev, real_data):
    notes = []
    for i, (lv, name, _) in enumerate(LEVELS):
        s = ev.weight[i].sum()
        if abs(s - 1.0) > 1e-6:
            notes.append(f"WARNING level {lv} weights sum to {s:.6f}, expected 1")
    perfect, _ = ev.score(ev.y_test)
    if perfect > 1e-9:
        notes.append(f"WARNING perfect forecast scored {perfect:.3e}, expected 0")
    if real_data:
        if ev.sizes != EXPECTED_SIZES:
            notes.append(f"WARNING level sizes {ev.sizes} differ from expected {EXPECTED_SIZES}")
        if sum(ev.sizes) != 42840:
            notes.append(f"WARNING total series {sum(ev.sizes)} != 42840")
    if not notes:
        notes.append("All sanity checks passed (weights sum to 1 per level, perfect forecast = 0"
                     + (", 42,840 series across 12 levels)." if real_data else ")."))
    return notes


# ----------------------------------------------------------------------------
# Synthetic data for --selftest
# ----------------------------------------------------------------------------
def make_synthetic(folder, n_days=140, seed=0):
    rng = np.random.default_rng(seed)
    depts = {"FOODS_1": "FOODS", "FOODS_2": "FOODS", "FOODS_3": "FOODS",
             "HOBBIES_1": "HOBBIES", "HOBBIES_2": "HOBBIES",
             "HOUSEHOLD_1": "HOUSEHOLD", "HOUSEHOLD_2": "HOUSEHOLD"}
    stores = {"CA_1": "CA", "CA_2": "CA", "CA_3": "CA", "CA_4": "CA",
              "TX_1": "TX", "TX_2": "TX", "TX_3": "TX",
              "WI_1": "WI", "WI_2": "WI", "WI_3": "WI"}
    rows = []
    for dept, cat in depts.items():
        for k in range(1, 4):
            item = f"{dept}_{k:03d}"
            for store, state in stores.items():
                base = rng.uniform(0.2, 6)
                dow = 1 + 0.4 * np.sin(np.arange(n_days) * 2 * np.pi / 7 + rng.uniform(0, 6))
                y = rng.poisson(base * dow)
                y[: rng.integers(0, 30)] = 0
                rows.append([f"{item}_{store}_evaluation", item, dept, cat, store, state] + list(y))
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + [f"d_{i}" for i in range(1, n_days + 1)]
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(folder, "sales_train_evaluation.csv"), index=False)
    cal = pd.DataFrame({"d": [f"d_{i}" for i in range(1, n_days + 1)],
                        "wm_yr_wk": [11101 + (i // 7) for i in range(n_days)]})
    cal.to_csv(os.path.join(folder, "calendar.csv"), index=False)
    pr = []
    for r in rows:
        for wk in sorted(cal["wm_yr_wk"].unique()):
            pr.append([r[4], r[1], wk, round(float(rng.uniform(1, 10)), 2)])
    pd.DataFrame(pr, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        os.path.join(folder, "sell_prices.csv"), index=False)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="Folder containing the M5 csv files")
    ap.add_argument("--out_dir", default="m5_outputs")
    ap.add_argument("--selftest", action="store_true", help="Use small synthetic M5-format data")
    args = ap.parse_args()

    t0 = time.time()
    tmp = None
    if args.selftest:
        tmp = tempfile.mkdtemp()
        make_synthetic(tmp)
        data_dir = tmp
    elif args.data_dir:
        data_dir = args.data_dir
    else:
        import kagglehub
        data_dir = kagglehub.competition_download("m5-forecasting-accuracy")
    print(f"Data folder: {data_dir}")

    meta, values, dcols, calendar, prices = load_m5(data_dir)
    print(f"Loaded {values.shape[0]} bottom series x {values.shape[1]} days")

    ev = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    print(f"Train: {ev.train_cols[0]}..{ev.train_cols[-1]} | Holdout: {ev.test_cols[0]}..{ev.test_cols[-1]}")

    notes = run_checks(ev, real_data=not args.selftest)

    results = {
        "Naive": ev.score(naive_forecast(ev.X_train)),
        "SeasonalNaive": ev.score(seasonal_naive_forecast(ev.X_train)),
    }
    table = format_table(ev, results)

    report = [
        "M5 DATA AND WRMSSE PIPELINE RESULTS (Phase 1 and 2)",
        f"Data folder: {data_dir}",
        f"Bottom series: {values.shape[0]} | Days: {values.shape[1]}",
        f"Training window: {ev.train_cols[0]} to {ev.train_cols[-1]}",
        f"Holdout window (28 days): {ev.test_cols[0]} to {ev.test_cols[-1]}",
        f"Weight window: last {WEIGHT_WINDOW} training days, dollar sales (units x price)",
        "Overall WRMSSE = mean of the 12 level scores",
        "",
        "SANITY CHECKS",
        *notes,
        "",
        "WRMSSE BY LEVEL",
        table,
        "",
        f"Runtime: {time.time() - t0:.1f} s",
    ]
    text = "\n".join(report)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "results_phase1_baselines.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
