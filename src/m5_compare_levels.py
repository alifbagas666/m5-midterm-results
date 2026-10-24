#!/usr/bin/env python3
"""
Level-by-level comparison of our saved forecasts with the published Yang et al. rows.

Purpose: Yang et al. do not state which 28-day window they used. Our models produce forecasts for two windows:
  holdout     d_1914 to d_1941
  validation  d_1886 to d_1913 (forecasts saved as predval_*.npy)
This script scores each saved forecast on both windows and prints all 12 WRMSSE levels next to the published
rows, so the window whose level pattern is closer to the paper can be judged. It trains nothing and only reads
files. It does not select any model; it only helps decide which window the published numbers are comparable to.

Note: validation forecasts are mildly optimistic for LightGBM (early stopping used that window), so a closer
match on validation is suggestive, not proof.

Usage
  python m5_compare_levels.py --names E1c_lightgbm,E1c_lightgbm_cal,E1c_store
  python m5_compare_levels.py --names E1c_lightgbm,chronos2_B_covariates --out_dir m5_outputs
Looks for pred_<name>.npy and predval_<name>.npy in --out_dir. A name with no predval file is scored on the holdout only.

Output: printed table and compare_levels.txt in --out_dir.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import (  # noqa: E402
    HORIZON,
    WRMSSEEvaluator,
    load_m5,
    seasonal_naive_forecast,
)

# Published rows (Yang et al., Table 1 and Table 2): Avg, then Levels 1 to 12
YANG = {
    "Yang Kaggle-Top1": [0.5230, 0.2006, 0.3123, 0.3992, 0.2843, 0.3725, 0.3953, 0.4792, 0.4815, 0.5742, 0.9649, 0.9283, 0.8838],
    "Yang AE LGB+PatchTST": [0.4989, 0.1455, 0.2907, 0.3830, 0.2244, 0.3392, 0.3688, 0.4631, 0.4625, 0.5623, 0.9528, 0.9188, 0.8758],
}


def row(label, avg, levels):
    return f"{label:<30}{avg:>7.4f}" + "".join(f"{v:>7.3f}" for v in levels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--out_dir", default="m5_outputs")
    ap.add_argument("--names", required=True, help="comma list of forecast names, files pred_<name>.npy / predval_<name>.npy")
    args = ap.parse_args()

    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    n, D = values.shape
    T = D - HORIZON
    ev_hold = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    ev_val = WRMSSEEvaluator(meta, values[:, :T], dcols[:T], calendar, prices)

    head = f"{'Model / window':<30}{'Avg':>7}" + "".join(f"{'L' + str(i):>7}" for i in range(1, 13))
    lines = [head, "-" * len(head)]
    for label, v in YANG.items():
        lines.append(row(label + " (paper)", v[0], v[1:]))
    lines.append("-" * len(head))

    sn_v = ev_val.score(seasonal_naive_forecast(ev_val.X_train))
    sn_h = ev_hold.score(seasonal_naive_forecast(ev_hold.X_train))
    lines.append(row("SeasonalNaive | validation", sn_v[0], sn_v[1]))
    lines.append(row("SeasonalNaive | holdout", sn_h[0], sn_h[1]))
    lines.append("-" * len(head))

    for nm in [s.strip() for s in args.names.split(",") if s.strip()]:
        ph = os.path.join(args.out_dir, f"pred_{nm}.npy")
        pv = os.path.join(args.out_dir, f"predval_{nm}.npy")
        if os.path.exists(pv):
            s = ev_val.score(np.load(pv))
            lines.append(row(f"{nm} | validation", s[0], s[1]))
        else:
            lines.append(f"{nm} | validation: no predval file")
        if os.path.exists(ph):
            s = ev_hold.score(np.load(ph))
            lines.append(row(f"{nm} | holdout", s[0], s[1]))
        else:
            lines.append(f"{nm} | holdout: no pred file")
        lines.append("")

    text = "\n".join([
        "LEVEL COMPARISON WITH PUBLISHED ROWS",
        "Validation window: d_%d to d_%d | Holdout window: d_%d to d_%d" % (T - HORIZON + 1, T, T + 1, D),
        "Lower is better. Level 1 = total, 9 = store x department, 10 = item, 11 = item x state, 12 = item x store.",
        "",
        *lines,
    ])
    print(text)
    with open(os.path.join(args.out_dir, "compare_levels.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {os.path.join(args.out_dir, 'compare_levels.txt')}")


if __name__ == "__main__":
    main()
