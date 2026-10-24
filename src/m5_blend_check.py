#!/usr/bin/env python3
"""
Blend check: how much does mixing saved forecasts help, with weights chosen on the validation window only?

For 2 to 4 saved forecasts (pred_<name>.npy for the holdout, predval_<name>.npy for the validation window), the
script
  1. scores every model alone on both windows,
  2. searches a grid of non-negative weights summing to 1 and picks the weights with the lowest VALIDATION WRMSSE,
  3. reports the holdout WRMSSE of those weights, of equal weights, and of each variant after a single global
     calibration factor (validation actual total / validation forecast total),
  4. prints the 12-level holdout table for the single models and the chosen blend.
Nothing is selected on the holdout. Training is not involved; it only reads files and takes a few minutes at most.

Caveat: LightGBM validation forecasts are mildly optimistic (their early stopping or round count saw that window),
so validation-fitted weights can lean slightly toward LightGBM.

Usage
  python m5_blend_check.py --names E1c_lightgbm_cal,chronos2_B_covariates
  python m5_blend_check.py --names E1c_lightgbm_cal,E1d_lightgbm_cal,chronos2_B_covariates --step 0.1
Output: printed report and blend_check.txt in --out_dir.
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, WRMSSEEvaluator, format_table, load_m5  # noqa: E402


def simplex_grid(k, step):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k - 1):
        if sum(c) <= n:
            yield np.array(list(c) + [n - sum(c)], dtype=float) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--out_dir", default="m5_outputs")
    ap.add_argument("--names", required=True, help="2 to 4 forecast names, comma separated")
    ap.add_argument("--step", type=float, default=0.1, help="weight grid step")
    args = ap.parse_args()

    names = [s.strip() for s in args.names.split(",") if s.strip()]
    if not 2 <= len(names) <= 4:
        raise SystemExit("Give 2 to 4 names")
    H, V = [], []
    for nm in names:
        ph = os.path.join(args.out_dir, f"pred_{nm}.npy")
        pv = os.path.join(args.out_dir, f"predval_{nm}.npy")
        for p in (ph, pv):
            if not os.path.exists(p):
                raise SystemExit(f"Missing file: {p}")
        H.append(np.load(ph).astype(np.float64))
        V.append(np.load(pv).astype(np.float64))

    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    n, D = values.shape
    T = D - HORIZON
    ev_hold = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    ev_val = WRMSSEEvaluator(meta, values[:, :T], dcols[:T], calendar, prices)

    def calib(pv):
        return float(ev_val.y_test.sum() / max(pv.sum(), 1e-9))

    lines = [f"{'Strategy':<46}{'Weights':<26}{'Validation':>11}{'Holdout':>9}{'HoldBias':>9}"]

    def add(label, w, plain=True):
        pv = sum(wi * v for wi, v in zip(w, V))
        ph = sum(wi * h for wi, h in zip(w, H))
        f = 1.0 if plain else calib(pv)
        sv, sh = ev_val.score(pv * f), ev_hold.score(ph * f)
        wt = " ".join(f"{x:.2f}" for x in w)
        lines.append(f"{label:<46}{wt:<26}{sv[0]:>11.4f}{sh[0]:>9.4f}{(ph * f).sum() / ev_hold.y_test.sum():>9.3f}")
        return sv[0], sh[0], sh

    for i, nm in enumerate(names):
        w = np.eye(len(names))[i]
        add(f"{nm} (alone)", w, True)
        add(f"{nm} (alone, calibrated)", w, False)

    eq = np.ones(len(names)) / len(names)
    add("equal weights", eq, True)
    add("equal weights, calibrated", eq, False)

    best = {True: (9e9, None), False: (9e9, None)}
    for w in simplex_grid(len(names), args.step):
        pv = sum(wi * v for wi, v in zip(w, V))
        for plain in (True, False):
            f = 1.0 if plain else calib(pv)
            s = ev_val.score(pv * f)[0]
            if s < best[plain][0]:
                best[plain] = (s, w)
    out = {}
    for plain, label in ((True, "best on validation"), (False, "best on validation, calibrated")):
        out[plain] = add(label, best[plain][1], plain)

    # level table for the calibrated validation-best blend
    wb = best[False][1]
    pvb = sum(wi * v for wi, v in zip(wb, V))
    phb = sum(wi * h for wi, h in zip(wb, H)) * calib(pvb)
    table = {nm[:14]: ev_hold.score(h) for nm, h in zip(names, H)}
    table["blend_cal"] = ev_hold.score(phb)

    text = "\n".join([
        "BLEND CHECK (weights chosen on the validation window only)",
        f"Validation window: d_{T - HORIZON + 1} to d_{T} | Holdout window: d_{T + 1} to d_{D} | grid step {args.step}",
        "Models: " + ", ".join(names),
        "Calibrated = one global multiplier fitted on validation (actual total / forecast total).",
        "Lower WRMSSE is better. HoldBias = total forecast / total actual on the holdout.",
        "",
        *lines,
        "",
        "WRMSSE BY LEVEL (holdout; single models, uncalibrated, and the validation-best calibrated blend)",
        format_table(ev_hold, table),
        "",
        "Reference: Yang et al. Kaggle Top-1 reproduction 0.5230 | LightGBM + PatchTST 0.4989 (window unconfirmed)",
        "Our E1b global LightGBM L3 on this holdout: 0.5422 (no validation forecasts saved).",
    ])
    print(text)
    with open(os.path.join(args.out_dir, "blend_check.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {os.path.join(args.out_dir, 'blend_check.txt')}")


if __name__ == "__main__":
    main()
