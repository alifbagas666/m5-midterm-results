"""
E8: Winner's six-set LightGBM average + direct Chronos-2 forecasts of the summed series (levels 1 to 9).

Rule fixed BEFORE the final window is scored (written here so it cannot be changed after seeing the result):
  * Base: the winner's equal-weight six-set average, unchanged at levels 10 to 12.
  * At levels 1 to 9 the forecast of each group is (1 - s) * winner + s * direct Chronos-2 with covariates (variant B).
  * Headline share s = 0.20. It is chosen on the cv4 window, whose sales growth (0.6 percent) is the closest to the
    final window (0.2 percent). The public window (4.1 percent growth) is reported but not used to choose.
  * All other shares are reported as sensitivity only. The final window is scored once.

Windows (28 days each): final d_1942..d_1969, public d_1914..d_1941, cv4 d_1886..d_1913.
Scoring uses the same WRMSSE code that reproduced the seasonal-naive baseline of the official results.

Usage:
  python m5_blend_final.py --data_dir m5_data --in_dir blend_inputs --out_dir blend_out
Expected files in --in_dir:
  winner_final.npy winner_public.npy winner_cv4.npy                      (30490 x 28 each)
  agg_chronos2_B_covariates_final.npy / _public.npy / _cv4.npy           (154 x 28 each)
Windows that need actuals which are not in the data folder are skipped with a message.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, LEVELS, WRMSSEEvaluator, load_m5  # noqa: E402

HEADLINE = 0.20
SHARES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
ORIGIN = {"final": 1941, "public": 1913, "cv4": 1885}


def blend_levels(ev, winner, direct, s):
    sizes = ev.sizes[:9]
    off = np.cumsum([0] + sizes)
    out = []
    for i, (A, sc, wt, Y) in enumerate(zip(ev.A, ev.scale, ev.weight, ev.y_agg)):
        P = np.asarray(A @ winner)
        if i < 9 and s > 0:
            P = (1 - s) * P + s * direct[off[i]:off[i + 1]]
        v = wt > 0
        r = np.zeros(len(sc))
        r[v] = np.sqrt(np.mean((Y[v] - P[v]) ** 2, axis=1) / sc[v])
        out.append(float(np.sum(wt * r)))
    return float(np.mean(out)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--in_dir", default="blend_inputs")
    ap.add_argument("--out_dir", default="blend_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    dcols = list(dcols)
    D = values.shape[1]
    lines = ["BLEND OF THE WINNER (SIX-SET AVERAGE) WITH DIRECT CHRONOS-2 FORECASTS AT LEVELS 1 TO 9",
             f"Days available in data folder: {D}. Headline share fixed in advance: {HEADLINE:.2f} (chosen on cv4).",
             "WRMSSE over all 12 levels, lower is better. Share 0.00 is the winner alone.", ""]
    res = {}
    for win in ["cv4", "public", "final"]:
        end = ORIGIN[win] + HORIZON
        if D < end:
            lines.append(f"Window {win}: skipped, the data folder has {D} days and {end} are needed for the actuals.\n")
            continue
        w = np.load(os.path.join(args.in_dir, f"winner_{win}.npy"))
        d = np.load(os.path.join(args.in_dir, f"agg_chronos2_B_covariates_{win}.npy"))
        ev = WRMSSEEvaluator(meta, values[:, :end], dcols[:end], calendar, prices)
        bias = float(w.sum() / ev.y_test.sum())
        lines.append(f"Window {win} (forecast d_{ORIGIN[win] + 1}..d_{end}), winner bias {bias:.3f}")
        lines.append(f"{'Share':>8}{'WRMSSE':>10}{'Change':>10}")
        base = None
        for s in SHARES:
            m, lv = blend_levels(ev, w, d, s)
            if s == 0:
                base = m
            mark = "  <- headline" if abs(s - HEADLINE) < 1e-9 else ""
            lines.append(f"{s:>8.2f}{m:>10.4f}{m - base:>+10.4f}{mark}")
            res[(win, s)] = (m, lv)
        lines.append("")
        lines.append(f"{'Level':<24}{'Winner alone':>14}{'Headline blend':>16}{'Change':>10}")
        for i, (lv, name, _) in enumerate(LEVELS):
            a, b = res[(win, 0.0)][1][i], res[(win, HEADLINE)][1][i]
            lines.append(f"{lv:>2} {name:<21}{a:>14.4f}{b:>16.4f}{b - a:>+10.4f}")
        lines.append("")
    if ("final", 0.0) in res:
        m0, mh = res[("final", 0.0)][0], res[("final", HEADLINE)][0]
        lines += ["FINAL WINDOW VERDICT (headline share, scored once)",
                  f"  Winner alone (our reproduction): {m0:.4f} | blend: {mh:.4f} | change {mh - m0:+.4f}",
                  f"  Official M5 winner: 0.5204 -> blend {'beats' if mh < 0.5204 else 'does not beat'} it",
                  f"  Our reproduction 0.5228 -> blend {'beats' if mh < m0 else 'does not beat'} it",
                  "  Note: the blend changes levels 1 to 9 only, so the levels are no longer exactly coherent."]
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "results_blend_final.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
