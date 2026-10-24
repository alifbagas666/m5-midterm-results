"""
Scores the foundation-model forecasts, first on their own, then blended into the winner.
Part 1 (STANDALONE): WRMSSE per level 1 to 9 of each model's direct forecast, next to the winner (summed bottom-up forecast) and Chronos-2 B.
  This is a sanity and quality check, not a selection step. Chronos-2 alone was also worse than the winner.
Part 2 (BLEND): the fixed rule (share 0.20, levels 1 to 9, levels 10 to 12 stay as the winner), each model separately, per window with a
  bootstrap interval, then a summary across windows. An average of all foundation models is an extra, clearly labelled.
usage: python m5_fm_score.py --data_dir m5_data --in_dir inputs --fm_dir fm_out --out_dir fm_score_out
  in_dir needs winner_<window>.npy and agg_chronos2_B_covariates_<window>.npy for the windows to be scored.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, LEVELS, WRMSSEEvaluator, load_m5  # noqa: E402
from m5_multiwindow import (ORIGINS, SHARE, boot_scores, make_counts, mix_levels, point_score,  # noqa: E402
                            sign_test_p, sq_errors)

ORDER = ["cv5", "cv2", "cv3", "cv4", "public", "final"]


def direct_levels(ev, winner, direct):
    """12 level forecasts where levels 1 to 9 are the direct forecast and 10 to 12 are the winner (for standalone scoring of 1 to 9)."""
    off = np.cumsum([0] + ev.sizes[:9])
    return [direct[off[i]:off[i + 1]] if i < 9 else np.asarray(A @ winner) for i, A in enumerate(ev.A)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--in_dir", default="inputs")
    ap.add_argument("--fm_dir", default="fm_out")
    ap.add_argument("--out_dir", default="fm_score_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("M5_WINDOW", "final")
    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    dcols = list(dcols)
    D = values.shape[1]
    tags = sorted({re.match(r"fm_(.+)_(cv5|cv2|cv3|cv4|public|final)\.npy$", os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(args.fm_dir, "fm_*.npy")) if re.match(r"fm_(.+)_(cv5|cv2|cv3|cv4|public|final)\.npy$", os.path.basename(p))})
    lines = ["FOUNDATION MODELS: standalone quality, then blend into the winner",
             f"Models found: {', '.join(tags) if tags else 'none'} | Chronos-2 B is the earlier reference. Days in data folder: {D}.",
             f"Blend rule fixed in advance: share {SHARE:.2f}, levels 1 to 9 only, levels 10 to 12 stay as the winner. Lower WRMSSE is better.", ""]
    standalone, blend_rows = [], {}
    all_direct = {}
    for win in ORDER:
        o = ORIGINS[win]
        end = o + HORIZON
        wp = os.path.join(args.in_dir, f"winner_{win}.npy")
        cp = os.path.join(args.in_dir, f"agg_chronos2_B_covariates_{win}.npy")
        if D < end or not os.path.exists(wp) or not os.path.exists(cp):
            standalone.append(f"WINDOW {win}: skipped (needs actuals up to d_{end}, winner and Chronos-2 files).\n")
            continue
        ev = WRMSSEEvaluator(meta, values[:, :end], dcols[:end], calendar, prices)
        w = np.load(wp).astype(np.float64)
        directs = {"Chronos-2 B": np.load(cp).astype(np.float64)}
        for t in tags:
            p = os.path.join(args.fm_dir, f"fm_{t}_{win}.npy")
            if os.path.exists(p):
                directs[t] = np.load(p).astype(np.float64)
        all_direct[win] = directs
        base_levels = [np.asarray(A @ w) for A in ev.A]
        # ---- standalone
        cols = {"winner": point_score(ev, sq_errors(ev, base_levels))[1][:9]}
        for name, d in directs.items():
            cols[name] = point_score(ev, sq_errors(ev, direct_levels(ev, w, d)))[1][:9]
        head = f"{'Level':<24}" + "".join(f"{c[:15]:>16}" for c in cols)
        standalone += [f"WINDOW {win} (forecast d_{o + 1}..d_{end}), standalone WRMSSE per level, direct forecasts of the summed series", head, "-" * len(head)]
        for i in range(9):
            standalone.append(f"{LEVELS[i][0]:>2} {LEVELS[i][1]:<21}" + "".join(f"{v[i]:>16.4f}" for v in cols.values()))
        standalone.append(f"{'mean of L1..L9':<24}" + "".join(f"{np.mean(v):>16.4f}" for v in cols.values()))
        standalone.append("")
        # ---- blend
        counts = make_counts(1000 + o)
        b_pt, _ = point_score(ev, sq_errors(ev, base_levels))
        b_bs = boot_scores(ev, sq_errors(ev, base_levels), counts)
        names = list(directs)
        fm_only = [n for n in names if n != "Chronos-2 B"]
        extra = {}
        if fm_only:
            extra["AVERAGE of Chronos-2 B + all models (extra)"] = np.mean([directs[n] for n in names], axis=0)
        for name, d in list(directs.items()) + list(extra.items()):
            lv = mix_levels(ev, w, d, SHARE)
            er = sq_errors(ev, lv)
            pt, _ = point_score(ev, er)
            bs = boot_scores(ev, er, counts)
            blend_rows.setdefault(name, {})[win] = (pt - b_pt, bs - b_bs, pt, b_pt)
    lines += ["PART 1: STANDALONE (direct forecasts of levels 1 to 9, scored as in the earlier Chronos-2 table)", ""] + standalone
    lines += ["PART 2: BLEND INTO THE WINNER (change in 12-level WRMSSE vs the winner alone, negative is better)", ""]
    for name, per in blend_rows.items():
        lines.append(f"{name}")
        lines.append(f"  {'Window':<8}{'Winner':>9}{'Blend':>9}{'Change':>9}{'95% interval of change':>26}{'P(better)':>11}")
        for win in ORDER:
            if win in per:
                ch, dd, pt, b0 = per[win]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                lines.append(f"  {win:<8}{b0:>9.4f}{pt:>9.4f}{ch:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>26}{float((dd < 0).mean()):>11.2f}")
        chs = np.array([per[w_][0] for w_ in ORDER if w_ in per])
        draws = np.mean([per[w_][1] for w_ in ORDER if w_ in per], axis=0)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        k = int((chs < 0).sum())
        lines.append(f"  Across {len(chs)} windows: mean change {chs.mean():+.4f} | improved {k} of {len(chs)} | sign test p = {sign_test_p(k, len(chs)):.3f} | "
                     f"95% interval of the mean change [{lo:+.4f}, {hi:+.4f}] | worst window {chs.max():+.4f}")
        lines.append("")
    lines.append("Notes: models are reported separately and none is picked by its result. The AVERAGE row is an extra, not the main claim. Windows share training history, "
                 "so the intervals (days resampled in blocks of 7) are a rough guide.")
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "results_fm.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
