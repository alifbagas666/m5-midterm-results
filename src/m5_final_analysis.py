"""
E9: Uncertainty and ablation for the blend of the winner (six-set LightGBM average) with direct Chronos-2 forecasts.

Nothing here is tuned. All variants use the same fixed share (0.20) unless the label says otherwise, and the headline
blend is the one fixed in advance (direct Chronos-2 with covariates, mean point, 20 percent at levels 1 to 9).

Part 1, ablation: what each piece adds, scored on every window that has actuals (cv4, public, final).
Part 2, uncertainty: circular block bootstrap over the 28 forecast days (blocks of 7 days, 4 blocks per draw, 2000 draws),
  giving an interval for the change in WRMSSE against the winner alone and for the WRMSSE itself.
  This resamples days only. It does not capture uncertainty from the choice of series or the model, so treat the
  intervals as a rough guide with only four weeks of data per window.
Part 3, secondary outcomes: stability across windows and level-1 bias (total forecast divided by total actual).

Usage:
  python m5_final_analysis.py --data_dir m5_data --in_dir analysis_inputs --out_dir analysis_out
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, LEVELS, WRMSSEEvaluator, load_m5  # noqa: E402

SHARE = 0.20
ORIGIN = {"cv4": 1885, "public": 1913, "final": 1941}
N_DRAWS, BLOCK, N_BLOCKS = 2000, 7, 4
OFFICIAL = 0.5204


def level_preds(ev, winner, kind, extra, s):
    """Aggregated forecasts for all 12 levels. Levels 1 to 9 are mixed with `extra`; levels 10 to 12 stay as the winner."""
    off = np.cumsum([0] + ev.sizes[:9])
    out = []
    for i, A in enumerate(ev.A):
        P = np.asarray(A @ winner)
        if i < 9 and kind is not None and s > 0:
            other = extra[off[i]:off[i + 1]] if kind == "direct" else np.asarray(A @ extra)
            P = (1 - s) * P + s * other
        out.append(P)
    return out


def sq_errors(ev, preds):
    return [(Y - P) ** 2 for Y, P in zip(ev.y_agg, preds)]


def point_score(ev, errs):
    lv = []
    for E, sc, w in zip(errs, ev.scale, ev.weight):
        v = w > 0
        r = np.zeros(len(sc))
        r[v] = np.sqrt(E[v].mean(axis=1) / sc[v])
        lv.append(float(np.sum(w * r)))
    return float(np.mean(lv)), lv


def boot_scores(ev, errs, counts):
    """counts: (B, 28) day counts. Returns (B,) overall WRMSSE per draw."""
    tot = np.zeros(counts.shape[0])
    for E, sc, w in zip(errs, ev.scale, ev.weight):
        v = np.where(w > 0)[0]
        for s in range(0, counts.shape[0], 250):
            M = (counts[s:s + 250] / counts.shape[1]) @ E[v].T  # (b, G)
            tot[s:s + 250] += (np.sqrt(M / sc[v]) * w[v]).sum(axis=1)
    return tot / len(ev.A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--in_dir", default="analysis_inputs")
    ap.add_argument("--out_dir", default="analysis_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(12345)

    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    dcols = list(dcols)
    D = values.shape[1]
    L = lambda n: np.load(os.path.join(args.in_dir, n))  # noqa: E731

    # fixed bootstrap draws, shared by every window and variant
    starts = rng.integers(0, HORIZON, size=(N_DRAWS, N_BLOCKS))
    counts = np.zeros((N_DRAWS, HORIZON))
    for b in range(N_DRAWS):
        for st in starts[b]:
            for j in range(BLOCK):
                counts[b, (st + j) % HORIZON] += 1

    lines = ["ABLATION, UNCERTAINTY AND SECONDARY OUTCOMES FOR THE BLEND (winner + direct Chronos-2 at levels 1 to 9)",
             f"Fixed share {SHARE:.2f} unless stated. WRMSSE over 12 levels, lower is better. Bootstrap: {N_DRAWS} draws, "
             f"circular blocks of {BLOCK} days, {N_BLOCKS} blocks per draw (days only, rough guide).",
             f"Days in data folder: {D}.", ""]
    summary = {}
    for win in ["cv4", "public", "final"]:
        end = ORIGIN[win] + HORIZON
        if D < end:
            lines += [f"WINDOW {win}: skipped, data folder has {D} days and {end} are needed.", ""]
            continue
        ev = WRMSSEEvaluator(meta, values[:, :end], dcols[:end], calendar, prices)
        w = L(f"winner_{win}.npy")
        variants = {
            "winner alone": (None, None, 0.0),
            "HEADLINE direct B mean 20%": ("direct", L(f"agg_chronos2_B_covariates_{win}.npy"), SHARE),
            "direct B mean 10%": ("direct", L(f"agg_chronos2_B_covariates_{win}.npy"), 0.10),
            "direct B mean 30%": ("direct", L(f"agg_chronos2_B_covariates_{win}.npy"), 0.30),
            "direct B median 20%": ("direct", L(f"agg_chronos2_B_covariates_{win}_median.npy"), SHARE),
            "direct A (no covariates) 20%": ("direct", L(f"agg_chronos2_A_univariate_{win}.npy"), SHARE),
            "bottom-up Chronos-2 C 20%": ("bu", L(f"bu_chronosC_{win}.npy"), SHARE),
            "bottom-up PatchTST P1 20%": ("bu", L(f"bu_patchtstP1_{win}.npy"), SHARE),
        }
        res = {}
        for name, (kind, extra, s) in variants.items():
            preds = level_preds(ev, w, kind, extra, s)
            errs = sq_errors(ev, preds)
            pt, lv = point_score(ev, errs)
            bs = boot_scores(ev, errs, counts)
            res[name] = (pt, lv, bs, preds)
        base_pt, _, base_bs, base_preds = res["winner alone"]
        lines.append(f"WINDOW {win} (forecast d_{ORIGIN[win] + 1}..d_{end})")
        lines.append(f"{'Variant':<32}{'WRMSSE':>9}{'Change':>9}{'95% interval of change':>26}{'P(better)':>11}")
        for name, (pt, lv, bs, _) in res.items():
            d = bs - base_bs
            lo, hi = np.percentile(d, [2.5, 97.5])
            pb = float((d < 0).mean()) if name != "winner alone" else float("nan")
            lines.append(f"{name:<32}{pt:>9.4f}{pt - base_pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>26}"
                         f"{'' if name == 'winner alone' else f'{pb:>11.2f}'}")
        pt_h, _, bs_h, preds_h = res["HEADLINE direct B mean 20%"]
        lo_w, hi_w = np.percentile(base_bs, [2.5, 97.5])
        lo_h, hi_h = np.percentile(bs_h, [2.5, 97.5])
        lines.append(f"  Winner alone WRMSSE 95% interval [{lo_w:.4f}, {hi_w:.4f}] | headline blend [{lo_h:.4f}, {hi_h:.4f}]")
        if win == "final":
            lines.append(f"  Share of draws where the headline blend is below the official {OFFICIAL}: {(bs_h < OFFICIAL).mean():.2f} "
                         f"(the official score comes from a different run of the winner's code, so this is a rough reference)")
            lines.append(f"  Share of draws where the winner alone is below the official {OFFICIAL}: {(base_bs < OFFICIAL).mean():.2f}")
        # level-1 bias
        tot_act = float(ev.y_agg[0].sum())
        lines.append(f"  Level 1 bias (total forecast / total actual): winner {base_preds[0].sum() / tot_act:.3f} | headline blend "
                     f"{preds_h[0].sum() / tot_act:.3f}")
        lines.append(f"  {'Level':<24}{'Winner':>9}{'Headline':>10}{'Change':>9}")
        for i, (lv_no, lv_name, _) in enumerate(LEVELS):
            a, b = res["winner alone"][1][i], res["HEADLINE direct B mean 20%"][1][i]
            lines.append(f"  {lv_no:>2} {lv_name:<21}{a:>9.4f}{b:>10.4f}{b - a:>+9.4f}")
        lines.append("")
        summary[win] = (base_pt, pt_h)

    if summary:
        lines.append("STABILITY ACROSS WINDOWS (headline blend minus winner alone, WRMSSE; negative is better)")
        for win, (b0, h) in summary.items():
            lines.append(f"  {win:<8} winner {b0:.4f} | blend {h:.4f} | change {h - b0:+.4f}")
        ch = [h - b0 for b0, h in summary.values()]
        lines.append(f"  Windows improved: {sum(c < 0 for c in ch)} of {len(ch)}. Worst-case change: {max(ch):+.4f}.")
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "results_final_analysis.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
