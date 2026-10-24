"""
E10: Does the blend gain hold across many windows? One script, two stages.

Rule fixed in advance (not tuned here): winner's six-set LightGBM average at all levels; at levels 1 to 9 mix 20 percent of
the direct Chronos-2 forecast with covariates (variant B, mean point). Levels 10 to 12 stay as the winner.
Secondary, labelled as such: the same blend followed by OLS reconciliation (all 12 levels made to add up).

Windows (28 forecast days each, train on the days before):
  cv5 d_1802..d_1829 | cv2 d_1830..d_1857 | cv3 d_1858..d_1885 | cv4 d_1886..d_1913 | public d_1914..d_1941 | final d_1942..d_1969

Stage "forecast": Chronos-2 zero-shot direct forecasts of the summed series (levels 1 to 9, 154 series) for every window
  that has no forecast file yet. About 2 minutes per window on a GPU. Existing files are never overwritten.
Stage "score": for every window that has winner_<win>.npy (30490 x 28) and the direct forecast, scores
  winner alone / blend / blend + OLS reconciliation, bootstrap interval per window (days resampled in blocks of 7),
  and a summary across windows (mean change, windows improved, sign test, bootstrap interval of the mean change).

Usage:
  python m5_multiwindow.py --data_dir m5_data --in_dir inputs --out_dir multiwindow_out [--stage all|forecast|score]
Files in --in_dir: winner_<win>.npy and agg_chronos2_B_covariates_<win>.npy (the latter is created by stage forecast).
Output: multiwindow_out/results_multiwindow.txt (plain text) and the new agg_chronos2_B_covariates_<win>.npy files.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, LEVELS, WRMSSEEvaluator, load_calendar, load_m5  # noqa: E402

ORIGINS = {"cv5": 1801, "cv2": 1829, "cv3": 1857, "cv4": 1885, "public": 1913, "final": 1941}
SHARE = 0.20
SENS = [0.10, 0.30]
N_DRAWS, BLOCK, N_BLOCKS = 2000, 7, 4


# ---------------------------------------------------------------- forecasts
def stage_forecast(args, meta, values, dcols, calendar, prices, cal_full):
    from m5_aggregate import build_inputs, forecast, level_slices  # noqa: F401
    from m5_chronos2 import MockPipeline, quantile_mean_weights
    todo = [w for w in ORIGINS if not os.path.exists(os.path.join(args.in_dir, f"agg_chronos2_B_covariates_{w}.npy"))
            and not os.path.exists(os.path.join(args.out_dir, f"agg_chronos2_B_covariates_{w}.npy"))
            and values.shape[1] >= ORIGINS[w] + HORIZON]
    if not todo:
        print("Forecast stage: nothing to do, all direct forecast files exist.", flush=True)
        return
    if args.mock:
        pipe = MockPipeline()
    else:
        import torch
        from chronos import Chronos2Pipeline
        cuda = torch.cuda.is_available()
        print("Loading", args.model, "cuda" if cuda else "CPU", flush=True)
        pipe = Chronos2Pipeline.from_pretrained(args.model, device_map="cuda" if cuda else "cpu")
    weights = quantile_mean_weights(pipe.quantiles)
    med_idx = int(np.argmin(np.abs(np.asarray(pipe.quantiles) - 0.5)))
    for w in todo:
        o = ORIGINS[w]
        t0 = time.time()
        ev = WRMSSEEvaluator(meta, values[:, : o + HORIZON], dcols[: o + HORIZON], calendar, prices)
        inputs = build_inputs(ev, meta, values, cal_full, dcols[: o + HORIZON], o, args.context_length, True)
        mean, _ = forecast(pipe, inputs, weights, med_idx)
        np.save(os.path.join(args.out_dir, f"agg_chronos2_B_covariates_{w}.npy"), mean)
        print(f"  forecast {w} (origin d_{o}) done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------- scoring helpers
def mix_levels(ev, winner, direct, s):
    off = np.cumsum([0] + ev.sizes[:9])
    out = []
    for i, A in enumerate(ev.A):
        P = np.asarray(A @ winner)
        if i < 9 and s > 0:
            P = (1 - s) * P + s * direct[off[i]:off[i + 1]]
        out.append(P)
    return out


def ols_reconcile(ev, levels, init):
    """Least squares: bottom b minimising sum over all levels of (y_base - A b)^2. Matrix-free conjugate gradient."""
    S = sp.vstack([sp.csr_matrix(A) for A in ev.A]).tocsr()
    ST = S.T.tocsr()
    Y = np.vstack(levels)
    M = LinearOperator((S.shape[1], S.shape[1]), matvec=lambda x: ST @ (S @ x), dtype=np.float64)
    rhs = ST @ Y
    b = np.zeros_like(rhs)
    worst = 0.0
    for j in range(rhs.shape[1]):
        x, _ = cg(M, rhs[:, j], x0=init[:, j], rtol=1e-9, maxiter=1000)
        b[:, j] = x
        worst = max(worst, float(np.linalg.norm(M.matvec(x) - rhs[:, j]) / (np.linalg.norm(rhs[:, j]) + 1e-12)))
    b = np.clip(b, 0, None)
    return [np.asarray(A @ b) for A in ev.A], worst


def sq_errors(ev, levels):
    return [(Y - P) ** 2 for Y, P in zip(ev.y_agg, levels)]


def point_score(ev, errs):
    lv = []
    for E, sc, w in zip(errs, ev.scale, ev.weight):
        v = w > 0
        r = np.zeros(len(sc))
        r[v] = np.sqrt(E[v].mean(axis=1) / sc[v])
        lv.append(float(np.sum(w * r)))
    return float(np.mean(lv)), lv


def boot_scores(ev, errs, counts):
    tot = np.zeros(counts.shape[0])
    for E, sc, w in zip(errs, ev.scale, ev.weight):
        v = np.where(w > 0)[0]
        for s in range(0, counts.shape[0], 250):
            M = (counts[s:s + 250] / counts.shape[1]) @ E[v].T
            tot[s:s + 250] += (np.sqrt(M / sc[v]) * w[v]).sum(axis=1)
    return tot / len(ev.A)


def make_counts(seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, HORIZON, size=(N_DRAWS, N_BLOCKS))
    counts = np.zeros((N_DRAWS, HORIZON))
    for b in range(N_DRAWS):
        for st in starts[b]:
            for j in range(BLOCK):
                counts[b, (st + j) % HORIZON] += 1
    return counts


def sign_test_p(k, n):
    """Two-sided exact sign test p-value, k successes out of n."""
    if n == 0:
        return float("nan")
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def stage_score(args, meta, values, dcols, calendar, prices):
    D = values.shape[1]
    lines = ["MULTI-WINDOW CHECK: winner (six-set LightGBM average) vs blend with direct Chronos-2 at levels 1 to 9",
             f"Fixed rule: share {SHARE:.2f}, direct Chronos-2 with covariates, mean point, levels 1 to 9 only. Reconciled = same blend, then OLS reconciliation (secondary).",
             f"WRMSSE over 12 levels, lower is better. Bootstrap per window: {N_DRAWS} draws, blocks of {BLOCK} days (days only, rough guide). Days in data folder: {D}.", ""]
    rows = []
    for k, (win, o) in enumerate(ORIGINS.items()):
        end = o + HORIZON
        wpath = os.path.join(args.in_dir, f"winner_{win}.npy")
        dpath = next((p for p in [os.path.join(args.in_dir, f"agg_chronos2_B_covariates_{win}.npy"),
                                  os.path.join(args.out_dir, f"agg_chronos2_B_covariates_{win}.npy")] if os.path.exists(p)), None)
        if D < end or not os.path.exists(wpath) or dpath is None:
            why = "data folder too short" if D < end else ("winner file missing" if not os.path.exists(wpath) else "direct forecast missing")
            lines += [f"WINDOW {win} (d_{o + 1}..d_{end}): skipped ({why}).", ""]
            continue
        ev = WRMSSEEvaluator(meta, values[:, :end], dcols[:end], calendar, prices)
        w = np.load(wpath).astype(np.float64)
        d = np.load(dpath).astype(np.float64)
        growth = float(values[:, o:end].sum() / values[:, o - HORIZON:o].sum() - 1)
        bias = float(w.sum() / ev.y_test.sum())
        counts = make_counts(1000 + o)
        base_levels = [np.asarray(A @ w) for A in ev.A]
        blend_levels = mix_levels(ev, w, d, SHARE)
        rec_levels, resid = ols_reconcile(ev, blend_levels, w)
        res = {}
        for name, lv in [("winner alone", base_levels), ("blend (headline)", blend_levels), ("blend + OLS reconciled", rec_levels)] + \
                        [(f"blend share {s:.2f}", mix_levels(ev, w, d, s)) for s in SENS]:
            errs = sq_errors(ev, lv)
            pt, per = point_score(ev, errs)
            res[name] = (pt, per, boot_scores(ev, errs, counts))
        b_pt, b_per, b_bs = res["winner alone"]
        lines.append(f"WINDOW {win} (forecast d_{o + 1}..d_{end}) | sales growth vs previous 28 days {growth * 100:+.1f}% | winner bias {bias:.3f}")
        lines.append(f"{'Variant':<26}{'WRMSSE':>9}{'Change':>9}{'95% interval of change':>26}{'P(better)':>11}")
        for name, (pt, per, bs) in res.items():
            dd = bs - b_bs
            lo, hi = np.percentile(dd, [2.5, 97.5])
            extra = "" if name == "winner alone" else f"{float((dd < 0).mean()):>11.2f}"
            lines.append(f"{name:<26}{pt:>9.4f}{pt - b_pt:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>26}{extra}")
        lines.append(f"  OLS solver relative residual (should be tiny): {resid:.1e}")
        lines.append(f"  {'Level':<24}{'Winner':>9}{'Blend':>9}{'Reconciled':>12}")
        for i, (lv_no, lv_name, _) in enumerate(LEVELS):
            lines.append(f"  {lv_no:>2} {lv_name:<21}{b_per[i]:>9.4f}{res['blend (headline)'][1][i]:>9.4f}{res['blend + OLS reconciled'][1][i]:>12.4f}")
        if win == "final":
            bh = res["blend (headline)"][2]
            lines.append(f"  Official M5 winner 0.5204: blend below it in {(bh < 0.5204).mean():.2f} of draws, winner alone in {(b_bs < 0.5204).mean():.2f}")
        lines.append("")
        rows.append((win, growth, b_pt, res["blend (headline)"][0], res["blend + OLS reconciled"][0],
                     res["blend (headline)"][2] - b_bs, res["blend + OLS reconciled"][2] - b_bs))

    if rows:
        n = len(rows)
        lines.append(f"SUMMARY ACROSS {n} WINDOWS (change = variant minus winner alone, negative is better)")
        lines.append(f"{'Window':<9}{'Growth':>8}{'Winner':>9}{'Blend':>9}{'Change':>9}{'Reconciled':>12}{'Change':>9}")
        for win, g, b0, h, r, _, _ in rows:
            lines.append(f"{win:<9}{g * 100:>7.1f}%{b0:>9.4f}{h:>9.4f}{h - b0:>+9.4f}{r:>12.4f}{r - b0:>+9.4f}")
        ch_h = np.array([h - b0 for _, _, b0, h, _, _, _ in rows])
        ch_r = np.array([r - b0 for _, _, b0, _, r, _, _ in rows])
        for label, ch, bidx in [("blend (headline)", ch_h, 5), ("blend + OLS reconciled", ch_r, 6)]:
            mean_draws = np.mean([r_[bidx] for r_ in rows], axis=0)  # windows resampled independently (different seeds)
            lo, hi = np.percentile(mean_draws, [2.5, 97.5])
            kk = int((ch < 0).sum())
            lines.append(f"  {label}: mean change {ch.mean():+.4f} | windows improved {kk} of {n} | sign test p = {sign_test_p(kk, n):.3f} | "
                         f"95% interval of the mean change [{lo:+.4f}, {hi:+.4f}] | worst window {ch.max():+.4f}")
        gs = np.array([g for _, g, _, _, _, _, _ in rows])
        if n >= 4 and gs.std() > 0:
            lines.append(f"  Correlation between window sales growth and blend change: {np.corrcoef(gs, ch_h)[0, 1]:+.2f} (descriptive only)")
        lines.append("  Notes: windows do not overlap but share training history, so they are not fully independent. The interval resamples days only.")
        lines.append("  The rule was fixed before the final window was scored; reconciliation is a secondary, labelled variant.")
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "results_multiwindow.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--in_dir", default="inputs")
    ap.add_argument("--out_dir", default="multiwindow_out")
    ap.add_argument("--stage", default="all", choices=["all", "forecast", "score"])
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--context_length", type=int, default=1024)
    ap.add_argument("--mock", action="store_true", help="test the pipeline with a fake forecaster (no real forecasts)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("M5_WINDOW", "final")
    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    dcols = list(dcols)
    cal_full = load_calendar(args.data_dir)
    print(f"Series {values.shape[0]}, days {values.shape[1]}", flush=True)
    if args.stage in ("all", "forecast"):
        stage_forecast(args, meta, values, dcols, calendar, prices, cal_full)
    if args.stage in ("all", "score"):
        stage_score(args, meta, values, dcols, calendar, prices)


if __name__ == "__main__":
    main()
