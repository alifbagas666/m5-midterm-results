"""
E11 (exploratory): add direct zero-shot Chronos-2 forecasts of the item-level aggregates, level 10 (item, 3049 series)
and level 11 (item x state, 9147 series), to the blend that already uses levels 1 to 9.

Rules fixed in advance, nothing is tuned here:
  - Same model and settings as the levels 1 to 9 run: amazon/chronos-2, covariates (weekday, event, SNAP share of the group),
    context 1024, horizon 28, mean implied by the quantiles.
  - Same share 0.20 of the direct forecast at each level it is applied to. The winner's bottom forecasts are not changed,
    so level 12 is untouched.
  - Variants: winner alone | HEADLINE (levels 1 to 9, the result already reported) | HEADLINE + level 10 | HEADLINE + levels 10 and 11.
  - The decision uses the five windows before the final one. The final window is scored once and labelled exploratory.

Stage forecast: writes lvl10_chronos2_B_<win>.npy (3049 x 28) and lvl11_chronos2_B_<win>.npy (9147 x 28) per window.
  Existing files are never overwritten, so the stage can be restarted.
Stage score: needs winner_<win>.npy and agg_chronos2_B_covariates_<win>.npy in --in_dir.

Usage:
  python m5_lvl1011.py --data_dir <M5 data, 1969 days> --in_dir inputs --out_dir lvl1011_out --stage all
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, LEVELS, WRMSSEEvaluator, load_calendar, load_m5  # noqa: E402
from m5_multiwindow import (N_DRAWS, boot_scores, make_counts, point_score, sign_test_p,  # noqa: E402
                            sq_errors)

ORDER = ["cv5", "cv2", "cv3", "cv4", "public", "final"]
ORIGINS = {"cv5": 1801, "cv2": 1829, "cv3": 1857, "cv4": 1885, "public": 1913, "final": 1941}
SHARE = 0.20
L10, L11 = 9, 10  # zero-based level indices
CHUNK = 1500


def build_inputs_levels(ev, meta, values, cal_full, dcols, origin, ctx_len, level_idx):
    cal = cal_full.set_index("d").loc[dcols]
    states = sorted(meta["state_id"].unique())
    state_code = np.searchsorted(states, meta["state_id"].to_numpy())
    onehot = np.zeros((len(meta), len(states)))
    onehot[np.arange(len(meta)), state_code] = 1.0
    snap = np.stack([cal[f"snap_{s}"].to_numpy().astype(np.float32) for s in states])
    wd = np.array([f"w{x}" for x in cal["wday"].to_numpy().astype(int)])
    event = cal["event_name_1"].notna().to_numpy().astype(np.float32)
    a = max(origin - ctx_len, 0)
    A = ev.A[level_idx]
    Y = np.asarray(A @ values[:, :origin], dtype=np.float32)
    size = np.asarray(A.sum(axis=1)).ravel()
    share = np.asarray(A @ onehot) / size[:, None]
    snap_g = (share @ snap).astype(np.float32)
    inputs = []
    for g in range(A.shape[0]):
        inputs.append({
            "target": Y[g, a:origin],
            "past_covariates": {"snap": snap_g[g, a:origin], "wday": wd[a:origin], "event": event[a:origin]},
            "future_covariates": {"snap": snap_g[g, origin:origin + HORIZON], "wday": wd[origin:origin + HORIZON],
                                  "event": event[origin:origin + HORIZON]},
        })
    return inputs


def stage_forecast(args, meta, values, dcols, calendar, prices, cal_full):
    from m5_aggregate import forecast
    from m5_chronos2 import MockPipeline, quantile_mean_weights
    todo = [w for w in ORDER if values.shape[1] >= ORIGINS[w] + HORIZON and not all(
        os.path.exists(os.path.join(args.out_dir, f"lvl{n}_chronos2_B_{w}.npy")) for n in (10, 11))]
    if not todo:
        print("Forecast stage: nothing to do.", flush=True)
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
        ev = WRMSSEEvaluator(meta, values[:, : o + HORIZON], dcols[: o + HORIZON], calendar, prices)
        for name, li in (("10", L10), ("11", L11)):
            path = os.path.join(args.out_dir, f"lvl{name}_chronos2_B_{w}.npy")
            if os.path.exists(path):
                continue
            t0 = time.time()
            inputs = build_inputs_levels(ev, meta, values, cal_full, dcols[: o + HORIZON], o, args.context_length, li)
            parts = []
            for s in range(0, len(inputs), CHUNK):
                mean, _ = forecast(pipe, inputs[s:s + CHUNK], weights, med_idx)
                parts.append(mean)
                print(f"    {w} level {name}: {min(s + CHUNK, len(inputs))}/{len(inputs)} series, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            np.save(path, np.concatenate(parts, axis=0))
            print(f"  forecast {w} level {name} (origin d_{o}) done in {(time.time() - t0) / 60:.1f} min", flush=True)


def level_list(ev, winner, direct19, d10, d11, use):
    """use: set of extra levels (0-based) mixed at SHARE. Levels 1 to 9 always use direct19 when 'h' in use."""
    off = np.cumsum([0] + ev.sizes[:9])
    out = []
    for i, A in enumerate(ev.A):
        P = np.asarray(A @ winner)
        if i < 9 and "h" in use:
            P = (1 - SHARE) * P + SHARE * direct19[off[i]:off[i + 1]]
        elif i == L10 and "10" in use:
            P = (1 - SHARE) * P + SHARE * d10
        elif i == L11 and "11" in use:
            P = (1 - SHARE) * P + SHARE * d11
        out.append(P)
    return out


def stage_score(args, meta, values, dcols, calendar, prices):
    counts = make_counts(2024)
    variants = [("winner alone", set()), ("HEADLINE levels 1-9", {"h"}),
                ("HEADLINE + level 10", {"h", "10"}), ("HEADLINE + levels 10, 11", {"h", "10", "11"})]
    lines = ["EXPLORATORY: direct Chronos-2 at the item-level aggregates (level 10 item, level 11 item x state)",
             f"Fixed rule: share {SHARE:.2f} of the direct forecast, same model and settings as levels 1 to 9. Level 12 stays as the winner.",
             f"WRMSSE over 12 levels, lower is better. Bootstrap: {N_DRAWS} draws, blocks of 7 days (days only, rough guide).",
             "Decision uses the five windows before the final one; the final window is shown once and is exploratory.", ""]
    summ = {}
    for win in ORDER:
        o = ORIGINS[win]
        paths = [os.path.join(args.in_dir, f"winner_{win}.npy"),
                 os.path.join(args.in_dir, f"agg_chronos2_B_covariates_{win}.npy"),
                 os.path.join(args.out_dir, f"lvl10_chronos2_B_{win}.npy"),
                 os.path.join(args.out_dir, f"lvl11_chronos2_B_{win}.npy")]
        if values.shape[1] < o + HORIZON or not all(os.path.exists(p) for p in paths):
            lines += [f"WINDOW {win}: skipped (missing data or files).", ""]
            continue
        w, d19, d10, d11 = [np.load(p) for p in paths]
        ev = WRMSSEEvaluator(meta, values[:, : o + HORIZON], dcols[: o + HORIZON], calendar, prices)
        res = {}
        for name, use in variants:
            lv = level_list(ev, w, d19, d10, d11, use)
            errs = sq_errors(ev, lv)
            pt, per = point_score(ev, errs)
            res[name] = (pt, per, boot_scores(ev, errs, counts))
        lines.append(f"WINDOW {win} (forecast d_{o + 1}..d_{o + HORIZON})" + ("   [exploratory, scored once]" if win == "final" else ""))
        lines.append(f"{'Variant':<28}{'WRMSSE':>9}{'vs winner':>11}{'vs HEADLINE':>13}{'95% interval vs HEADLINE':>28}{'P(better)':>11}")
        base, head = res["winner alone"], res["HEADLINE levels 1-9"]
        for name, _ in variants:
            pt, _, bs = res[name]
            d = bs - head[2]
            lo, hi = np.percentile(d, [2.5, 97.5])
            extra = "" if name in ("winner alone", "HEADLINE levels 1-9") else f"{f'[{lo:+.4f}, {hi:+.4f}]':>28}{(d < 0).mean():>11.2f}"
            lines.append(f"{name:<28}{pt:>9.4f}{pt - base[0]:>+11.4f}{pt - head[0]:>+13.4f}{extra}")
        only = [np.asarray(A @ w) for A in ev.A]
        only[L10], only[L11] = d10, d11
        dl = point_score(ev, sq_errors(ev, only))[1]
        lines.append(f"  Levels 10 / 11 (WRMSSE): winner {base[1][L10]:.4f} / {base[1][L11]:.4f} | "
                     f"direct alone {dl[L10]:.4f} / {dl[L11]:.4f} | "
                     f"blend {res['HEADLINE + levels 10, 11'][1][L10]:.4f} / {res['HEADLINE + levels 10, 11'][1][L11]:.4f}")
        lines.append("")
        summ[win] = {n: res[n][0] for n, _ in variants}
    if summ:
        lines.append("SUMMARY (change = extended variant minus HEADLINE, negative is better)")
        for tag, wins in (("five windows before final", [x for x in ORDER[:-1] if x in summ]), ("all windows with results", list(summ))):
            if not wins:
                continue
            lines.append(f"  {tag}: {', '.join(wins)}")
            for n in ("HEADLINE + level 10", "HEADLINE + levels 10, 11"):
                ch = [summ[x][n] - summ[x]["HEADLINE levels 1-9"] for x in wins]
                k = sum(c < 0 for c in ch)
                lines.append(f"    {n:<28} mean change {np.mean(ch):+.4f} | improved {k} of {len(ch)} | sign test p = "
                             f"{sign_test_p(k, len(ch)):.3f} | worst {max(ch):+.4f}")
        lines += ["", "Notes: windows share training history; intervals resample days only; this is exploratory and was run after",
                  "the main result was fixed. If the five-window result is not clearly better, the headline stays unchanged."]
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "results_lvl1011.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--in_dir", default="inputs")
    ap.add_argument("--out_dir", default="lvl1011_out")
    ap.add_argument("--stage", default="all", choices=["all", "forecast", "score"])
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--context_length", type=int, default=1024)
    ap.add_argument("--mock", action="store_true")
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
