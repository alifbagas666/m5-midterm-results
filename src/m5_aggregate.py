"""
E7: Direct forecasts on the aggregate series (M5 levels 1 to 9) with zero-shot Chronos-2.

Idea (hypothesis, not a result): the bottom-up sums of Chronos-2 / PatchTST bottom forecasts are weak at the top
of the hierarchy (Level 1: 0.37 against 0.20 for the winner average). Forecasting the aggregate series directly,
where the data are smooth, may do better. This script produces those direct forecasts and scores them.

Three origins, all scored on the competition-style 28-day window that follows the origin:
  final  : origin d_1941, forecast d_1942..d_1969   (D-28 where D = number of days available)
  public : origin d_1913, forecast d_1914..d_1941
  cv4    : origin d_1885, forecast d_1886..d_1913
The weights of the evaluator for each window are computed only from data before the origin.

Variants: A_univariate (no covariates), B_covariates (weekday, event, SNAP share of the group).
No training on M5 is done. Level 10 to 12 are not forecast here (they stay bottom-up).

Outputs (plain text and npy, in --out_dir):
  agg_chronos2_{variant}_{origin}.npy   shape (154, 28): levels 1 to 9 stacked in level order
  results_aggregate.txt                 per level scores, direct vs optional bottom-up references

Usage:
  python m5_aggregate.py --data_dir m5_data --out_dir m5_outputs_agg
  optional bottom-up reference (repeatable), all three files are (30490, 28) arrays:
  --bu "name:final.npy:public.npy:cv4.npy"
  python m5_aggregate.py --selftest
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import (  # noqa: E402
    HORIZON, LEVELS, WRMSSEEvaluator, load_m5, load_calendar, make_synthetic,
)
from m5_chronos2 import MockPipeline, augment_calendar, quantile_mean_weights, to_np  # noqa: E402

N_AGG_LEVELS = 9  # levels 1..9


def level_slices(sizes):
    out, s = [], 0
    for g in sizes[:N_AGG_LEVELS]:
        out.append((s, s + g))
        s += g
    return out


def build_inputs(ev, meta, values, cal_full, dcols, origin, ctx_len, with_cov):
    """Aggregated target + covariates for every group at levels 1..9, using days before `origin` only for the target."""
    cal = cal_full.set_index("d").loc[dcols]
    states = sorted(meta["state_id"].unique())
    state_code = np.searchsorted(states, meta["state_id"].to_numpy())
    onehot = np.zeros((len(meta), len(states)))
    onehot[np.arange(len(meta)), state_code] = 1.0
    snap = np.stack([cal[f"snap_{s}"].to_numpy().astype(np.float32) for s in states])  # (S, D)
    wd = np.array([f"w{x}" for x in cal["wday"].to_numpy().astype(int)])
    event = cal["event_name_1"].notna().to_numpy().astype(np.float32)
    a = max(origin - ctx_len, 0)
    inputs = []
    for A in ev.A[:N_AGG_LEVELS]:
        Y = np.asarray(A @ values[:, :origin], dtype=np.float32)  # (G, origin)
        size = np.asarray(A.sum(axis=1)).ravel()
        share = np.asarray(A @ onehot) / size[:, None]  # (G, S) share of members per state
        snap_g = (share @ snap).astype(np.float32)  # (G, D)
        for g in range(A.shape[0]):
            tgt = Y[g, a:origin]
            if not with_cov:
                inputs.append(tgt)
                continue
            inputs.append({
                "target": tgt,
                "past_covariates": {"snap": snap_g[g, a:origin], "wday": wd[a:origin], "event": event[a:origin]},
                "future_covariates": {"snap": snap_g[g, origin:origin + HORIZON],
                                      "wday": wd[origin:origin + HORIZON],
                                      "event": event[origin:origin + HORIZON]},
            })
    return inputs


def forecast(pipe, inputs, weights, med_idx):
    preds = pipe.predict(inputs, prediction_length=HORIZON, batch_size=64, cross_learning=False)
    arr = np.stack([to_np(t)[0] for t in preds])  # (n, n_q, H)
    mean = np.tensordot(arr, weights, axes=([1], [0]))
    return np.clip(mean, 0, None).astype(np.float32), np.clip(arr[:, med_idx, :], 0, None).astype(np.float32)


def score_direct(ev, agg_pred, slices):
    """RMSSE-weighted score per level for direct forecasts of levels 1..9."""
    out = []
    for i, (s, e) in enumerate(slices):
        P = agg_pred[s:e]
        Y, sc, w = ev.y_agg[i], ev.scale[i], ev.weight[i]
        valid = w > 0
        r = np.zeros(len(sc))
        r[valid] = np.sqrt(np.mean((Y[valid] - P[valid]) ** 2, axis=1) / sc[valid])
        out.append(float(np.sum(w * r)))
    return out


def bottom_up_levels(ev, bottom_pred):
    P = np.asarray(bottom_pred, dtype=np.float64)
    return ev.score(P)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default="m5_outputs_agg")
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--context_length", type=int, default=1024)
    ap.add_argument("--bu", action="append", default=[], help="name:final.npy:public.npy:cv4.npy")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    if args.selftest:
        tmp = tempfile.mkdtemp()
        make_synthetic(tmp, n_days=300)
        augment_calendar(tmp, 300)
        data_dir, args.context_length = tmp, 120
    else:
        data_dir = args.data_dir or "m5_data"
    os.environ.setdefault("M5_WINDOW", "final")
    meta, values, dcols, calendar, prices = load_m5(data_dir)
    cal_full = load_calendar(data_dir)
    n, D = values.shape
    dcols = list(dcols)
    origins = {"final": D - HORIZON, "public": D - 2 * HORIZON, "cv4": D - 3 * HORIZON}
    print(f"Series {n}, days {D}, origins {origins}", flush=True)

    if args.selftest:
        pipe = MockPipeline()
    else:
        import torch
        from chronos import Chronos2Pipeline
        cuda = torch.cuda.is_available()
        print("Loading", args.model, "cuda" if cuda else "CPU", flush=True)
        pipe = Chronos2Pipeline.from_pretrained(args.model, device_map="cuda" if cuda else "cpu")
    weights = quantile_mean_weights(pipe.quantiles)
    med_idx = int(np.argmin(np.abs(np.asarray(pipe.quantiles) - 0.5)))

    os.makedirs(args.out_dir, exist_ok=True)
    bu = {}
    for spec in args.bu:
        name, *paths = spec.split(":")
        # allow Windows drive letters: rejoin pieces like "C" + "\\path"
        fixed, buf = [], ""
        for p in paths:
            if len(p) == 1 and not buf:
                buf = p + ":"
            else:
                fixed.append(buf + p)
                buf = ""
        assert len(fixed) == 3, f"--bu needs name and three files, got {spec}"
        bu[name] = {k: np.load(f) for k, f in zip(["final", "public", "cv4"], fixed)}

    rows = {}  # (label, origin) -> list of 9 level scores
    for oname, o in origins.items():
        ev = WRMSSEEvaluator(meta, values[:, : o + HORIZON], dcols[: o + HORIZON], calendar, prices)
        slices = level_slices(ev.sizes)
        for variant, with_cov in [("A_univariate", False), ("B_covariates", True)]:
            inputs = build_inputs(ev, meta, values, cal_full, dcols[: o + HORIZON], o, args.context_length, with_cov)
            mean, med = forecast(pipe, inputs, weights, med_idx)
            np.save(os.path.join(args.out_dir, f"agg_chronos2_{variant}_{oname}.npy"), mean)
            np.save(os.path.join(args.out_dir, f"agg_chronos2_{variant}_{oname}_median.npy"), med)
            rows[(f"direct_{variant}", oname)] = score_direct(ev, mean, slices)
            print(f"{oname} {variant}: L1 {rows[(f'direct_{variant}', oname)][0]:.4f}", flush=True)
        for name, d in bu.items():
            rows[(f"bottomup_{name}", oname)] = bottom_up_levels(ev, d[oname])[:N_AGG_LEVELS]

    labels = sorted({k[0] for k in rows})
    lines = ["AGGREGATE-LEVEL DIRECT FORECASTS (Chronos-2 zero-shot), WRMSSE per level, lower is better",
             f"Origins: {origins} | context {args.context_length} | days available {D}",
             "Direct rows come from forecasting the summed series; bottomup rows come from the supplied bottom forecasts.",
             "Hypothesis test only; no weights fitted here.", ""]
    for oname in origins:
        lines.append(f"WINDOW: {oname} (forecast d_{origins[oname] + 1}..d_{origins[oname] + HORIZON})")
        head = f"{'Level':<24}" + "".join(f"{l[:26]:>28}" for l in labels)
        lines += [head, "-" * len(head)]
        for i in range(N_AGG_LEVELS):
            lines.append(f"{LEVELS[i][0]:>2} {LEVELS[i][1]:<21}" +
                         "".join(f"{rows[(l, oname)][i]:>28.4f}" if (l, oname) in rows else f"{'':>28}" for l in labels))
        lines.append(f"{'mean of L1..L9':<24}" +
                     "".join(f"{np.mean(rows[(l, oname)]):>28.4f}" if (l, oname) in rows else f"{'':>28}" for l in labels))
        lines.append("")
    lines.append(f"Runtime {(time.time() - t0) / 60:.1f} min")
    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "results_aggregate.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
