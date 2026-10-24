"""
E11: zero-shot foundation-model forecasts of the summed series (M5 levels 1 to 9, 154 series) for six windows.
One model per run (each in its own Python environment). Same inputs for every model as the Chronos-2 run:
  target = summed sales, context 1024 days; covariates known for the whole context and horizon:
  SNAP share of the group, event flag, weekday (7 one-hot columns). No training on M5.
usage: python m5_fm_forecast.py --model t0|timestral|timesfm3|tirex2g|mock --data_dir <M5 data> --out_dir fm_out
Output: fm_out/fm_<model>_<window>.npy (154 x 28 mean-like point), fm_<model>_<window>_median.npy, fm_<model>_log.txt
Windows: cv5 origin 1801, cv2 1829, cv3 1857, cv4 1885, public 1913, final 1941 (forecast the 28 days after the origin).
Existing output files are never overwritten.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, WRMSSEEvaluator, load_calendar, load_m5  # noqa: E402

ORIGINS = {"cv5": 1801, "cv2": 1829, "cv3": 1857, "cv4": 1885, "public": 1913, "final": 1941}
CTX = 1024
N_AGG_LEVELS = 9


def mean_weights(levels):
    p = np.concatenate([[0.0], np.asarray(levels, dtype=np.float64), [1.0]])
    dp = np.diff(p)
    w = np.zeros(len(p))
    w[:-1] += dp / 2
    w[1:] += dp / 2
    out = w[1:-1].copy()
    out[0] += w[0]
    out[-1] += w[-1]
    return out


def build(ev, meta, values, cal_full, o):
    """Targets (154 arrays of length CTX) and covariates (154 arrays (9, CTX + HORIZON)) for origin o."""
    a = max(o - CTX, 0)
    cal = cal_full.set_index("d").iloc[a:o + HORIZON]
    states = sorted(meta["state_id"].unique())
    onehot = np.zeros((len(meta), len(states)))
    onehot[np.arange(len(meta)), np.searchsorted(states, meta["state_id"].to_numpy())] = 1.0
    snap = np.stack([cal[f"snap_{s}"].to_numpy().astype(np.float32) for s in states])
    wd = cal["wday"].to_numpy().astype(int)
    wd_oh = np.stack([(wd == k).astype(np.float32) for k in range(1, 8)])
    event = cal["event_name_1"].notna().to_numpy().astype(np.float32)[None]
    T = o - a
    assert snap.shape[1] == T + HORIZON, (snap.shape, T)
    targets, covs = [], []
    for A in ev.A[:N_AGG_LEVELS]:
        Y = np.asarray(A @ values[:, a:o], dtype=np.float32)
        size = np.asarray(A.sum(axis=1)).ravel()
        share = np.asarray(A @ onehot) / size[:, None]
        snap_g = (share @ snap).astype(np.float32)
        for g in range(A.shape[0]):
            targets.append(Y[g])
            covs.append(np.concatenate([snap_g[g][None], event, wd_oh], axis=0))
    return targets, covs


# ---------------------------------------------------------------- adapters: return (mean_point, median) each (n, HORIZON)
class Mock:
    name, used_covariates = "mock", True

    def predict(self, targets, covs):
        out = np.stack([np.tile(t[-7:], 4) for t in targets]).astype(np.float32)
        return out, out


class T0:
    name, used_covariates = "t0", True

    def __init__(self):
        import torch
        from t0 import T0Forecaster
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma").eval().to(self.dev)
        self.levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    def predict(self, targets, covs):
        t = self.torch
        means, meds = [], []
        for i in range(0, len(targets), 16):
            ctx = t.tensor(np.stack(targets[i:i + 16]), dtype=t.float32, device=self.dev)
            fut = t.tensor(np.stack(covs[i:i + 16]), dtype=t.float32, device=self.dev)
            with t.no_grad():
                out = self.model.predict(ctx, horizon=HORIZON, quantile_levels=self.levels, future_covariates=fut)
            q = out.quantiles.float().cpu().numpy()
            means.append(np.tensordot(q, mean_weights(self.levels), axes=([2], [0])))
            meds.append(out.median.float().cpu().numpy())
        return np.concatenate(means), np.concatenate(meds)


class Timestral:
    name, used_covariates = "timestral", True

    def __init__(self):
        import torch
        from timestral import TimestralForecaster
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TimestralForecaster.from_pretrained("mistralai/Timestral-209M", device=self.dev)

    def predict(self, targets, covs):
        means, meds = [], []
        for i in range(0, len(targets), 16):
            ctx = np.stack(targets[i:i + 16])[:, None, :].astype(np.float32)  # (B, 1, T)
            kn = np.stack(covs[i:i + 16]).astype(np.float32)  # (B, C, T + H)
            fc = self.model.forecast(ctx, prediction_length=HORIZON, known_covariates=kn)
            q = np.asarray(fc.quantiles)[:, 0]  # (B, H, 21)
            lv = np.asarray(fc.quantile_levels, dtype=np.float64)
            means.append(np.tensordot(q, mean_weights(lv), axes=([2], [0])))
            meds.append(np.asarray(fc.median)[:, 0])
        return np.concatenate(means), np.concatenate(meds)


class TimesFM3:
    name, used_covariates = "timesfm3", True

    def __init__(self):
        from timesfm3 import ModelConfig, TimesFM3Evaluator
        cfg = ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=16, device="cuda")
        self.fc = TimesFM3Evaluator(cfg)
        self.mode = None

    def _run(self, targets, covs, one_d):
        ctx = [t if one_d else t[None, :] for t in targets]
        outs = list(self.fc.predict_batch(contexts=ctx, horizon=HORIZON, past_future_covariates=list(covs), return_quantiles=True))
        pts = np.stack([np.asarray(o.forecast, dtype=np.float64).reshape(-1)[:HORIZON] for o in outs])
        qs = np.stack([np.asarray(o.quantiles, dtype=np.float64).reshape(HORIZON, -1) for o in outs])
        return pts, qs[:, :, qs.shape[2] // 2]

    def predict(self, targets, covs):
        pts, meds = [], []
        for i in range(0, len(targets), 32):
            tg, cv = targets[i:i + 32], covs[i:i + 32]
            if self.mode is None:  # the README is unclear whether a single variate is (1, T) or (T,): try both once
                try:
                    p, m = self._run(tg, cv, False)
                    self.mode = False
                except Exception as e:  # noqa: BLE001
                    print("  timesfm3: (1, T) input failed:", repr(e)[:200], "- trying 1-D input", flush=True)
                    p, m = self._run(tg, cv, True)
                    self.mode = True
            else:
                p, m = self._run(tg, cv, self.mode)
            pts.append(p)
            meds.append(m)
        return np.concatenate(pts), np.concatenate(meds)


class TiRex2G:
    name = "tirex2g"

    def __init__(self):
        import torch
        from tirex2 import TimeseriesType, load_model
        self.torch, self.TS = torch, TimeseriesType
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = load_model("NX-AI/TiRex-2-gifteval-zs", device=self.dev)
            self.repo = "NX-AI/TiRex-2-gifteval-zs"
        except Exception as e:  # noqa: BLE001
            print("  tirex2g: decontaminated variant failed to load:", repr(e)[:200], "- using NX-AI/TiRex-2", flush=True)
            self.model = load_model("NX-AI/TiRex-2", device=self.dev)
            self.repo = "NX-AI/TiRex-2"
        self.used_covariates = True
        self.variant = None
        self.levels = np.arange(1, 10) / 10.0

    def _ts(self, t, c, variant):
        T = self.torch
        tg = T.tensor(t, dtype=T.float32)[None, :]
        if variant == "none":
            return self.TS(target=tg, past_covariates=None, future_covariates=None)
        L = len(t)
        c = T.tensor(c, dtype=T.float32)
        if variant == "past_ctx_future_h":
            return self.TS(target=tg, past_covariates=c[:, :L], future_covariates=c[:, L:])
        return self.TS(target=tg, past_covariates=None, future_covariates=c)  # "future_all"

    def _forecast(self, targets, covs, variant):
        res = []
        for i in range(0, len(targets), 16):
            ts = [self._ts(t, c, variant) for t, c in zip(targets[i:i + 16], covs[i:i + 16])]
            out = self.model.forecast(timeseries=ts, prediction_length=HORIZON, output_type="numpy")
            res += [np.asarray(o).reshape(-1, 9, HORIZON)[0] for o in out]  # (9, H) for the first target row
        q = np.stack(res).transpose(0, 2, 1)  # (n, H, 9)
        assert np.isfinite(q).all()
        return np.tensordot(q, mean_weights(self.levels), axes=([2], [0])), q[:, :, 4]

    def predict(self, targets, covs):
        if self.variant is None:
            for v in ["past_ctx_future_h", "future_all", "none"]:
                try:
                    out = self._forecast(targets[:16], covs[:16], v)
                    self.variant = v
                    self.used_covariates = v != "none"
                    print(f"  tirex2g: covariate layout that worked: {v} ({self.repo})", flush=True)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  tirex2g: layout {v} failed:", repr(e)[:200], flush=True)
            if self.variant is None:
                raise RuntimeError("TiRex-2 failed with every covariate layout")
        return self._forecast(targets, covs, self.variant)


MODELS = {"mock": Mock, "t0": T0, "timestral": Timestral, "timesfm3": TimesFM3, "tirex2g": TiRex2G}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--out_dir", default="fm_out")
    ap.add_argument("--windows", default=",".join(ORIGINS))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("M5_WINDOW", "final")
    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    dcols = list(dcols)
    cal_full = load_calendar(args.data_dir)
    log = os.path.join(args.out_dir, f"fm_{args.model}_log.txt")

    def say(m):
        print(m, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(m + "\n")

    say(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] model {args.model}, series {values.shape[0]}, days {values.shape[1]}, calendar days {len(cal_full)}")
    wins = [w for w in args.windows.split(",") if w in ORIGINS]
    todo = [w for w in wins if not os.path.exists(os.path.join(args.out_dir, f"fm_{args.model}_{w}.npy")) and values.shape[1] >= ORIGINS[w]]
    if not todo:
        say("nothing to do")
        return
    t0 = time.time()
    model = MODELS[args.model]()
    say(f"model loaded in {(time.time() - t0) / 60:.1f} min")
    for w in todo:
        o = ORIGINS[w]
        t1 = time.time()
        ev = WRMSSEEvaluator(meta, values[:, :o], dcols[:o], calendar, prices)  # only the aggregation matrices are used here
        targets, covs = build(ev, meta, values, cal_full, o)
        mean, med = model.predict(targets, covs)
        mean = np.clip(np.asarray(mean, dtype=np.float32), 0, None)
        med = np.clip(np.asarray(med, dtype=np.float32), 0, None)
        assert mean.shape == (154, HORIZON) and np.isfinite(mean).all(), mean.shape
        tag = args.model if getattr(model, "used_covariates", True) else args.model + "_nocov"
        np.save(os.path.join(args.out_dir, f"fm_{tag}_{w}.npy"), mean)
        np.save(os.path.join(args.out_dir, f"fm_{tag}_{w}_median.npy"), med)
        ratio = float(mean[0].sum() / values[:, o - HORIZON:o].sum())
        say(f"  {w} (origin d_{o}) done in {(time.time() - t1) / 60:.1f} min | saved as fm_{tag}_{w}.npy | level 1 forecast total / previous 28 days actual {ratio:.3f}")
    say("done")


if __name__ == "__main__":
    main()
