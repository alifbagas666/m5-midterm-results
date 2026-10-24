#!/usr/bin/env python3
"""
E2: PatchTST on M5 (global model, trained on M5 bottom-level series), with an optional covariate variant.

  P0_univariate   E2   PatchTST on sales history only (channel independent, patching, RevIN-style level scaling)
  P1_covariates   E2b  same backbone, plus known-future covariates (price ratio, SNAP, weekday, event)
                       concatenated into the forecast head

One global model per configuration and per window:
  holdout     trained on d_1..d_1913, forecasts d_1914..d_1941 (the 28-day test window)
  validation  trained on d_1..d_1885, forecasts d_1886..d_1913 (a separate training run)

The validation forecasts are what the ensemble weights are fitted on later, so nothing is chosen on the holdout.
Training settings are fixed in advance (see the config block) and are not tuned on the holdout.

Model
  input: last --context days of sales (default 224), divided by the series level (mean of the last 56 days + 0.1)
  patching: patch length 16, stride 8, transformer encoder (pre-norm), flatten head, direct 28-day output
  loss: MSE in level-scaled units, masked for days before the item was on sale
  point forecast: model output x level, clipped at zero

Outputs (in --out_dir)
  pred_patchtst_<cfg>.npy      holdout forecast, shape (30490, 28)
  predval_patchtst_<cfg>.npy   validation-window forecast, shape (30490, 28)
  results_patchtst.txt         plain-text log: ladder, 12-level table, training curves, timings

Usage
  python m5_patchtst.py --quick       # small training run to check the setup and timing
  python m5_patchtst.py               # full run (holdout and validation, both configurations)
  python m5_patchtst.py --configs P0_univariate    # one configuration only
  python m5_patchtst.py --selftest    # synthetic data, tiny model, no GPU needed

Requires m5_data_and_wrmsse.py in the same folder and a CUDA PyTorch for the full run.
"""
import argparse
import math
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import (  # noqa: E402
    HORIZON,
    WRMSSEEvaluator,
    find_file,
    format_table,
    load_m5,
    load_calendar,
    make_synthetic,
    seasonal_naive_forecast,
)

LEVEL_WIN = 56
LEVEL_EPS = 0.1


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def augment_calendar(folder, n_days, seed=1):
    rng = np.random.default_rng(seed)
    cal = pd.read_csv(os.path.join(folder, "calendar.csv"))
    dates = pd.date_range("2015-01-01", periods=n_days)
    cal["date"] = dates.strftime("%Y-%m-%d")
    cal["wday"] = dates.dayofweek + 1
    cal["month"] = dates.month
    ev = rng.random(n_days) < 0.05
    cal["event_name_1"] = np.where(ev, "EventX", None)
    cal["event_type_1"] = np.where(ev, "Cultural", None)
    for st in ["CA", "TX", "WI"]:
        cal[f"snap_{st}"] = (rng.random(n_days) < 0.3).astype(int)
    cal.to_csv(os.path.join(folder, "calendar.csv"), index=False)


class M5Tensors:
    """Sales (with pre-launch days masked), price, and shared calendar covariates, all as arrays."""

    def __init__(self, meta, values, dcols, cal_full, prices):
        self.n, self.D = values.shape
        cal = cal_full.set_index("d").loc[dcols]
        wk = cal["wm_yr_wk"].to_numpy()
        weeks = np.unique(wk)
        widx = np.searchsorted(weeks, wk)
        p = prices[prices["wm_yr_wk"].isin(weeks)]
        key = pd.MultiIndex.from_frame(meta[["item_id", "store_id"]])
        r = key.get_indexer(pd.MultiIndex.from_frame(p[["item_id", "store_id"]]))
        c = np.searchsorted(weeks, p["wm_yr_wk"].to_numpy())
        ok = r >= 0
        Pw = np.full((self.n, len(weeks)), np.nan, dtype=np.float32)
        Pw[r[ok], c[ok]] = p["sell_price"].to_numpy(dtype=np.float32)[ok]
        P = Pw[:, widx]
        valid = ~np.isnan(P)
        first = np.where(valid.any(axis=1), valid.argmax(axis=1), 0)
        fill = P[np.arange(self.n), first]
        fill = np.where(np.isnan(fill), 0.0, fill).astype(np.float32)
        P = np.where(valid, P, fill[:, None]).astype(np.float32)
        # price relative to the item's own running level (mean of its last 365 days), always causal-ish scale
        pm = np.maximum(np.nanmean(P, axis=1, keepdims=True), 1e-3)
        self.price_rel = (P / pm).astype(np.float32)
        self.mask = valid.astype(np.float32)  # 1 where item is on sale
        self.Y = np.where(valid, values, 0.0).astype(np.float32)

        states = sorted(meta["state_id"].unique())
        self.state_code = pd.factorize(meta["state_id"], sort=True)[0]
        self.snap = np.stack([cal[f"snap_{s}"].to_numpy().astype(np.float32) for s in states])  # (3, D)
        wd = cal["wday"].to_numpy().astype(int) - 1
        self.wday_onehot = np.eye(7, dtype=np.float32)[wd]  # (D, 7)
        self.event = cal["event_name_1"].notna().to_numpy().astype(np.float32)  # (D,)

    def future_cov(self, idx, start, h=HORIZON):
        """Known-future covariates for series idx over days [start, start+h): (len(idx), h, 10)."""
        sl = slice(start, start + h)
        m = len(idx)
        price = self.price_rel[idx, sl][:, :, None]
        snap = self.snap[self.state_code[idx], sl][:, :, None]
        wd = np.broadcast_to(self.wday_onehot[sl][None], (m, h, 7))
        ev = np.broadcast_to(self.event[sl][None, :, None], (m, h, 1))
        return np.concatenate([price, snap, wd, ev], axis=2).astype(np.float32)


N_COV = 10


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(cfg, use_cov):
    import torch
    import torch.nn as nn

    class PatchTST(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx, self.pl, self.st = cfg["context"], cfg["patch_len"], cfg["stride"]
            self.n_patch = (self.ctx - self.pl) // self.st + 1
            d = cfg["d_model"]
            self.embed = nn.Linear(self.pl, d)
            self.pos = nn.Parameter(torch.randn(1, self.n_patch, d) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg["heads"], dim_feedforward=cfg["ff"], dropout=cfg["dropout"],
                batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=cfg["layers"], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.use_cov = use_cov
            self.cov_dim = HORIZON * N_COV if use_cov else 0
            self.head = nn.Sequential(
                nn.Dropout(cfg["dropout"]),
                nn.Linear(self.n_patch * d + self.cov_dim, cfg["head_hidden"]),
                nn.GELU(),
                nn.Linear(cfg["head_hidden"], HORIZON),
            )

        def forward(self, x, cov=None):
            # x: (B, ctx) level-scaled sales
            p = x.unfold(1, self.pl, self.st)  # (B, n_patch, pl)
            z = self.enc(self.embed(p) + self.pos)
            z = self.norm(z).flatten(1)
            if self.use_cov:
                z = torch.cat([z, cov.flatten(1)], dim=1)
            return self.head(z)

    return PatchTST()


# ----------------------------------------------------------------------------
# Train / predict for one window
# ----------------------------------------------------------------------------
def make_batch(T_, idx, ends, ctx, use_cov, torch, dev):
    """Gather contexts ending at `ends` (exclusive) and the 28-day targets after them."""
    pos_c = ends[:, None] + np.arange(-ctx, 0)[None, :]
    pos_t = ends[:, None] + np.arange(HORIZON)[None, :]
    Yc = T_.Y[idx[:, None], np.clip(pos_c, 0, None)] * (pos_c >= 0)
    Mc = T_.mask[idx[:, None], np.clip(pos_c, 0, None)] * (pos_c >= 0)
    lv = (Yc[:, -LEVEL_WIN:].sum(1) / np.maximum(Mc[:, -LEVEL_WIN:].sum(1), 1.0)) + LEVEL_EPS
    out = {"x": Yc / lv[:, None], "lv": lv}
    if pos_t.max() < T_.D:
        out["y"] = T_.Y[idx[:, None], pos_t] / lv[:, None]
        out["m"] = T_.mask[idx[:, None], pos_t]
    if use_cov:
        out["cov"] = _cov_batch(T_, idx, ends)
    return {k: torch.from_numpy(np.ascontiguousarray(v)).float().to(dev) for k, v in out.items()}


def _cov_batch(T_, idx, ends):
    pos_t = ends[:, None] + np.arange(HORIZON)[None, :]
    price = T_.price_rel[idx[:, None], pos_t][:, :, None]
    snap = T_.snap[T_.state_code[idx][:, None], pos_t][:, :, None]
    wd = T_.wday_onehot[pos_t]
    ev = T_.event[pos_t][:, :, None]
    return np.concatenate([price, snap, wd, ev], axis=2).astype(np.float32)


def train_and_predict(T_, origin, use_cov, cfg, steps, dev, log, seed=0):
    """Train on data before `origin`, forecast origin..origin+28. Returns (n, 28) forecast."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ctx = cfg["context"]
    model = build_model(cfg, use_cov).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"    model parameters: {n_params / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # training end points: forecast target must lie fully before `origin`
    last_end = origin - HORIZON
    first_end = max(ctx // 2, last_end - cfg["train_span"])
    n = T_.n
    curve = []
    t0 = time.time()
    model.train()
    run = 0.0
    for step in range(steps):
        idx = rng.integers(0, n, size=cfg["batch"])
        ends = rng.integers(first_end, last_end + 1, size=cfg["batch"])
        b = make_batch(T_, idx, ends, ctx, use_cov, torch, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            out = model(b["x"], b.get("cov"))
        out = out.float()
        err = (out - b["y"]) ** 2 * b["m"]
        loss = err.sum() / b["m"].sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if (step + 1) % max(1, steps // 10) == 0:
            curve.append((step + 1, run))
            log(f"    step {step + 1}/{steps} | smoothed train loss {run:.4f} | {time.time() - t0:.0f} s")

    model.eval()
    preds = np.zeros((n, HORIZON), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, 2048):
            idx = np.arange(s, min(s + 2048, n))
            ends = np.full(len(idx), origin)
            b = make_batch(T_, idx, ends, ctx, use_cov, torch, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                o = model(b["x"], b.get("cov"))
            preds[idx] = (o.float() * b["lv"][:, None]).cpu().numpy()
    return np.clip(preds, 0, None), curve


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
CONFIG = dict(context=224, patch_len=16, stride=8, d_model=128, heads=8, ff=256, layers=3,
              dropout=0.1, head_hidden=512, lr=5e-4, wd=1e-2, batch=512, train_span=1100)
ALL_CONFIGS = {"P0_univariate": False, "P1_covariates": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default="m5_outputs")
    ap.add_argument("--steps", type=int, default=12000, help="training steps per model")
    ap.add_argument("--configs", default="P0_univariate,P1_covariates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_val", action="store_true")
    ap.add_argument("--quick", action="store_true", help="400 steps, holdout only, prints a time estimate")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    cfg = dict(CONFIG)

    if args.selftest:
        tmp = tempfile.mkdtemp()
        make_synthetic(tmp, n_days=240)
        augment_calendar(tmp, 240)
        data_dir = tmp
        cfg.update(context=64, d_model=32, heads=4, ff=64, layers=1, head_hidden=64, batch=128, train_span=100)
        args.steps = 60
    elif args.data_dir:
        data_dir = args.data_dir
    elif os.path.isdir("m5_data"):
        data_dir = "m5_data"
    else:
        import kagglehub
        data_dir = kagglehub.competition_download("m5-forecasting-accuracy")
    log(f"Data folder: {data_dir}")

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_note = f"cuda: {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else "CPU (no CUDA found, this will be slow)"
    log(f"Device: {dev_note}")

    meta, values, dcols, calendar, prices = load_m5(data_dir)
    cal_full = load_calendar(data_dir)
    n, D = values.shape
    T = D - HORIZON
    log(f"Series: {n} | days: {D} | holdout origin: {T} | validation origin: {T - HORIZON}")
    T_ = M5Tensors(meta, values, dcols, cal_full, prices)

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in names:
        if c not in ALL_CONFIGS:
            raise SystemExit(f"Unknown config {c}. Choose from {list(ALL_CONFIGS)}")

    if args.quick:
        steps = 400
        for c in names:
            t1 = time.time()
            f, curve = train_and_predict(T_, T, ALL_CONFIGS[c], cfg, steps, dev, log, args.seed)
            dt = time.time() - t1
            log(f"{c}: {steps} steps in {dt:.0f} s | NaN count {int(np.isnan(f).sum())} | "
                f"estimated {dt / steps * args.steps / 60:.1f} min per model at {args.steps} steps "
                f"({'x2 with validation' if not args.no_val else 'holdout only'})")
        return

    ev_hold = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    ev_val = None if args.no_val else WRMSSEEvaluator(meta, values[:, :T], dcols[:T], calendar, prices)

    os.makedirs(args.out_dir, exist_ok=True)
    results, ladder, curves = {}, [], {}
    for c in names:
        t1 = time.time()
        use_cov = ALL_CONFIGS[c]
        log(f"\n{c}: holdout model (trained to d_{T})")
        f, curve = train_and_predict(T_, T, use_cov, cfg, args.steps, dev, log, args.seed)
        s = ev_hold.score(f)
        np.save(os.path.join(args.out_dir, f"pred_patchtst_{c}.npy"), f)
        curves[c] = curve
        vs = None
        if ev_val is not None:
            log(f"{c}: validation model (trained to d_{T - HORIZON})")
            fv, _ = train_and_predict(T_, T - HORIZON, use_cov, cfg, args.steps, dev, log, args.seed)
            vs = ev_val.score(fv)
            np.save(os.path.join(args.out_dir, f"predval_patchtst_{c}.npy"), fv)
        mins = (time.time() - t1) / 60
        results[c] = s
        ladder.append((c, s[0], None if vs is None else vs[0], mins))
        log(f"{c}: holdout WRMSSE {s[0]:.4f} | validation {'n/a' if vs is None else f'{vs[0]:.4f}'} | {mins:.1f} min")

    table = {"SeasonalNaive": ev_hold.score(seasonal_naive_forecast(ev_hold.X_train))}
    table.update(results)
    lines = [f"{'Config':<20}{'Holdout':>10}{'Validation':>12}{'Min':>7}"]
    for c, h, v, m in ladder:
        lines.append(f"{c:<20}{h:>10.4f}{('n/a' if v is None else f'{v:.4f}'):>12}{m:>7.1f}")
    curve_lines = []
    for c, cv in curves.items():
        curve_lines.append(f"  {c}: " + ", ".join(f"{s}:{l:.4f}" for s, l in cv))

    report = [
        "PATCHTST ON M5 (global model trained on M5)",
        f"Data folder: {data_dir}",
        f"Device: {dev_note} | steps per model: {args.steps} | seed: {args.seed}",
        "Settings (fixed in advance, not tuned on the holdout): " + ", ".join(f"{k}={v}" for k, v in cfg.items()),
        f"Training window: d_1 to d_{T} | Holdout (28 days): d_{T + 1} to d_{D}",
        "Validation models are trained separately on data up to d_%d and forecast d_%d to d_%d." % (T - HORIZON, T - HORIZON + 1, T),
        "",
        "LADDER (WRMSSE, lower is better)",
        *lines,
        "",
        "TRAINING CURVES (step:smoothed loss, holdout model)",
        *curve_lines,
        "",
        "WRMSSE BY LEVEL (holdout)",
        format_table(ev_hold, table),
        "",
        "REFERENCE VALUES (published, window to be confirmed)",
        "  Original M5 winner: 0.520 | Yang et al. Top-1 reproduction: 0.5230 | Yang et al. LightGBM + PatchTST: 0.4989",
        "  Our E1b L3 (LightGBM, unweighted, all features): 0.5422",
        "",
        f"Runtime: {(time.time() - t0) / 60:.1f} min",
    ]
    text = "\n".join(report)
    path = os.path.join(args.out_dir, "results_patchtst.txt")
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
