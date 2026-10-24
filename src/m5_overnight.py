"""
OVERNIGHT RUN: every idea to lower WRMSSE below the 0.20 headline, tested honestly. One script, plain-text output.

PART A (CPU, saved forecasts, minutes): writes results_overnight_A.txt
PART B (GPU, Chronos-2, about 15 to 40 minutes): new direct forecasts at levels 1 to 9
        ctx512 / ctx2048   : same model and covariates, 512 days or the whole history as context
        grpLevel           : cross-learning, all series of one level are forecast together
        grpAll             : cross-learning, all 154 aggregate series forecast together
PART C (CPU): repeats part A with the new forecasts as extra candidates: writes results_overnight_FINAL.txt

IDEAS TESTED (levels 1 to 9 blend with a direct foundation-model forecast; level 12 stays the winner)
  1  TOP-DOWN ALIGNMENT: the winner's item forecasts are multiplied by (blended aggregate / winner aggregate) of their parent
     series (level 1, 3 or 9; whole 28 days or per day; shrunk by a power alpha). Items then add up to the blend (second-place trick).
  2  HORIZON SHARE: the share changes linearly over the 28 days.
  3  SHRUNK correction (alpha < 1) inside idea 1.
  4  PER-GROUP / PER-LEVEL share, and a choice of direct forecast (Chronos-2, other models, averages, new GPU variants).
  5  STACK: best group blend, then the best alignment on top.
HONEST TEST: every rule picks its settings WITHOUT the window it is scored on. The five windows before the final one use the
other four of those five; the final window uses those five and is scored once. Nothing is tuned on the final window.

Usage:
  python m5_overnight.py --data_dir <M5 data with 1969 days> --in_dir inputs --fm_dir fm_out --lvl_dir lvl1011_out --out_dir overnight_out
  python m5_overnight.py --selftest        (synthetic data and a mock model, no GPU)
"""
import argparse
import glob
import itertools
import os
import re
import sys
import tempfile
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, WRMSSEEvaluator, load_calendar, load_m5, make_synthetic  # noqa: E402
from m5_multiwindow import sign_test_p  # noqa: E402

WINS = ["cv5", "cv2", "cv3", "cv4", "public", "final"]
GRID = np.round(np.arange(0.0, 0.601, 0.05), 2)
I20 = int(np.where(GRID == 0.20)[0][0])
GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
GROUP_NAMES = ["L1-3", "L4-6", "L7-9", "L10-11"]
ALIGN_CFGS = [(L, mode, a) for L in (0, 2, 8) for mode in ("whole", "daily") for a in (0.5, 0.75, 1.0)]
H_GRID = [(a, b) for a in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40) for b in (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)]
LOG = None


def say(msg):
    print(msg, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")


# ------------------------------------------------------------------ window container
class Win:
    def __init__(self, name, ev, winner, cands, d10, d11):
        self.name, self.ev = name, ev
        self.Wbot = winner.astype(np.float64)
        self.Wagg = [np.asarray(A @ self.Wbot) for A in ev.A]
        self.cands, self.D10, self.D11 = cands, d10.astype(np.float64), d11.astype(np.float64)
        self.off = np.cumsum([0] + ev.sizes[:9])
        self.grp = [np.asarray(A.argmax(axis=0)).ravel() for A in ev.A[:9]]
        self.cache, self.lcache = {}, {}

    def direct(self, cand, i):
        if i == 9:
            return self.D10
        if i == 10:
            return self.D11
        return self.cands[cand][self.off[i]:self.off[i + 1]].astype(np.float64)

    def score(self, i, P):
        ev = self.ev
        w = ev.weight[i]
        v = w > 0
        r = np.sqrt(np.mean((ev.y_agg[i][v] - P[v]) ** 2, axis=1) / ev.scale[i][v])
        return float((w[v] * r).sum())

    def ls(self, cand, i, k):
        """Level score of level i (0-based) with share GRID[k] of candidate `cand`."""
        if i == 11 or k == 0:
            key = ("W", i)
        else:
            key = ("C2" if i >= 9 else cand, i, k)
        if key not in self.lcache:
            if key[0] == "W":
                self.lcache[key] = self.score(i, self.Wagg[i])
            else:
                s = GRID[k]
                self.lcache[key] = self.score(i, (1 - s) * self.Wagg[i] + s * self.direct(key[0], i))
        return self.lcache[key]


def levels_blend(w, spec):
    return [w.ls(spec[i][0], i, spec[i][1]) for i in range(11)] + [w.ls("C2", 11, 0)]


def levels_h(w, a, b):
    h = np.arange(HORIZON)
    s = np.clip(a + b * (h - 13.5) / 13.5, 0, 0.6)[None, :]
    lv = [w.score(i, (1 - s) * w.Wagg[i] + s * w.direct("C2", i)) for i in range(9)]
    return lv + [w.ls("C2", 9, 0), w.ls("C2", 10, 0), w.ls("C2", 11, 0)]


def levels_align(w, spec, cfg):
    L, mode, alpha = cfg
    cand, k = spec[L]
    if k == 0:
        return levels_blend(w, spec)
    s = GRID[k]
    Bl = (1 - s) * w.Wagg[L] + s * w.direct(cand, L)
    if mode == "whole":
        r = Bl.sum(axis=1, keepdims=True) / np.maximum(w.Wagg[L].sum(axis=1, keepdims=True), 1e-9)
    else:
        r = Bl / np.maximum(w.Wagg[L], 1e-9)
    r = np.clip(r, 0.5, 2.0) ** alpha
    scaled = w.Wbot * r[w.grp[L]]
    lv = []
    for i, A in enumerate(w.ev.A):
        P = np.asarray(A @ scaled)
        if i in (9, 10) and spec[i][1] > 0:
            s2 = GRID[spec[i][1]]
            P = (1 - s2) * P + s2 * w.direct("C2", i)
        lv.append(w.score(i, P))
    return lv


def levels_cfg(w, spec, cfg):
    key = (spec, cfg)
    if key not in w.cache:
        if cfg is None:
            w.cache[key] = levels_blend(w, spec)
        elif cfg[0] == "H":
            w.cache[key] = levels_h(w, cfg[1], cfg[2])
        else:
            w.cache[key] = levels_align(w, spec, cfg)
    return w.cache[key]


def cost_cfg(w, spec, cfg):
    return float(np.mean(levels_cfg(w, spec, cfg)))


# ------------------------------------------------------------------ selection rules
def sel_groups(W, tr, groups, cands):
    spec = [None] * 11
    for g in groups:
        best = None
        cl = ["C2"] if min(g) >= 9 else cands
        for cand in cl:
            for k in range(len(GRID)):
                c = np.mean([sum(W[x].ls(cand, i, k) for i in g) for x in tr])
                if best is None or c < best[0] - 1e-12:
                    best = (c, cand, k)
        for i in g:
            spec[i] = (best[1], best[2]) if best[2] > 0 else ("C2", 0)
    return tuple(spec)


def sel_global(W, tr, upto):
    best = min(range(len(GRID)), key=lambda k: np.mean(
        [sum(W[x].ls("C2", i, k) for i in range(upto)) + sum(W[x].ls("C2", i, 0) for i in range(upto, 12)) for x in tr]))
    return tuple([("C2", best if best else 0)] * upto + [("C2", 0)] * (11 - upto))


def sel_cfg(W, tr, spec, cfgs):
    best = (np.mean([cost_cfg(W[x], spec, None) for x in tr]), None)
    for c in cfgs:
        v = np.mean([cost_cfg(W[x], spec, c) for x in tr])
        if v < best[0] - 1e-12:
            best = (v, c)
    return best[1]


def spec_text(spec):
    parts = []
    for n, g in zip(GROUP_NAMES, GROUPS):
        c, k = spec[g[0]]
        same = all(spec[i] == spec[g[0]] for i in g)
        parts.append(f"{n}={GRID[k]:.2f}" + (f"({c})" if k > 0 and c != "C2" else "") + ("" if same else "*"))
    return " ".join(parts)


def cfg_text(cfg):
    if cfg is None:
        return "no alignment"
    if cfg[0] == "H":
        return f"share a={cfg[1]:.2f} b={cfg[2]:+.2f}"
    return f"align L{cfg[0] + 1} {cfg[1]} alpha {cfg[2]}"


# ------------------------------------------------------------------ analysis
def analysis(args, meta, values, dcols, calendar, prices, D, evs, out_name, extra_gpu):
    wins_all, W, tags = [], {}, {}
    for idx, name in enumerate(WINS):
        o = D - HORIZON * (6 - idx)
        if o < 400 and not args.selftest:
            continue
        p_w = os.path.join(args.in_dir, f"winner_{name}.npy")
        p_c = os.path.join(args.in_dir, f"agg_chronos2_B_covariates_{name}.npy")
        p10 = os.path.join(args.lvl_dir, f"lvl10_chronos2_B_{name}.npy")
        p11 = os.path.join(args.lvl_dir, f"lvl11_chronos2_B_{name}.npy")
        if not all(os.path.exists(p) for p in (p_w, p_c, p10, p11)):
            say(f"window {name}: skipped (missing winner, Chronos-2 or level 10/11 file)")
            continue
        if name not in evs:
            evs[name] = WRMSSEEvaluator(meta, values[:, : o + HORIZON], dcols[: o + HORIZON], calendar, prices)
        cands = {"C2": np.load(p_c)}
        for p in glob.glob(os.path.join(args.fm_dir, f"fm_*_{name}.npy")):
            m = re.match(rf"fm_(.+)_{name}\.npy$", os.path.basename(p))
            if m and not m.group(1).endswith("median"):
                cands["FM_" + m.group(1)] = np.load(p)
        for p in glob.glob(os.path.join(args.out_dir, "gpu_fc", f"agg_chronos2_B_*_{name}.npy")):
            m = re.match(rf"agg_chronos2_B_(.+)_{name}\.npy$", os.path.basename(p))
            if m:
                cands["G_" + m.group(1)] = np.load(p)
        tags[name] = cands
        W[name] = (name, evs[name], np.load(p_w), cands, np.load(p10), np.load(p11))
        wins_all.append(name)
    if len(wins_all) < 4:
        say("Not enough windows for the analysis.")
        return None
    common = set.intersection(*[set(tags[n]) for n in wins_all])
    base = sorted(common)
    derived = {}
    fm = [c for c in base if c.startswith("FM_")]
    if fm:
        derived["AVG_ALL"] = ["C2"] + fm
    if "FM_timesfm3" in base:
        derived["AVG_C2_TFM"] = ["C2", "FM_timesfm3"]
    gx = [c for c in ("G_ctx512", "G_ctx2048") if c in base]
    if len(gx) == 2:
        derived["AVG_CTX3"] = ["C2"] + gx
    for n in wins_all:
        for dn, parts in derived.items():
            tags[n][dn] = np.mean([tags[n][p] for p in parts], axis=0)
    cands = ["C2"] + [c for c in sorted(set(base) | set(derived)) if c != "C2"]
    W = {n: Win(*W[n]) for n in wins_all}
    pre = [n for n in wins_all if n != "final"]
    say(f"[{out_name}] windows {wins_all}; candidates {cands}")

    H = tuple([("C2", I20)] * 9 + [("C2", 0)] * 2)
    EXT = tuple([("C2", I20)] * 11)
    WIN0 = tuple([("C2", 0)] * 11)
    res, desc, spc = {}, {}, {}

    def fixed(name, spec, cfg=None):
        res[name] = {n: cost_cfg(W[n], spec, cfg) for n in wins_all}
        desc[name] = {n: "" for n in wins_all}
        spc[name] = {n: (spec, cfg) for n in wins_all}

    def loo(name, rule):
        res[name], desc[name], spc[name] = {}, {}, {}
        for n in wins_all:
            tr = [x for x in pre if x != n]
            spec, cfg = rule(tr)
            res[name][n] = cost_cfg(W[n], spec, cfg)
            desc[name][n] = spec_text(spec) + ("" if cfg is None else " | " + cfg_text(cfg))
            spc[name][n] = (spec, cfg)

    fixed("winner alone", WIN0)
    fixed("HEADLINE 0.20, levels 1-9", H)
    fixed("EXTENSION 0.20, levels 1-11", EXT)
    t0 = time.time()
    loo("T1 GLOBAL share, levels 1-9", lambda tr: (sel_global(W, tr, 9), None))
    loo("T2 GLOBAL share, levels 1-11", lambda tr: (sel_global(W, tr, 11), None))
    loo("T3 PER-GROUP share (Chronos-2)", lambda tr: (sel_groups(W, tr, GROUPS, ["C2"]), None))
    loo("T4 PER-LEVEL share (Chronos-2)", lambda tr: (sel_groups(W, tr, [[i] for i in range(11)], ["C2"]), None))
    if len(cands) > 1:
        loo("M1 PER-GROUP share + choice of forecast", lambda tr: (sel_groups(W, tr, GROUPS, cands), None))
    say(f"  share rules done {time.time() - t0:.0f}s")

    def rule_h(tr):
        best = min(H_GRID, key=lambda ab: np.mean([cost_cfg(W[x], H, ("H",) + ab) for x in tr]))
        if np.mean([cost_cfg(W[x], H, ("H",) + best) for x in tr]) >= np.mean([cost_cfg(W[x], H, None) for x in tr]) - 1e-12:
            return H, None
        return H, ("H",) + best
    loo("H1 HORIZON share (levels 1-9)", rule_h)
    say(f"  horizon rule done {time.time() - t0:.0f}s")
    loo("A1 TOP-DOWN ALIGNMENT on the headline", lambda tr: (H, sel_cfg(W, tr, H, ALIGN_CFGS)))
    say(f"  alignment rule done {time.time() - t0:.0f}s")
    loo("A2 ALIGNMENT on EXTENSION (levels 1-11)", lambda tr: (EXT, sel_cfg(W, tr, EXT, ALIGN_CFGS)))
    stack_src = "M1 PER-GROUP share + choice of forecast" if len(cands) > 1 else "T3 PER-GROUP share (Chronos-2)"

    def rule_stack(tr):
        sp = sel_groups(W, tr, GROUPS, cands)
        return sp, sel_cfg(W, tr, sp, ALIGN_CFGS)
    loo("S1 STACK: group blend + alignment", rule_stack)
    say(f"  stack rule done {time.time() - t0:.0f}s")

    fixed_rows = []
    for c in cands:
        nm = f"(fixed 0.20, levels 1-9) forecast = {c}"
        fixed(nm, tuple([(c, I20)] * 9 + [("C2", 0)] * 2))
        fixed_rows.append(nm)
    orc = {}
    for n in wins_all:
        sp = sel_groups(W, [n], [[i] for i in range(11)], cands)
        orc[n] = cost_cfg(W[n], sp, None)

    # ------------------------------------------------------------ write
    L = ["OVERNIGHT RESULTS: lower WRMSSE than the 0.20 headline? (lower is better)",
         f"Windows with data: {', '.join(wins_all)}. Honest test: each window is scored with settings chosen WITHOUT it",
         "(five pre-final windows use the other four; the final window uses the five). mean5 = mean of the five pre-final windows.",
         f"Candidate direct forecasts for levels 1-9: {', '.join(cands)}",
         "Reference scores on the final window: official winner 0.5204, our reproduction 0.5228, preprint stretch 0.4989.", ""]
    bias = ", ".join(f"{n} {W[n].Wagg[0].sum() / W[n].ev.y_agg[0].sum():.3f}" for n in wins_all)
    L += [f"Winner bias (total forecast / total actual): {bias}", ""]
    hdr = f"{'Rule':<46}" + "".join(f"{n:>9}" for n in wins_all) + f"{'mean5':>9}"
    L += ["WRMSSE", hdr]
    order = ["winner alone", "HEADLINE 0.20, levels 1-9", "EXTENSION 0.20, levels 1-11"] + \
            [k for k in res if k[:2] in ("T1", "T2", "T3", "T4", "M1", "H1", "A1", "A2", "S1")]
    for k in order:
        L.append(f"{k:<46}" + "".join(f"{res[k][n]:>9.4f}" for n in wins_all) + f"{np.mean([res[k][n] for n in pre]):>9.4f}")
    L.append(f"{'(reference only) per-level best ON the window':<46}" + "".join(f"{orc[n]:>9.4f}" for n in wins_all)
             + f"{np.mean([orc[n] for n in pre]):>9.4f}   not honest, upper bound")
    L += ["", "FIXED RULE, WHICH DIRECT FORECAST IS BEST AT SHARE 0.20 (no selection, levels 1-9 only)", hdr]
    for k in fixed_rows:
        L.append(f"{k:<46}" + "".join(f"{res[k][n]:>9.4f}" for n in wins_all) + f"{np.mean([res[k][n] for n in pre]):>9.4f}")
    L += ["", "CHANGE VS HEADLINE (negative is better); 'better' counts the five pre-final windows", hdr + f"{'better':>8}{'p':>7}"]
    Hh = res["HEADLINE 0.20, levels 1-9"]
    for k in order[2:] + fixed_rows:
        ch = {n: res[k][n] - Hh[n] for n in wins_all}
        kk = sum(ch[n] < 0 for n in pre)
        L.append(f"{k:<46}" + "".join(f"{ch[n]:>+9.4f}" for n in wins_all) + f"{np.mean([ch[n] for n in pre]):>+9.4f}"
                 f"{kk:>5}/{len(pre)}{sign_test_p(kk, len(pre)):>7.3f}")
    L += ["", "SETTINGS CHOSEN FOR THE FINAL WINDOW (from the five earlier windows). * = levels in the group differ"]
    for k in order[3:]:
        if "final" in desc[k] and desc[k]["final"]:
            L.append(f"  {k}: {desc[k]['final']}")
    if "final" in wins_all:
        L += ["", "LEVEL SCORES ON THE FINAL WINDOW (WRMSSE per level)"]
        show = ["winner alone", "HEADLINE 0.20, levels 1-9", stack_src, "S1 STACK: group blend + alignment"]
        L.append(f"{'Level':<8}" + "".join(f"{s[:22]:>24}" for s in show))
        lvs = {s: levels_cfg(W["final"], *spc[s]["final"]) for s in show}
        for i in range(12):
            L.append(f"{'L' + str(i + 1):<8}" + "".join(f"{lvs[s][i]:>24.4f}" for s in show))
    L += ["", "HOW TO READ",
          "1. A rule is worth adopting only if it beats HEADLINE on the five held-out windows (better 4/5 or 5/5) AND on the final window.",
          "2. The 'reference only' row is the most per-level tuning could ever give; the gap to the honest rows is overfitting.",
          "3. Windows share history and are only four weeks each, so p values are rough. Do not claim to beat 0.5204 unless the",
          "   final-window gain is clearly larger than the earlier bootstrap interval (about 0.01).",
          "4. Alignment makes the item forecasts add up to the blended aggregate, so those rows are coherent by construction.",
          f"Runtime of this analysis {(time.time() - t0) / 60:.1f} min"]
    text = "\n".join(L)
    with open(os.path.join(args.out_dir, out_name), "w") as f:
        f.write(text + "\n")
    print(text, flush=True)
    return text


# ------------------------------------------------------------------ GPU stage
def stage_gpu(args, meta, values, dcols, calendar, prices, cal_full, D, evs):
    from m5_aggregate import build_inputs
    from m5_chronos2 import MockPipeline, quantile_mean_weights, to_np
    gdir = os.path.join(args.out_dir, "gpu_fc")
    os.makedirs(gdir, exist_ok=True)
    if args.mock:
        pipe = MockPipeline()
    else:
        import torch
        from chronos import Chronos2Pipeline
        cuda = torch.cuda.is_available()
        say(f"Loading amazon/chronos-2 on {'cuda' if cuda else 'CPU'}")
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if cuda else "cpu")
    wts = quantile_mean_weights(pipe.quantiles)

    def fc(inputs, bs, cross):
        pr = pipe.predict(inputs, prediction_length=HORIZON, batch_size=bs, cross_learning=cross)
        arr = np.stack([to_np(t)[0] for t in pr])
        return np.clip(np.tensordot(arr, wts, axes=([1], [0])), 0, None).astype(np.float32)

    for idx, name in enumerate(WINS):
        o = D - HORIZON * (6 - idx)
        if o < 400 and not args.selftest:
            continue
        todo = [v for v in ("ctx512", "ctx2048", "grpLevel", "grpAll")
                if not os.path.exists(os.path.join(gdir, f"agg_chronos2_B_{v}_{name}.npy"))]
        if not todo:
            continue
        if name not in evs:
            evs[name] = WRMSSEEvaluator(meta, values[:, : o + HORIZON], dcols[: o + HORIZON], calendar, prices)
        ev = evs[name]
        sizes = ev.sizes[:9]
        for v in todo:
            t0 = time.time()
            try:
                ctx = {"ctx512": 512, "ctx2048": 2048}.get(v, 1024)
                inputs = build_inputs(ev, meta, values, cal_full, dcols[: o + HORIZON], o, ctx, True)
                if v in ("ctx512", "ctx2048"):
                    out = fc(inputs, 64, False)
                elif v == "grpAll":
                    out = fc(inputs, len(inputs), True)
                else:
                    parts, s = [], 0
                    for g in sizes:
                        parts.append(fc(inputs[s:s + g], g, True))
                        s += g
                    out = np.concatenate(parts, axis=0)
                np.save(os.path.join(gdir, f"agg_chronos2_B_{v}_{name}.npy"), out)
                say(f"  GPU {name} {v} done in {time.time() - t0:.0f}s")
            except Exception as e:  # keep going, log and skip
                say(f"  GPU {name} {v} FAILED: {e!r}")
                with open(os.path.join(args.out_dir, "gpu_errors.txt"), "a") as f:
                    f.write(f"{name} {v}\n{traceback.format_exc()}\n")


# ------------------------------------------------------------------ selftest fixtures
def make_fixtures(args, meta, values, dcols, calendar, prices, D):
    from m5_data_and_wrmsse import build_aggregators
    rng = np.random.default_rng(3)
    A = build_aggregators(meta)
    sizes = [a.shape[0] for a in A]
    for d in (args.in_dir, args.fm_dir, args.lvl_dir):
        os.makedirs(d, exist_ok=True)
    for idx, name in enumerate(WINS):
        o = D - HORIZON * (6 - idx)
        y = values[:, o:o + HORIZON]
        np.save(os.path.join(args.in_dir, f"winner_{name}.npy"), y * 0.93 * rng.uniform(0.7, 1.3, y.shape))
        agg = np.vstack([np.asarray(a @ y) for a in A[:9]])
        np.save(os.path.join(args.in_dir, f"agg_chronos2_B_covariates_{name}.npy"), agg * rng.uniform(0.85, 1.15, agg.shape))
        for tag in ("timesfm3", "t0"):
            np.save(os.path.join(args.fm_dir, f"fm_{tag}_{name}.npy"), agg * rng.uniform(0.8, 1.2, agg.shape))
        for n_, li in (("10", 9), ("11", 10)):
            z = np.asarray(A[li] @ y)
            np.save(os.path.join(args.lvl_dir, f"lvl{n_}_chronos2_B_{name}.npy"), z * rng.uniform(0.8, 1.2, z.shape))


# ------------------------------------------------------------------ main
def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--in_dir", default="inputs")
    ap.add_argument("--fm_dir", default="fm_out")
    ap.add_argument("--lvl_dir", default="lvl1011_out")
    ap.add_argument("--out_dir", default="overnight_out")
    ap.add_argument("--stage", default="all", choices=["all", "analysis", "gpu"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        from m5_chronos2 import augment_calendar
        tmp = tempfile.mkdtemp()
        make_synthetic(tmp, n_days=420)
        augment_calendar(tmp, 420)
        args.data_dir, args.mock = tmp, True
        args.in_dir, args.fm_dir, args.lvl_dir = [os.path.join(tmp, x) for x in ("in", "fm", "lvl")]
        args.out_dir = os.path.join(tmp, "out")
    os.makedirs(args.out_dir, exist_ok=True)
    LOG = os.path.join(args.out_dir, "run_log.txt")
    os.environ.setdefault("M5_WINDOW", "final")
    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    dcols = list(dcols)
    cal_full = load_calendar(args.data_dir)
    D = values.shape[1]
    say(f"Series {values.shape[0]}, days {D}")
    if args.selftest:
        make_fixtures(args, meta, values, dcols, calendar, prices, D)
    evs = {}
    if args.stage in ("all", "analysis"):
        try:
            analysis(args, meta, values, dcols, calendar, prices, D, evs, "results_overnight_A.txt", False)
        except Exception:
            say("PART A FAILED: " + traceback.format_exc())
    if args.stage in ("all", "gpu"):
        try:
            stage_gpu(args, meta, values, dcols, calendar, prices, cal_full, D, evs)
        except Exception:
            say("PART B FAILED: " + traceback.format_exc())
    if args.stage in ("all", "analysis") and glob.glob(os.path.join(args.out_dir, "gpu_fc", "*.npy")):
        try:
            analysis(args, meta, values, dcols, calendar, prices, D, evs, "results_overnight_FINAL.txt", True)
        except Exception:
            say("PART C FAILED: " + traceback.format_exc())
    say("ALL DONE")


if __name__ == "__main__":
    main()
