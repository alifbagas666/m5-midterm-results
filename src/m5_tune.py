"""
Tuned blend shares, tested honestly. Uses forecasts that are already saved. No model is run.

Idea: WRMSSE is the plain mean of 12 level scores, and each level score depends only on that level's forecast.
So we score every window once at a grid of shares (0 to 0.60) for each level, store the table, and then every
tuning rule is just a lookup. Nothing is tuned on the window being scored:
  - for each of the five windows before the final one: the share is chosen on the other four of those five;
  - for the final window: the share is chosen on those five windows only, and the final is scored once.

Needs, per window: winner_<w>.npy, agg_chronos2_B_covariates_<w>.npy (in --in_dir),
                   lvl10_chronos2_B_<w>.npy, lvl11_chronos2_B_<w>.npy (in --lvl_dir).
Usage:
  python m5_tune.py --data_dir <M5 data, 1969 days> --in_dir inputs --lvl_dir lvl1011_out --out_dir tune_out
Output: tune_out/results_tune.txt  (plain text) and tune_out/level_table.npz (the cached table, reused on rerun).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5_data_and_wrmsse import HORIZON, WRMSSEEvaluator, load_m5  # noqa: E402
from m5_multiwindow import point_score, sign_test_p, sq_errors  # noqa: E402

ORDER = ["cv5", "cv2", "cv3", "cv4", "public", "final"]
ORIGINS = {"cv5": 1801, "cv2": 1829, "cv3": 1857, "cv4": 1885, "public": 1913, "final": 1941}
GRID = np.round(np.arange(0.0, 0.601, 0.05), 2)
I20 = int(np.where(GRID == 0.20)[0][0])
GROUPS = [("L1-3", [0, 1, 2]), ("L4-6", [3, 4, 5]), ("L7-9", [6, 7, 8]), ("L10-11", [9, 10])]


def build_table(args):
    meta, values, dcols, calendar, prices = load_m5(args.data_dir)
    dcols = list(dcols)
    tab, have = {}, []
    for w in ORDER:
        o = ORIGINS[w]
        paths = [os.path.join(args.in_dir, f"winner_{w}.npy"), os.path.join(args.in_dir, f"agg_chronos2_B_covariates_{w}.npy"),
                 os.path.join(args.lvl_dir, f"lvl10_chronos2_B_{w}.npy"), os.path.join(args.lvl_dir, f"lvl11_chronos2_B_{w}.npy")]
        if values.shape[1] < o + HORIZON or not all(os.path.exists(p) for p in paths):
            print(f"window {w}: skipped (missing files or days)", flush=True)
            continue
        win, d19, d10, d11 = [np.load(p) for p in paths]
        ev = WRMSSEEvaluator(meta, values[:, : o + HORIZON], dcols[: o + HORIZON], calendar, prices)
        off = np.cumsum([0] + ev.sizes[:9])
        T = np.zeros((len(GRID), 12))
        for k, s in enumerate(GRID):
            lv = []
            for i, A in enumerate(ev.A):
                P = np.asarray(A @ win)
                if i < 9:
                    P = (1 - s) * P + s * d19[off[i]:off[i + 1]]
                elif i == 9:
                    P = (1 - s) * P + s * d10
                elif i == 10:
                    P = (1 - s) * P + s * d11
                lv.append(P)
            T[k] = point_score(ev, sq_errors(ev, lv))[1]
            print(f"  {w} share {s:.2f}: WRMSSE (levels 1-11 at this share) {T[k].mean():.4f}", flush=True)
        tab[w] = T
        have.append(w)
    np.savez(os.path.join(args.out_dir, "level_table.npz"), windows=np.array(have), **{w: tab[w] for w in have})
    return tab


def load_table(args):
    p = os.path.join(args.out_dir, "level_table.npz")
    if os.path.exists(p) and not args.rebuild:
        z = np.load(p)
        print("Using cached", p, flush=True)
        return {w: z[w] for w in z["windows"]}
    return build_table(args)


def cost(T, shares):
    """WRMSSE of a window given a share index per level (12 entries; level 12 always index 0)."""
    return float(np.mean([T[shares[l], l] for l in range(12)]))


def fixed(idx_l1_9, idx_l10_11=0):
    return [idx_l1_9] * 9 + [idx_l10_11] * 2 + [0]


def pick_global(tr, tab, with_item):
    best = min(range(len(GRID)), key=lambda k: np.mean([cost(tab[w], fixed(k, k if with_item else 0)) for w in tr]))
    return fixed(best, best if with_item else 0), f"{GRID[best]:.2f}"


def pick_perlevel(tr, tab):
    sh = []
    for l in range(11):
        sh.append(int(np.argmin(np.mean([tab[w][:, l] for w in tr], axis=0))))
    return sh + [0], " ".join(f"{GRID[k]:.2f}" for k in sh)


def pick_group(tr, tab):
    sh = [0] * 12
    for _, ls in GROUPS:
        k = int(np.argmin(np.mean([[tab[w][j, ls].sum() for j in range(len(GRID))] for w in tr], axis=0)))
        for l in ls:
            sh[l] = k
    return sh, " ".join(f"{n}={GRID[sh[ls[0]]]:.2f}" for n, ls in GROUPS)


def pick_shrunk(tr, tab):
    sh, _ = pick_perlevel(tr, tab)
    out = [int(round((sh[l] + I20) / 2)) if l < 11 else 0 for l in range(12)]
    return out, " ".join(f"{GRID[k]:.2f}" for k in out[:11])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="m5_data")
    ap.add_argument("--in_dir", default="inputs")
    ap.add_argument("--lvl_dir", default="lvl1011_out")
    ap.add_argument("--out_dir", default="tune_out")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tab = load_table(args)
    wins = [w for w in ORDER if w in tab]
    pre = [w for w in wins if w != "final"]
    if len(pre) < 3:
        print("Not enough windows.")
        return
    rules = [("GLOBAL share, levels 1-9", lambda tr: pick_global(tr, tab, False)),
             ("GLOBAL share, levels 1-11", lambda tr: pick_global(tr, tab, True)),
             ("PER-LEVEL share, levels 1-11", lambda tr: pick_perlevel(tr, tab)),
             ("PER-GROUP share (L1-3, L4-6, L7-9, L10-11)", lambda tr: pick_group(tr, tab)),
             ("PER-LEVEL shrunk halfway to 0.20", lambda tr: pick_shrunk(tr, tab))]
    fixed_rules = [("winner alone", fixed(0)), ("HEADLINE 0.20, levels 1-9", fixed(I20)),
                   ("EXTENSION 0.20, levels 1-11", fixed(I20, I20))]
    res, chosen = {}, {}
    for n, sh in fixed_rules:
        res[n] = {w: cost(tab[w], sh) for w in wins}
    for n, f in rules:
        res[n] = {}
        for w in wins:
            tr = [x for x in pre if x != w]
            sh, txt = f(tr)
            res[n][w] = cost(tab[w], sh)
            if w == "final" or (w == pre[-1] and "final" not in wins):
                chosen[n] = txt
    orc = {}
    for w in wins:
        sh, _ = pick_perlevel([w], tab)
        orc[w] = cost(tab[w], sh)
    L = ["TUNED BLEND SHARES, HONEST TEST (no model run, saved forecasts only)",
         "WRMSSE, lower is better. Each window is scored with shares chosen WITHOUT looking at that window:",
         "windows before the final one use the other four of those five; the final window uses those five.",
         f"Share grid: {GRID[0]:.2f} to {GRID[-1]:.2f} step 0.05. Windows with data: {', '.join(wins)}.", ""]
    L.append(f"{'Rule':<46}" + "".join(f"{w:>9}" for w in wins) + f"{'mean5':>9}")
    for n in res:
        L.append(f"{n:<46}" + "".join(f"{res[n][w]:>9.4f}" for w in wins) + f"{np.mean([res[n][w] for w in pre]):>9.4f}")
    L.append(f"{'(reference only) per-level best ON the window':<46}" + "".join(f"{orc[w]:>9.4f}" for w in wins)
             + f"{np.mean([orc[w] for w in pre]):>9.4f}   <- not honest, upper bound")
    L += ["", "CHANGE VS HEADLINE (negative is better)"]
    L.append(f"{'Rule':<46}" + "".join(f"{w:>9}" for w in wins) + f"{'mean5':>9}{'better':>8}{'p':>7}")
    H = res["HEADLINE 0.20, levels 1-9"]
    for n in res:
        if n.startswith("winner") or n.startswith("HEADLINE"):
            continue
        ch = {w: res[n][w] - H[w] for w in wins}
        k = sum(ch[w] < 0 for w in pre)
        L.append(f"{n:<46}" + "".join(f"{ch[w]:>+9.4f}" for w in wins) + f"{np.mean([ch[w] for w in pre]):>+9.4f}"
                 f"{k:>5}/{len(pre)}{sign_test_p(k, len(pre)):>7.3f}")
    L += ["", "SHARES CHOSEN FOR THE LAST WINDOW (trained on the windows before it)"]
    for n, t in chosen.items():
        L.append(f"  {n}: {t}")
    L += ["", "HOW TO READ: a rule is only worth using if it beats HEADLINE on the five held-out windows AND on the final window.",
          "If it wins on the five but not the final, treat it as unproven. Windows share history, so p values are rough.",
          "The 'reference only' row shows the most any per-level tuning could ever give; the gap to the honest rows is overfitting."]
    t = "\n".join(L)
    open(os.path.join(args.out_dir, "results_tune.txt"), "w").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
