import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("M5_WINDOW", "public")
from m5_data_and_wrmsse import HORIZON, WRMSSEEvaluator, load_m5
from m5_multiwindow import mix_levels, sq_errors, point_score, ORIGINS
meta, values, dcols, calendar, prices = load_m5(sys.argv[1]); dcols = list(dcols)
SH = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
tab = {}
for win, o in ORIGINS.items():
    if o + HORIZON > values.shape[1]: continue
    ev = WRMSSEEvaluator(meta, values[:, :o + HORIZON], dcols[:o + HORIZON], calendar, prices)
    w = np.load(f"inputs_all/winner_{win}.npy").astype(float); d = np.load(f"inputs_all/agg_chronos2_B_covariates_{win}.npy").astype(float)
    tab[win] = np.array([point_score(ev, sq_errors(ev, mix_levels(ev, w, d, s)))[0] for s in SH])
wins = list(tab)
print("Change vs winner alone (WRMSSE) by share"); print(f"{'window':<8}" + "".join(f"{s:>8.2f}" for s in SH[1:]) + "   best")
for k in wins:
    c = tab[k] - tab[k][0]; print(f"{k:<8}" + "".join(f"{x:>+8.4f}" for x in c[1:]) + f"   {SH[int(np.argmin(c))]:.2f}")
print("\nLeave-one-window-out: pick the share that minimises the mean change on the OTHER windows, apply it to the held-out one")
for k in wins:
    oth = [j for j in wins if j != k]
    m = np.mean([tab[j] - tab[j][0] for j in oth], axis=0); i = int(np.argmin(m))
    fixed = SH.index(0.20)
    print(f"held-out {k:<7} learned share {SH[i]:.2f}: change {tab[k][i] - tab[k][0]:+.4f} | fixed 0.20: change {tab[k][fixed] - tab[k][0]:+.4f}")
m = np.mean([tab[j] - tab[j][0] for j in wins], axis=0)
print("\nPooled mean change over the", len(wins), "windows by share:", " ".join(f"{s:.2f}:{x:+.4f}" for s, x in zip(SH[1:], m[1:])))
