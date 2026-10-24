import os, sys, time
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("M5_WINDOW", "public")
from m5_data_and_wrmsse import HORIZON, WRMSSEEvaluator, load_m5
data_dir = sys.argv[1]
ORIGIN = {"cv4": 1885, "public": 1913}
meta, values, dcols, calendar, prices = load_m5(data_dir)
dcols = list(dcols)
SH = 0.20
for win, o in ORIGIN.items():
    end = o + HORIZON
    ev = WRMSSEEvaluator(meta, values[:, :end], dcols[:end], calendar, prices)
    w = np.load(f"inputs/winner_{win}.npy").astype(np.float64)
    d = np.load(f"inputs/agg_chronos2_B_covariates_{win}.npy").astype(np.float64)
    off = np.cumsum([0] + ev.sizes[:9])
    S = sp.vstack([sp.csr_matrix(A) for A in ev.A]).tocsr()
    base = []
    for i, A in enumerate(ev.A):
        P = np.asarray(A @ w)
        if i < 9:
            P = (1 - SH) * P + SH * d[off[i]:off[i + 1]]
        base.append(P)
    Y = np.vstack(base)
    def lev_score(b):
        return ev.score(np.clip(b, 0, None))
    def levels_from_bottom(b):
        return ev.score(np.clip(b, 0, None))
    # reference: blended levels (incoherent) from the earlier script
    from itertools import chain
    def score_levels(levels):
        out = []
        for (P, sc, wt, Yt) in zip(levels, ev.scale, ev.weight, ev.y_agg):
            v = wt > 0
            r = np.zeros(len(sc)); r[v] = np.sqrt(np.mean((Yt[v] - P[v]) ** 2, axis=1) / sc[v])
            out.append(float(np.sum(wt * r)))
        return float(np.mean(out)), out
    print(f"\n{win}: winner alone {ev.score(w)[0]:.4f} | blend incoherent {score_levels(base)[0]:.4f}")
    size = np.asarray(S.sum(axis=1)).ravel()
    for name, wts in [("OLS", np.ones(S.shape[0])), ("structural", 1.0 / size)]:
        Wd = sp.diags(wts)
        ST = S.T.tocsr()
        M = LinearOperator((S.shape[1], S.shape[1]), matvec=lambda x, S=S, ST=ST, wts=wts: ST @ (wts * (S @ x)), dtype=np.float64)
        rhs = S.T @ (wts[:, None] * Y)
        t0 = time.time()
        b = np.zeros((S.shape[1], HORIZON))
        # warm start from the winner's bottom forecast
        for j in range(HORIZON):
            x, info = cg(M, rhs[:, j], x0=w[:, j], rtol=1e-8, maxiter=500)
            b[:, j] = x
        rec = ev.score(np.clip(b, 0, None))
        print(f"  {name:<11} reconciled {rec[0]:.4f} (change vs blend {rec[0] - score_levels(base)[0]:+.4f}, vs winner {rec[0] - ev.score(w)[0]:+.4f}) [{time.time() - t0:.0f}s]")
        print("   levels:", " ".join(f"{x:.3f}" for x in rec[1]))
