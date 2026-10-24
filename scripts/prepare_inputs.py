"""Build inputs/ (the folder the scoring scripts expect) from the saved forecasts in forecasts/.
Copies files only. Run from the repository root:  python scripts/prepare_inputs.py
"""
import shutil, glob, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(root, "inputs"); os.makedirs(out, exist_ok=True)
n = 0
for pat in ["forecasts/winner_lightgbm/winner_*.npy", "forecasts/chronos2_aggregate/agg_chronos2_B_covariates_*.npy"]:
    for f in glob.glob(os.path.join(root, pat)):
        shutil.copy(f, os.path.join(out, os.path.basename(f))); n += 1
print(f"copied {n} files to {out}")
