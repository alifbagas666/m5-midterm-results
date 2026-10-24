"""
Explainability of the winner's non-recursive per-store LightGBM models (set nr_store) with exact TreeSHAP (LightGBM's own pred_contrib).
READ ONLY on the winner's folder: nothing there is changed, no model is retrained, nothing is deleted. Results go to --out_dir.

What it does, for a few stores:
  1. Reuses the winner's own prepare_data() from its PREDICT script (executed up to the "Predict" section), so the features are exactly the ones used for the forecast.
  2. Takes a random sample of rows from the forecast window, loads the saved model non_recur_model_<store>.bin and computes SHAP values.
     The objective is Tweedie (log link), so SHAP values are contributions to the log of the forecast, summed over features they give log(prediction).
  3. Writes mean |SHAP| per feature (global importance), feature families (lags, rolling means, price, calendar, encodings, ...),
     the sign of the effect (correlation of feature value and SHAP value), and the model's own gain importance for comparison.

usage: python m5_shap_nr.py --root C:\\A1_lean --stores CA_1,TX_1,WI_1 --n_rows 4000 --cv private --out_dir C:\\A1_shap
  --cv private = the original final window d_1942..d_1969 (default). Other keys exist in the winner's own table (public, cv4, ...).
"""
import argparse
import os
import pickle
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def find_predict_script(root):
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f == "2-1. nonrecursive_store_PREDICT.py":
                return os.path.join(dp, f)
    raise SystemExit("2-1. nonrecursive_store_PREDICT.py not found under " + root)


def family(name):
    n = name.lower()
    if "lag" in n and "roll" not in n:
        return "lags"
    if "roll" in n:
        return "rolling means and stds"
    if "enc" in n or n.startswith("enc_"):
        return "target encodings"
    if "price" in n or n.startswith("pr_"):
        return "price"
    if n.startswith("tm_") or "snap" in n or "event" in n or "wday" in n or "month" in n or "year" in n:
        return "calendar, events, SNAP"
    if n.endswith("_id") or n in ("item_id", "dept_id", "cat_id", "state_id", "store_id"):
        return "item and group identity"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="winner's folder that contains '3. code', '5. models' and '2. data'")
    ap.add_argument("--stores", default="CA_1,TX_1,WI_1")
    ap.add_argument("--n_rows", type=int, default=4000)
    ap.add_argument("--cv", default="private")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    script = find_predict_script(args.root)
    root = os.path.dirname(os.path.dirname(os.path.dirname(script)))
    os.environ["A1_DIR"] = root
    os.chdir(root)
    src = open(script, encoding="utf-8").read()
    cut = src.find("########################### Predict")
    if cut < 0:
        raise SystemExit("could not find the Predict section in the winner's script")
    ns = {"__name__": "winner_defs"}
    exec(compile(src[:cut], script, "exec"), ns)  # imports, parameters and prepare_data only; nothing is trained or predicted
    prepare_data, remove_feature, validation = ns["prepare_data"], ns["remove_feature"], ns["validation"]
    lo, hi = validation[args.cv]
    model_dir = ns["model_dir"]
    rng = np.random.default_rng(0)

    per_store, gain_store, sign_store, notes = {}, {}, {}, []
    for store in [s.strip() for s in args.stores.split(",") if s.strip()]:
        t0 = time.time()
        grid_df = prepare_data(store)
        model_var = list(grid_df.columns[~grid_df.columns.isin(remove_feature)])
        vl = (grid_df["d"] > lo) & (grid_df["d"] <= hi)
        X = grid_df.loc[vl, model_var]
        n = min(args.n_rows, len(X))
        X = X.iloc[np.sort(rng.choice(len(X), size=n, replace=False))]
        model = pickle.load(open(os.path.join(model_dir, f"non_recur_model_{store}.bin"), "rb"))
        names = model.feature_name()
        assert names == model_var or set(names) == set(model_var), "feature names of the saved model differ from the prepared data"
        X = X[names]
        contrib = model.predict(X, pred_contrib=True)  # (n, n_features + 1), last column is the base value
        sv = np.asarray(contrib)[:, :-1]
        base = float(np.asarray(contrib)[:, -1].mean())
        per_store[store] = pd.Series(np.abs(sv).mean(axis=0), index=names)
        gain_store[store] = pd.Series(model.feature_importance("gain"), index=names)
        Xv = X.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        sg = {}
        for j, nm in enumerate(names):
            col = Xv[:, j]
            ok = np.isfinite(col)
            sg[nm] = float(np.corrcoef(col[ok], sv[ok, j])[0, 1]) if ok.sum() > 50 and col[ok].std() > 0 and sv[ok, j].std() > 0 else np.nan
        sign_store[store] = pd.Series(sg)
        notes.append(f"{store}: {n} rows sampled from d_{lo + 1}..d_{hi}, {len(names)} features, base value (log scale) {base:.3f}, mean total |SHAP| {np.abs(sv).sum(axis=1).mean():.3f}, {(time.time() - t0) / 60:.1f} min")
        print(notes[-1], flush=True)
        del grid_df, X, contrib, sv, model

    df = pd.DataFrame(per_store)
    df["mean_abs_shap"] = df.mean(axis=1)
    df["share_percent"] = 100 * df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    df["gain_share_percent"] = 100 * pd.DataFrame(gain_store).apply(lambda c: c / c.sum()).mean(axis=1)
    df["effect_sign_corr"] = pd.DataFrame(sign_store).mean(axis=1)
    df["family"] = [family(i) for i in df.index]
    df = df.sort_values("mean_abs_shap", ascending=False)
    df.to_csv(os.path.join(args.out_dir, "shap_nr_store_features.csv"))

    L = ["SHAP EXPLAINABILITY OF THE WINNER'S NON-RECURSIVE PER-STORE LIGHTGBM MODELS (set nr_store)",
         f"Window key '{args.cv}' (days d_{lo + 1}..d_{hi}). Exact TreeSHAP from LightGBM (pred_contrib). Values are on the log scale of the forecast (Tweedie, log link).",
         "Mean |SHAP| = the average size of a feature's push on the log forecast. Share = that feature's part of the total. 'Effect sign' = correlation of the feature value with its SHAP value "
         "(positive: higher value raises the forecast; this is a rough direction, not a causal effect).", ""] + notes + [""]
    L.append("FEATURE FAMILIES (share of total mean |SHAP|)")
    fam = df.groupby("family")["share_percent"].sum().sort_values(ascending=False)
    for k, v in fam.items():
        L.append(f"  {k:<28}{v:>7.1f}%")
    L += ["", "TOP 25 FEATURES (average over the stores)", f"  {'Feature':<34}{'Family':<26}{'Share %':>8}{'Gain %':>8}{'Effect sign':>13}"]
    for nm, r in df.head(25).iterrows():
        L.append(f"  {nm:<34}{r['family']:<26}{r['share_percent']:>8.1f}{r['gain_share_percent']:>8.1f}{r['effect_sign_corr']:>+13.2f}")
    L += ["", "TOP 10 PER STORE (mean |SHAP|)"]
    for store in per_store:
        top = df[store].sort_values(ascending=False).head(10)
        L.append(f"  {store}: " + ", ".join(f"{k} ({v:.3f})" for k, v in top.items()))
    L += ["", "Limits: only the nr_store set (one of six); a sample of rows; SHAP explains the model, not the world. Recursive sets are not covered."]
    text = "\n".join(L)
    open(os.path.join(args.out_dir, "results_shap.txt"), "w", encoding="utf-8").write(text + "\n")
    print("\n" + text)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = df.head(20).iloc[::-1]
        plt.figure(figsize=(7, 6))
        plt.barh(top.index, top["mean_abs_shap"])
        plt.xlabel("mean |SHAP| (log scale of the forecast)")
        plt.title("nr_store: top 20 features")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "shap_nr_store_top20.png"), dpi=150)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", repr(e)[:100])


if __name__ == "__main__":
    main()
