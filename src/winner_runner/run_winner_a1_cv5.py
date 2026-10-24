#!/usr/bin/env python3
"""
Run the 1st place M5 Accuracy solution (YeonJun In, M5-methods folder A1) on the competition's own final window, then score it with our evaluator.

What this does
  The winner's own scripts are run in order (preprocessing, then for each of six model sets: train and predict),
  with only small compatibility fixes (this package keeps the winner's own final window: train to d_1941,
  forecast d_1942 to d_1969, scored on the released actuals in sales_test_evaluation.csv):
    - small compatibility fixes for current pandas and LightGBM (see PATCH_LOG.txt); no model setting is touched
  The six sets are: non-recursive and recursive models trained per store (10 models each), per store x category
  (30) and per store x department (70). The winner averaged all six with equal weights.
  Afterwards every finished set and the equal-weight average of the finished sets is scored with the same WRMSSE
  evaluator as the rest of the project.

Time and memory (honest expectations)
  Needs about 16 GB RAM free and about 15 GB disk. Preprocessing takes tens of minutes. Training uses 3000 trees per
  model on CPU. The non-recursive sets are the cheaper ones; the recursive sets use much larger trees and are the
  slowest. The full job of 220 models can take days on an ordinary CPU, so use --sets to run it in stages.
  Finished sets are remembered; a rerun skips them.

Usage
  python run_winner_a1.py --sets nr_store            # cheapest start: non-recursive, one model per store
  python run_winner_a1.py --sets nr_store,nr_store_cat,nr_store_dept
  python run_winner_a1.py                            # all six sets, in order of cost
  python run_winner_a1.py --score_only               # rescore whatever has finished
  Default window is the original final one (d_1942 to d_1969); this needs sales_test_evaluation.csv next to the other
  data files. For d_1914 to d_1941 build the package in public mode and add --window public.

Needs the three M5 files calendar.csv, sell_prices.csv, sales_train_evaluation.csv and sample_submission.csv
(found automatically under --data_dir). Packages: pandas, numpy, scipy, lightgbm, psutil, scikit-learn, tqdm.

Outputs (in --out_dir, default m5_outputs_A1)
  pred_A1_<set>.npy        forecast of one set, shape (30490, 28), same row order as sales_train_evaluation.csv
  pred_A1_winner.npy       equal-weight average of the finished sets
  results_winner_A1.txt    plain-text log with scores, bias and the 12-level table
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SETS = {
    "nr_store": ("2-1. nonrecursive_store", "submission_kaggle_nonrecursive_store.csv"),
    "nr_store_cat": ("2-2. nonrecursive_store_cat", "submission_kaggle_nonrecursive_store_cat.csv"),
    "nr_store_dept": ("2-3. nonrecursive_store_dept", "submission_kaggle_nonrecursive_store_dept.csv"),
    "r_store": ("1-1. recursive_store", "submission_kaggle_recursive_store.csv"),
    "r_store_cat": ("1-2. recursive_store_cat", "submission_kaggle_recursive_store_cat.csv"),
    "r_store_dept": ("1-3. recursive_store_dept", "submission_kaggle_recursive_store_dept.csv"),
}
RAW_FILES = ["sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv", "sample_submission.csv"]


def find_file(root, name):
    for dirpath, _, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    return None


def run_script(path, cwd, env, log_path):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        rc = subprocess.run([sys.executable, "-u", path], cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    mins = (time.time() - t0) / 60
    return rc, mins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="folder holding the M5 csv files (searched recursively)")
    ap.add_argument("--work_dir", default=HERE, help="where the winner's folder structure is created")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--sets", default="all", help="comma list of: " + ",".join(SETS))
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--window", default="cv5", choices=["final", "public", "cv5"],
                    help="final = the winner's own final window d_1942..d_1969 (needs sales_test_evaluation.csv to score); "
                         "public = d_1914..d_1941 (package built in public mode); cv5 = d_1802..d_1829 (package built in cv5 mode)")
    args = ap.parse_args()

    work = os.path.abspath(args.work_dir)
    out_dir = args.out_dir or os.path.join(work, "m5_outputs_A1")
    os.makedirs(out_dir, exist_ok=True)
    # file names: final window -> pred_A1_*.npy / results_winner_A1.txt ; public window (validation forecasts) -> predval_A1_*.npy / results_winner_A1_public.txt
    PREFIX = {"final": "pred", "public": "predval", "cv5": "predcv5"}[args.window]
    RESULTS_NAME = {"final": "results_winner_A1.txt", "public": "results_winner_A1_public.txt", "cv5": "results_winner_A1_cv5.txt"}[args.window]
    chosen = list(SETS) if args.sets == "all" else [s.strip() for s in args.sets.split(",") if s.strip()]
    for s in chosen:
        if s not in SETS:
            raise SystemExit(f"Unknown set {s}. Choose from {list(SETS)}")

    # ---- packages
    missing = []
    for mod in ["pandas", "numpy", "scipy", "lightgbm", "psutil", "sklearn", "tqdm"]:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append({"sklearn": "scikit-learn"}.get(mod, mod))
    if missing:
        raise SystemExit("Missing packages: " + ", ".join(missing) + "\nInstall with: pip install " + " ".join(missing))

    # ---- data
    data_dir = args.data_dir
    if data_dir is None:
        for cand in [os.path.join(HERE, "m5_data"), os.path.join(HERE, "..", "m5_data"), os.path.join(os.getcwd(), "m5_data")]:
            if os.path.isdir(cand):
                data_dir = cand
                break
    if data_dir is None:
        raise SystemExit("Could not find the M5 data. Pass --data_dir <folder with the M5 csv files>.")
    raw = os.path.join(work, "2. data")
    for d in ["2. data/processed", "4. logs", "5. models", "6. submissions/before_ensemble", "run_logs"]:
        os.makedirs(os.path.join(work, d), exist_ok=True)
    for f in RAW_FILES:
        dst = os.path.join(raw, f)
        if not os.path.exists(dst):
            src = find_file(data_dir, f)
            if src is None and f == "sample_submission.csv":
                # not in some copies of the data: rebuild it (ids of the 30,490 series for validation, then evaluation; F1..F28 = 0)
                import pandas as pd
                ids = pd.read_csv(os.path.join(raw, "sales_train_evaluation.csv"), usecols=["id"])["id"]
                ids = ids.str.replace("_evaluation", "_validation")
                ss = pd.DataFrame({"id": list(ids) + [i.replace("_validation", "_evaluation") for i in ids]})
                for k in range(1, 29):
                    ss[f"F{k}"] = 0
                ss.to_csv(dst, index=False)
                print("sample_submission.csv not found, rebuilt it from the sales ids", flush=True)
                continue
            if src is None:
                raise SystemExit(f"{f} not found under {data_dir}")
            print(f"Copying {f} ...", flush=True)
            shutil.copy(src, dst)

    env = dict(os.environ, A1_DIR=work)
    code = os.path.join(work, "3. code")
    t_start = time.time()
    status = []

    if not args.score_only:
        marker = os.path.join(work, "run_logs", "done_preprocessing")
        if not os.path.exists(marker):
            print("Preprocessing (tens of minutes, needs lots of RAM) ...", flush=True)
            rc, mins = run_script(os.path.join(code, "1. preprocessing", "1. preprocessing.py"), work, env,
                                  os.path.join(work, "run_logs", "preprocessing.log"))
            if rc != 0:
                raise SystemExit(f"Preprocessing failed (exit {rc}). See run_logs/preprocessing.log")
            open(marker, "w").write("ok\n")
            status.append(f"preprocessing: {mins:.1f} min")
        else:
            status.append("preprocessing: already done")

        for s in chosen:
            stem, csv = SETS[s]
            csv_path = os.path.join(work, "6. submissions", "before_ensemble", csv)
            if os.path.exists(csv_path):
                status.append(f"{s}: already done")
                print(f"{s}: already done, skipping", flush=True)
                continue
            for kind in ("train", "predict"):
                sub = "2. train" if kind == "train" else "3. predict"
                fname = f"{stem}_{kind.upper()}.py"
                print(f"{s}: {kind} ({fname}) ...", flush=True)
                rc, mins = run_script(os.path.join(code, sub, fname), work, env,
                                      os.path.join(work, "run_logs", f"{s}_{kind}.log"))
                status.append(f"{s} {kind}: {mins:.1f} min (exit {rc})")
                print(f"  {s} {kind} finished in {mins:.1f} min (exit {rc})", flush=True)
                if rc != 0:
                    raise SystemExit(f"{s} {kind} failed. See run_logs/{s}_{kind}.log")
            if not os.path.exists(csv_path):
                raise SystemExit(f"{s}: expected output not found: {csv_path}")

    # ---- scoring
    import numpy as np
    import pandas as pd
    from m5_data_and_wrmsse import WRMSSEEvaluator, format_table, load_m5, seasonal_naive_forecast

    meta, values, dcols, calendar, prices = load_m5(raw)
    if args.window == "final":
        tpath = None
        for base in [data_dir, work, HERE]:
            try:
                tpath = find_file(base, "sales_test_evaluation.csv")
            except Exception:
                tpath = None
            if tpath:
                break
        if tpath is None:
            raise SystemExit("Final window needs sales_test_evaluation.csv (actual sales d_1942..d_1969) under --data_dir.")
        tdf = pd.read_csv(tpath)
        tcols = sorted([c for c in tdf.columns if c.startswith("d_")], key=lambda c: int(c[2:]))
        assert len(tcols) == 28, f"expected 28 test days, found {len(tcols)}"
        key = meta["item_id"] + "|" + meta["store_id"]
        tdf = tdf.set_index(tdf["item_id"] + "|" + tdf["store_id"])
        tv = tdf.reindex(key)[tcols].to_numpy(dtype=np.float64)
        assert not np.isnan(tv).any(), "test file does not cover every series"
        values = np.concatenate([values, tv], axis=1)
        dcols = list(dcols) + tcols
    if args.window == "cv5":
        # fold ending d_1829: holdout d_1802..d_1829, training history up to d_1801 (the M5 protocol on that fold)
        values = values[:, :1829]
        dcols = list(dcols)[:1829]
    ev = WRMSSEEvaluator(meta, values, dcols, calendar, prices)
    total_actual = ev.y_test.sum()
    preds, res = {}, {}
    for s in SETS:
        csv_path = os.path.join(work, "6. submissions", "before_ensemble", SETS[s][1])
        if not os.path.exists(csv_path):
            continue
        sub = pd.read_csv(csv_path).drop_duplicates("id").set_index("id")
        fcols = [f"F{i}" for i in range(1, 29)]
        arr = sub.reindex(meta["id"])[fcols].fillna(0).to_numpy(dtype=np.float64)
        arr = np.clip(arr, 0, None)
        preds[s] = arr
        res[s] = ev.score(arr)
        np.save(os.path.join(out_dir, f"{PREFIX}_A1_{s}.npy"), arr.astype(np.float32))
    if not preds:
        raise SystemExit("No finished set to score yet.")
    avg = np.mean(list(preds.values()), axis=0)
    res["average"] = ev.score(avg)
    np.save(os.path.join(out_dir, f"{PREFIX}_A1_winner.npy"), avg.astype(np.float32))

    lines = [f"{'Set':<16}{'WRMSSE':>9}{'Bias':>8}"]
    for s in list(preds) + ["average"]:
        a = avg if s == "average" else preds[s]
        lines.append(f"{s:<16}{res[s][0]:>9.4f}{a.sum() / total_actual:>8.3f}")
    pub = args.window == "public"
    table = {"SeasonalNaive": ev.score(seasonal_naive_forecast(ev.X_train))}
    for s in list(preds)[:3]:
        table[s] = res[s]
    table["average"] = res["average"]
    n_done = len(preds)
    report = [
        "WINNER'S CODE (M5 1st place, folder A1) RUN ON " + {"final": "THE ORIGINAL FINAL WINDOW", "public": "OUR PUBLIC WINDOW", "cv5": "AN EARLIER VALIDATION FOLD (cv5)"}[args.window],
        ("Trained on days up to d_1941, forecast d_1942 to d_1969, scored on the released actuals (the competition's final window)."
         if args.window == "final" else
         ("Trained on days up to d_1913, forecast d_1914 to d_1941 (the winner's own 'public' setting)."
          if args.window == "public" else
          "Trained on days up to d_1801, forecast d_1802 to d_1829 (the winner's own 'cv5' fold; target encodings rebuilt on days up to d_1801).")),
        "Settings of the winner's models are unchanged. Compatibility patches are listed in PATCH_LOG.txt.",
        f"Sets finished: {n_done} of 6 ({', '.join(preds)}). The winner averaged all six with equal weights;",
        "the 'average' row is the equal-weight average of the sets finished so far.",
        "Bias = total forecast / total actual (1.000 is unbiased).",
        "",
        "RESULTS (WRMSSE on the holdout, lower is better)",
        *lines,
        "",
        "WRMSSE BY LEVEL (first three finished sets and the average)",
        format_table(ev, table),
        "",
        "REFERENCE VALUES (published, window to be confirmed)",
        "  Original M5 winner (final window d_1942 to d_1969): 0.520",
        "  Yang et al. Kaggle Top-1 reproduction: 0.5230 | Yang et al. LightGBM + PatchTST: 0.4989",
        "  Our numbers on d_1914 to d_1941 (a different window): E1b L3 0.5422 | E1d non-recursive per-store port 0.6959 raw, 0.5992 calibrated",
        "",
        "STEPS: " + " | ".join(status),
        f"Runtime of this invocation: {(time.time() - t_start) / 60:.1f} min",
    ]
    text = "\n".join(report)
    with open(os.path.join(out_dir, RESULTS_NAME), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved: {os.path.join(out_dir, RESULTS_NAME)}")


if __name__ == "__main__":
    main()
