"""Builds the winner's package (M5 1st place, folder A1) for extra earlier windows from the finished cv4 package.
Only the window selector changes (which days are train and which are forecast). No model setting, no training step.
Windows (28 forecast days each):  cv3 = train to d_1857, forecast d_1858..d_1885   (winner's own 'cv3')
                                  cv2 = train to d_1829, forecast d_1830..d_1857   (winner's own 'cv2')
                                  cv5 = train to d_1801, forecast d_1802..d_1829   (added by us, same rule)
usage: python make_extra_pkgs.py BASE_CV4_DIR OUT_DIR
"""
import os, re, shutil, sys
base, out = sys.argv[1], sys.argv[2]
WIN = {"cv3": 3, "cv2": 4, "cv5": 5}  # KKK: origin = 1941 - 28*KKK
for W, K in WIN.items():
    origin = 1941 - 28 * K
    end = origin + 28
    dst = os.path.join(out, f"pkg_{W}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(base, dst, ignore=shutil.ignore_patterns("2. data", "4. logs", "5. models", "6. submissions", "run_logs", "m5_outputs_A1", "drv.txt", "__pycache__"))
    log = []
    def sub(path, pat, rep, exp=None, regex=True):
        t = open(path, encoding="utf-8").read()
        t2, n = (re.subn(pat, rep, t, flags=re.M) if regex else (t.replace(pat, rep), t.count(pat)))
        if exp is not None and n != exp:
            raise SystemExit(f"{path}: {pat!r} expected {exp} got {n}")
        open(path, "w", encoding="utf-8").write(t2)
        log.append(f"{os.path.relpath(path, dst)}: {pat[:40]!r} x{n}")
    code = os.path.join(dst, "3. code")
    for sd in os.listdir(code):
        for f in os.listdir(os.path.join(code, sd)):
            p = os.path.join(code, sd, f)
            if not f.endswith(".py"):
                continue
            if "nonrecursive" in f and ("TRAIN" in f or "PREDICT" in f):
                sub(p, r"^cvs = \['cv4'\].*$", f"cvs = ['{W}']  # train to d_{origin}, forecast d_{origin + 1}..d_{end}", 1)
                if W == "cv5":
                    sub(p, r"^(\s*)'cv4' : \[1885,1913\],", r"\1'cv4' : [1885,1913],\n\1'cv5' : [1801,1829],", 1)
            if "recursive_" in f and "nonrecursive" not in f and ("TRAIN" in f or "PREDICT" in f):
                sub(p, r"^ver, KKK = 'cv4', 2.*$", f"ver, KKK = '{W}', {K}  # END_TRAIN = 1941 - 28*{K} = {origin}", 1)
    sub(os.path.join(code, "1. preprocessing", "1. preprocessing.py"), r"^grid_df\.loc\[grid_df\['d'\]>\(1941-56\), 'sales'\] = np\.nan.*$",
        f"grid_df.loc[grid_df['d']>(1941-{28 * K}), 'sales'] = np.nan  # {W}: target encodings may only use days up to d_{origin}", 1)
    r = os.path.join(dst, "run_winner_a1.py")
    sub(r, "values[:, :1913]", f"values[:, :{end}]", regex=False)
    sub(r, "list(dcols)[:1913]", f"list(dcols)[:{end}]", regex=False)
    sub(r, "d_1886..d_1913, training history up to d_1885", f"d_{origin + 1}..d_{end}, training history up to d_{origin}", regex=False)
    sub(r, "fold ending d_1913", f"fold ending d_{end}", regex=False)
    sub(r, "up to d_1885, forecast d_1886 to d_1913", f"up to d_{origin}, forecast d_{origin + 1} to d_{end}", regex=False)
    sub(r, "cv4 = d_1886..d_1913 (package built in cv4 mode)", f"{W} = d_{origin + 1}..d_{end} (package built in {W} mode)", regex=False)
    sub(r, "target encodings rebuilt on days up to d_1885", f"target encodings rebuilt on days up to d_{origin}", regex=False)
    sub(r, "cv4", W, regex=False)
    t = open(r, encoding="utf-8").read()
    assert "1886" not in t and "1885" not in t.replace(f"{end}", "") or W == "cv3", "stray cv4 numbers left"
    assert "d_1886" not in t and "[:1913]" not in t
    open(os.path.join(dst, "README_" + W.upper() + ".txt"), "w").write(
        f"Winner's LightGBM (M5 1st place, A1), earlier fold {W}: train to d_{origin}, forecast d_{origin + 1}..d_{end}.\n"
        f"Built from the cv4 package; only the window selector changes. Target encodings use days up to d_{origin}.\n"
        f"Run: python run_winner_a1.py --data_dir <folder with calendar, sell_prices, sales_train_evaluation> --window {W}\n"
        f"Outputs (m5_outputs_A1): pred{W}_A1_<set>.npy, pred{W}_A1_winner.npy, results_winner_A1_{W}.txt\n")
    open(os.path.join(dst, "PATCH_LOG_" + W.upper() + ".txt"), "w").write("\n".join(log) + "\n")
    print(W, "built:", dst)
