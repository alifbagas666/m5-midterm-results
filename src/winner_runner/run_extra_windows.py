"""
Runs the winner's six LightGBM sets (M5 1st place, folder A1) on the extra earlier windows, one window after another.
Windows (28 forecast days each): cv3 train to d_1857, forecast d_1858..d_1885 | cv2 d_1830..d_1857 | cv5 d_1802..d_1829.
Nothing is deleted or changed. Finished sets are skipped on a rerun (run_winner_a1.py remembers them).
Before every window it checks free disk space and stops (without touching anything) if it is below --min_free_gb.
At the end (or after every finished window) it copies the small result files into <root>/send and zips them.

usage: python run_extra_windows.py --data_dir <folder with calendar.csv, sell_prices.csv, sales_train_evaluation.csv>
       [--root <this folder>] [--windows cv3,cv2,cv5] [--min_free_gb 55]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile

SETS = ["nr_store", "nr_store_cat", "nr_store_dept", "r_store", "r_store_cat", "r_store_dept"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--windows", default="cv3,cv2,cv5")
    ap.add_argument("--min_free_gb", type=float, default=55.0)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    send = os.path.join(root, "send")
    os.makedirs(send, exist_ok=True)
    status = os.path.join(root, "EXTRA_STATUS.txt")

    def log(m):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        with open(status, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def make_zip():
        zp = os.path.join(root, "send_pc3_windows.zip")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for fn in sorted(os.listdir(send)):
                z.write(os.path.join(send, fn), fn)
            if os.path.exists(status):
                z.write(status, "EXTRA_STATUS.txt")
        log(f"zip written: {zp}")

    for W in [w.strip() for w in args.windows.split(",") if w.strip()]:
        pkg = os.path.join(root, f"pkg_{W}")
        out = os.path.join(pkg, "m5_outputs_A1")
        done = all(os.path.exists(os.path.join(out, f"pred{W}_A1_{s}.npy")) for s in SETS) and \
            os.path.exists(os.path.join(out, f"pred{W}_A1_winner.npy"))
        if not done:
            free = shutil.disk_usage(root).free / 1e9
            log(f"{W}: free disk {free:.0f} GB (need {args.min_free_gb:.0f})")
            if free < args.min_free_gb:
                log(f"STOP before {W}: not enough free disk. Nothing was deleted. Free space or move old model folders yourself, then rerun the same command.")
                make_zip()
                sys.exit(2)
            t0 = time.time()
            log(f"{W}: start (six sets, about 11 hours)")
            with open(os.path.join(root, f"driver_{W}.txt"), "a", encoding="utf-8", errors="replace") as lf:
                rc = subprocess.run([sys.executable, "-u", "run_winner_a1.py", "--data_dir", args.data_dir, "--window", W],
                                    cwd=pkg, stdout=lf, stderr=subprocess.STDOUT).returncode
            log(f"{W}: finished with exit code {rc} after {(time.time() - t0) / 3600:.1f} h")
            if rc != 0:
                log(f"STOP: {W} failed. See driver_{W}.txt and {pkg}\\run_logs. Do not delete anything; send me the last 40 lines of driver_{W}.txt.")
                make_zip()
                sys.exit(1)
        else:
            log(f"{W}: already finished, skipping the run")
        missing = [s for s in SETS if not os.path.exists(os.path.join(out, f"pred{W}_A1_{s}.npy"))]
        if missing:
            log(f"STOP: {W} has no forecast file for {missing}")
            make_zip()
            sys.exit(1)
        shutil.copy2(os.path.join(out, f"pred{W}_A1_winner.npy"), os.path.join(send, f"winner_{W}.npy"))
        rp = os.path.join(out, f"results_winner_A1_{W}.txt")
        if os.path.exists(rp):
            shutil.copy2(rp, os.path.join(send, f"results_winner_A1_{W}.txt"))
        log(f"{W}: result files copied to {send}")
        make_zip()
    log("ALL WINDOWS DONE")


if __name__ == "__main__":
    main()
