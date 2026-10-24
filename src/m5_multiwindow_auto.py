"""
One command for PC 2, then leave it running.
  1. Makes the Chronos-2 direct forecasts for the new windows (stage forecast of m5_multiwindow.py).
  2. Waits for the PC 3 zip (send_pc3_windows.zip) to be put into the folder <root>\\drop.
     It starts scoring as soon as winner_cv3, winner_cv2 and winner_cv5 are all inside the zip.
     To score with fewer windows (for example only cv3 finished), create an empty file RUN_NOW.txt in the drop folder.
  3. Scores all windows, checks that the earlier numbers reproduce, and writes send_pc2_multiwindow.zip.
It only reads, runs and writes new files. It deletes nothing and never overwrites an input file.

usage: python m5_multiwindow_auto.py --data_dir <M5 data, 1969 days> [--root <this folder>] [--poll 60] [--mock]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

NEED = ["winner_cv3.npy", "winner_cv2.npy", "winner_cv5.npy"]
EXPECT = {"winner": {"cv4": 0.5167, "public": 0.6415, "final": 0.5228}, "blend": {"cv4": 0.5151, "public": 0.6245, "final": 0.5195}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    inp, outd, drop = [os.path.join(root, x) for x in ("inputs", "multiwindow_out", "drop")]
    for d in (inp, outd, drop):
        os.makedirs(d, exist_ok=True)
    status = os.path.join(root, "AUTO_STATUS.txt")

    def log(m):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
        print(line, flush=True)
        open(status, "a", encoding="utf-8").write(line + "\n")

    base = [sys.executable, "-u", os.path.join(root, "m5_multiwindow.py"), "--data_dir", args.data_dir, "--in_dir", inp, "--out_dir", outd]
    if args.mock:
        base.append("--mock")
    log("step 1: Chronos-2 direct forecasts for the new windows")
    rc = subprocess.run(base + ["--stage", "forecast"], cwd=root).returncode
    if rc != 0:
        log(f"STOP: forecast stage failed (exit {rc}). Send me the error text. Nothing was deleted.")
        sys.exit(1)
    zf = os.path.join(root, "send_pc2_forecasts.zip")
    if not os.path.exists(zf):
        with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
            for w in ("cv3", "cv2", "cv5"):
                p = os.path.join(outd, f"agg_chronos2_B_covariates_{w}.npy")
                if os.path.exists(p):
                    z.write(p, os.path.basename(p))
    log("step 1 done. Waiting for the PC 3 zip in: " + drop + " (put send_pc3_windows.zip there; RUN_NOW.txt there means score with what exists)")

    handled = set()
    while True:
        run_now = os.path.exists(os.path.join(drop, "RUN_NOW.txt"))
        for fn in sorted(os.listdir(drop)):
            if fn.lower().endswith(".zip") and fn not in handled:
                p = os.path.join(drop, fn)
                try:
                    with zipfile.ZipFile(p) as z:
                        names = z.namelist()
                        got = [n for n in names if os.path.basename(n).startswith("winner_") and n.endswith(".npy")]
                        if not got:
                            continue
                        ex = os.path.join(root, "from_pc3_" + os.path.splitext(fn)[0])
                        os.makedirs(ex, exist_ok=True)
                        z.extractall(ex)
                except zipfile.BadZipFile:
                    continue  # still being copied
                for r, _, fs in os.walk(ex):
                    for f in fs:
                        if f.startswith("winner_") and f.endswith(".npy") and not os.path.exists(os.path.join(inp, f)):
                            shutil.copy2(os.path.join(r, f), os.path.join(inp, f))
                handled.add(fn)
                log(f"read {fn}: winner files now in inputs: " + ", ".join(sorted(f for f in os.listdir(inp) if f.startswith("winner_"))))
        have = [n for n in NEED if os.path.exists(os.path.join(inp, n))]
        if len(have) == len(NEED) or (run_now and have):
            break
        time.sleep(args.poll)

    for w in ("cv3", "cv2", "cv5"):  # copy the new direct forecasts next to the winner files (the scorer also reads multiwindow_out)
        p = os.path.join(outd, f"agg_chronos2_B_covariates_{w}.npy")
        q = os.path.join(inp, f"agg_chronos2_B_covariates_{w}.npy")
        if os.path.exists(p) and not os.path.exists(q):
            shutil.copy2(p, q)
    log("step 3: scoring all windows with the fixed rule")
    rc = subprocess.run(base + ["--stage", "score"], cwd=root).returncode
    if rc != 0:
        log(f"STOP: scoring failed (exit {rc}). Send me the error text.")
        sys.exit(1)
    res = open(os.path.join(outd, "results_multiwindow.txt"), encoding="utf-8").read()
    checks = []
    for m in re.finditer(r"WINDOW (\w+) .*?\n.*?\nwinner alone\s+([\d.]+).*?\nblend \(headline\)\s+([\d.]+)", res):
        win, wa, bl = m.group(1), float(m.group(2)), float(m.group(3))
        if win in EXPECT["winner"]:
            ok = abs(wa - EXPECT["winner"][win]) <= 0.0002 and abs(bl - EXPECT["blend"][win]) <= 0.0002
            checks.append(f"  {win}: winner {wa:.4f} (expected {EXPECT['winner'][win]}), blend {bl:.4f} (expected {EXPECT['blend'][win]}) -> {'OK' if ok else 'DIFFERS'}")
    log("reproduction check of earlier numbers:\n" + ("\n".join(checks) if checks else "  no earlier window was scored (data folder too short?)"))
    zp = os.path.join(root, "send_pc2_multiwindow.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(outd, "results_multiwindow.txt"), "results_multiwindow_PC2.txt")
        z.write(status, "AUTO_STATUS.txt")
    log(f"ALL DONE. Send me this file: {zp}")


if __name__ == "__main__":
    main()
