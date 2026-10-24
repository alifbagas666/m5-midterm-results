Extra windows for the winner's LightGBM (M5 1st place, A1). Same code and settings as the cv4 run; only the window selector differs.
  cv3: train to d_1857, forecast d_1858..d_1885
  cv2: train to d_1829, forecast d_1830..d_1857
  cv5: train to d_1801, forecast d_1802..d_1829
Run:  python run_extra_windows.py --data_dir <M5 data folder>
Time: about 11 hours per window. Disk: about 45 GB per window. RAM: about 16 GB free.
Output: send_pc3_windows.zip (winner_<window>.npy and the results text for each finished window).
