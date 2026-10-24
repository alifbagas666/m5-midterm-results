Multi-window check of the blend (winner six-set LightGBM + direct Chronos-2 at levels 1 to 9).
Part 1: python m5_multiwindow.py --data_dir <M5 data, 1969 days> --in_dir inputs --out_dir multiwindow_out --stage forecast
Part 2 (after the PC 3 winner files are in inputs): the same command with --stage score
Output: multiwindow_out/results_multiwindow.txt
Rule fixed in advance: share 0.20, direct Chronos-2 with covariates, mean point, levels 1 to 9. OLS reconciliation is a labelled secondary variant.
