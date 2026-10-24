# Results

Raw result files, in the order the work was done.

| Folder | Contents |
|---|---|
| 01_baselines_and_early_models | Naive baselines, own LightGBM (E1, E1b), Chronos-2 and PatchTST alone (public window and final window) |
| 02_winner_reproduction | Winner reproduction results, patch logs and status files for the six windows |
| 03_chronos2_aggregate | Direct Chronos-2 on the 154 aggregate series |
| 04_six_window_blend | `results_multiwindow.txt` covers the FIVE windows cv5, cv2, cv3, cv4, public. The final window numbers are in 05 and 06. |
| 05_foundation_models | Four other foundation models, standalone and blend |
| 06_ablation_bootstrap | Ablation, bootstrap and bias analysis, including the final window |
| 07_reconciliation | OLS reconciliation test |
| 08_shap | SHAP on nr_store |
| 09_levels_10_11 | Direct Chronos-2 at item levels (exploratory) |

Overnight tuning results (PC 2, launched 21 October) were not available when this repository was built.
