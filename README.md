# M5 hierarchical forecasting: blending the winner with zero-shot foundation models

Master's project (Yuan Ze University, Industrial Engineering and Management). We take the M5 winner's
six-set LightGBM forecasts and blend in 20 percent of a zero-shot foundation-model forecast
(Chronos-2, direct, with weekday, event and SNAP covariates) at the 154 aggregate series (levels 1 to 9).
Item levels (10 to 12) stay with the winner.

## Main result (WRMSSE, lower is better)

| Window | Winner (our reproduction) | Blend | Change |
|---|---|---|---|
| cv5 d1802-1829 | 0.6730 | 0.6545 | -0.0185 |
| cv2 d1830-1857 | 0.6477 | 0.6379 | -0.0098 |
| cv3 d1858-1885 | 0.6562 | 0.6291 | -0.0271 |
| cv4 d1886-1913 | 0.5167 | 0.5151 | -0.0016 |
| public d1914-1941 | 0.6415 | 0.6245 | -0.0170 |
| final d1942-1969 | 0.5228 | 0.5195 | -0.0033 |

The blend is better on 6 of 6 windows (mean change -0.0129, 95 percent interval -0.0182 to -0.0072,
sign test p = 0.031). On the final window the gain is small and inside the noise (probability the blend is better, 0.73),
so we do **not** claim to beat the M5 winner. The official winner score on that window is 0.5204; our reproduction is 0.5228.
Gains track the winner's under-forecasting (bias), not sales growth. Four other foundation models give the same pattern.

## Read this first

`M5_midterm_results.ipynb` shows the results and re-scores them from the saved forecasts.
`docs/REPORT.md` (and `docs/REPORT.pdf`) explains what we did, why, every result and its interpretation, limits and open work.
`docs/REPORT_FULL_with_code_and_results.pdf` adds all code and raw result files.

## Layout

| Folder | Contents |
|---|---|
| `src/` | All scripts we wrote |
| `forecasts/` | Saved forecasts for every window (winner, Chronos-2 aggregates, four other models, levels 10 and 11) |
| `results/` | Raw result files (see `results/README.md`) |
| `docs/` | Report and figures |
| `scripts/prepare_inputs.py` | Builds `inputs/` from `forecasts/` |
| `data/` | Empty; see `data/README.md` for the M5 data |

## Reproduce the scores

```
pip install -r requirements.txt
python scripts/prepare_inputs.py
python src/m5_data_and_wrmsse.py --selftest
python src/m5_multiwindow.py --data_dir DATA --in_dir inputs --out_dir out_mw --stage score
```

More commands are in section 11 of the report. The winner's own code (Mcompetitions/M5-methods, folder A1) is not included.
Our forecasts from it are in `forecasts/winner_lightgbm/`.

## Status

The overnight tuning and top-down alignment job (`src/m5_overnight.py`) was launched on 21 October; its results are not in this repository yet.
Early single-window experiments E1c and E1d exist as code only.

## License

MIT for our code (see `LICENSE`). The M5 data, the winner's code and the pretrained models have their own licenses.
