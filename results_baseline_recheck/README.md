# Corrected Baseline Evaluation Results

This directory contains the corrected evaluation of the baseline adversarial images (from `outputs_baseline`) against all 5 victim models (including `IR152`).

## Why this exists
The original evaluation script (`evaluate_subset.py`) had a bug where it crashed and skipped victims like `IR152` if paths were not resolved properly between WSL and Windows. We ran the fixed `evaluate_subset.py` script on the original baseline images to get a complete and correct raw similarity CSV (`subset_raw_similarities_long.csv`).

This verified raw CSV was then used to build the final baseline summaries found in the `final_check/` folder.
