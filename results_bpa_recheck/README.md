# Corrected BPA_CNN Evaluation Results

This directory contains the corrected evaluation of the `BPA_CNN` adversarial images (from `outputs_bpa`) against all 5 victim models (including `IR152`).

## Why this exists
Similar to the baselines, the first time we evaluated `BPA_CNN`, the original evaluation script failed to evaluate against all 5 victims. We ran the fixed `evaluate_subset.py` script on the `BPA_CNN` images to generate a completely new, complete, and correct raw similarity CSV (`subset_raw_similarities_long.csv`).

This verified raw CSV was then used to build the final `BPA_CNN` summaries found in the `final_check_bpa/` folder.
