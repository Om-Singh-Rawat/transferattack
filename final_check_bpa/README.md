# Final BPA_CNN Checks

This folder contains the verified, final evaluation summaries for the newly proposed `BPA_CNN` attack.

## How we got here
Because the original evaluation script had a bug that caused it to prematurely fail and skip certain victim models (like IR152), we re-ran the fixed evaluation script (`evaluate_subset.py`) on the `BPA_CNN` adversarial images to generate a completely new, correct `subset_raw_similarities_long.csv` (saved in `results_bpa_recheck/`).

We then used the summarization script (`build_subset_baselines.py`) on this corrected raw CSV to compute the final, accurate breach rate and impact mean for `BPA_CNN`.

## How to reproduce
To regenerate the summary files in this folder from the corrected raw BPA CSV, run the following command from the `transferattack` root directory:

```bash
python scripts/build_subset_baselines.py \
    --raw-long-csv results_bpa_recheck/subset_raw_similarities_long.csv \
    --input-csv docs/subset_input_pairs.csv \
    --thresholds-json core/verification_thresholds.json \
    --output-dir final_check_bpa
```
