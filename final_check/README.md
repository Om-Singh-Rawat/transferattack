# Final Baseline Checks

This folder contains the verified, final evaluation summaries for the original baseline attacks (`SI_NI_FGSM`, `MI_FGSM`, `MI_ADMIX_DI_TI`, `TI_FGSM`, and `PGD`).

## How we got here
The original raw cosine similarities file from `results_baseline/subset_raw_similarities_long.csv` was used. This file correctly contains the similarities across all 5 victim models (Facenet512, ArcFace, GhostFaceNet, VGG-Face, and IR152).

We used the summarization script (`build_subset_baselines.py`) on this raw CSV to compute the final, accurate breach rates and impact means.

## How to reproduce
To regenerate the summary files in this folder from the original raw baseline CSV, run the following command from the `transferattack` root directory:

```bash
python scripts/build_subset_baselines.py \
    --raw-long-csv results_baseline/subset_raw_similarities_long.csv \
    --input-csv docs/subset_input_pairs.csv \
    --thresholds-json core/verification_thresholds.json \
    --output-dir final_check
```
