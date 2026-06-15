#!/bin/bash

# ==============================================================================
# Kaggle Pipeline Script for Baseline Transfer Attacks
# 
# Assumes the following directory structure in Kaggle:
# /kaggle/working/
# ├── interns/
# │   ├── dataset_extractedfaces/
# │   ├── ir152.pth
# │   └── ir152.py
# └── transferattack/
#     ├── run_kaggle_baselines.sh (this script)
#     ├── core/
#     ├── experiments/
#     └── scripts/
# ==============================================================================

# Ensure we are inside the transferattack folder
cd /kaggle/working/transferattack


# Setup Environment Variables for deterministic GPU execution
# (Disabled because TF_DETE RMINISTIC_OPS requires hardcoded random seeds in all tf.random ops)
export PYTHONPATH="."
# export TF_DETERMINISTIC_OPS=1
# export TF_CUDNN_DETERMINISTIC=1
# export PYTHONHASHSEED=0


# Configuration
INPUT_CSV="docs/subset_input_pairs.csv"
DATASET_ROOT="../interns/dataset_extractedfaces"
ADV_DIR="outputs_baseline"
RESULTS_DIR="results_baseline_recheck"
ATTACKS="PGD,MI_FGSM,TI_FGSM,SI_NI_FGSM,MI_ADMIX_DI_TI"

# Create output directories
mkdir -p $ADV_DIR
mkdir -p $RESULTS_DIR

echo "======================================"
echo "Phase 1: Generation (Vanilla Baselines on GPU)"
echo "======================================"

for MODEL in ArcFace Facenet512 GhostFaceNet VGG-Face; do
    echo "--------------------------------------"
    echo "Generating adversarial images for attacker: $MODEL"
    python experiments/run_vanilla_subset_generation.py \
        --input-csv $INPUT_CSV \
        --dataset-root $DATASET_ROOT \
        --output-root $ADV_DIR \
        --attacker-model $MODEL \
        --attacks $ATTACKS || echo "WARNING: Generation failed for $MODEL"
done

echo "======================================"
echo "Phase 2: Evaluation (Cosine Similarities on GPU)"
echo "======================================"

python scripts/evaluate_subset.py \
    --input-csv $INPUT_CSV \
    --dataset-root $DATASET_ROOT \
    --adv-dir $ADV_DIR \
    --output-csv $RESULTS_DIR/subset_raw_similarities_long.csv \
    --attacks $ATTACKS || echo "WARNING: Evaluation encountered an error"

echo "======================================"
echo "Phase 3: Summary (Building Tables)"
echo "======================================"

# Outputting to final_check folder to mirror the local WSL structure
mkdir -p final_check
python scripts/build_subset_baselines.py \
    --raw-long-csv $RESULTS_DIR/subset_raw_similarities_long.csv \
    --input-csv $INPUT_CSV \
    --thresholds-json core/verification_thresholds.json \
    --output-dir final_check || echo "WARNING: Summarizer encountered an error"

echo "======================================"
echo "Kaggle Pipeline Complete! Check /kaggle/working/transferattack/final_check/"
echo "======================================"
