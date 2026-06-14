# ======================================================================
# DeCowA Attack Pipeline — Generation, Evaluation, Summary
# ======================================================================

# Configuration
$INPUT_CSV = "docs/subset_input_pairs.csv"
$DATASET_ROOT = "../interns/dataset_extractedfaces"
$ADV_DIR = "outputs_decowa"
$RESULTS_DIR = "results_decowa"
$SUMMARY_DIR = "results_decowa_summary"
$ATTACKS = "DECOWA"

# Create output directories
New-Item -ItemType Directory -Force -Path $ADV_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $SUMMARY_DIR | Out-Null

Write-Host "======================================"
Write-Host "  DeCowA Attack Pipeline"
Write-Host "======================================"
Write-Host ""

# ------------------------------------------------------------------
# Phase 1: Generate Adversarial Images
# ------------------------------------------------------------------
Write-Host "======================================"
Write-Host "Phase 1: Generation (DeCowA Adversarial Images)"
Write-Host "======================================"

$models = @("ArcFace", "Facenet512", "GhostFaceNet", "VGG-Face")

foreach ($MODEL in $models) {
    Write-Host "--------------------------------------"
    Write-Host "Generating DeCowA adversarial images for attacker: $MODEL"
    Write-Host "--------------------------------------"

    python experiments/run_vanilla_subset_generation.py `
        --input-csv $INPUT_CSV `
        --dataset-root $DATASET_ROOT `
        --output-root $ADV_DIR `
        --attacker-model $MODEL `
        --attacks $ATTACKS

    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Generation failed for $MODEL" -ForegroundColor Yellow
    } else {
        Write-Host "Done: $MODEL" -ForegroundColor Green
    }
}

# ------------------------------------------------------------------
# Phase 2: Evaluate Cosine Similarities
# ------------------------------------------------------------------
Write-Host ""
Write-Host "======================================"
Write-Host "Phase 2: Evaluation (Cosine Similarities)"
Write-Host "======================================"

python scripts/evaluate_subset.py `
    --input-csv $INPUT_CSV `
    --dataset-root $DATASET_ROOT `
    --adv-dir $ADV_DIR `
    --output-csv "$RESULTS_DIR/subset_raw_similarities_long.csv" `
    --attacks $ATTACKS

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Evaluation encountered an error" -ForegroundColor Yellow
} else {
    Write-Host "Evaluation complete!" -ForegroundColor Green
}

# ------------------------------------------------------------------
# Phase 3: Build Summary Tables (Breach Rate, Impact, etc.)
# ------------------------------------------------------------------
Write-Host ""
Write-Host "======================================"
Write-Host "Phase 3: Summary (Breach Rate, Impact, Attacker-Victim)"
Write-Host "======================================"

python scripts/build_subset_baselines.py `
    --raw-long-csv "$RESULTS_DIR/subset_raw_similarities_long.csv" `
    --input-csv $INPUT_CSV `
    --thresholds-json core/verification_thresholds.json `
    --output-dir $SUMMARY_DIR

if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Summary generation encountered an error" -ForegroundColor Yellow
} else {
    Write-Host "Summary tables generated!" -ForegroundColor Green
}

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
Write-Host ""
Write-Host "======================================"
Write-Host "  Pipeline Complete!"
Write-Host "======================================"
Write-Host ""
Write-Host "Output locations:"
Write-Host "  Adversarial images:    $ADV_DIR/"
Write-Host "  Raw similarities:      $RESULTS_DIR/subset_raw_similarities_long.csv"
Write-Host "  Attack summary:        $SUMMARY_DIR/subset_attack_summary.csv"
Write-Host "  Summary by goal:       $SUMMARY_DIR/subset_attack_summary_by_goal.csv"
Write-Host "  Attacker-victim table: $SUMMARY_DIR/subset_attacker_victim_summary.csv"
Write-Host "  Full eval table:       $SUMMARY_DIR/subset_attack_eval_long.csv"
Write-Host ""
