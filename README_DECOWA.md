# 🚀 Final Project Completion Summary: Evaluating DECOWA

This document serves as the comprehensive final summary of the **DECOWA** (Dual-stage Evolution with COrrelation-guided WArping) attack evaluation. It outlines the complete workflow, including the generation of adversarial images across two separate Kaggle sessions, the merging of split evaluation results, and the computation of the final breach rate leaderboards.

The primary goal was to benchmark **DECOWA** — a state-of-the-art transferable adversarial attack — against the five established vanilla baselines (SI\_NI\_FGSM, MI\_FGSM, MI\_ADMIX\_DI\_TI, TI\_FGSM, and PGD) and our custom BPA\_CNN implementation.

---

## 1. Adversarial Image Generation

The adversarial images for DECOWA were generated on **Google Colab / Kaggle** using GPU acceleration. Due to session time limits, the generation was split into two batches:

- **Batch 1 (Attackers: ArcFace & GhostFaceNet):** Generated adversarial images using ArcFace and GhostFaceNet as surrogate models.
- **Batch 2 (Attackers: Facenet512 & VGG-Face):** Generated adversarial images using Facenet512 and VGG-Face as surrogate models.

All generated images and their path CSVs are stored in the `outputs_decowa_colab/` folder, organized by attacker model:

```
outputs_decowa_colab/
├── ArcFace/DECOWA/          # 30 adversarial images
├── Facenet512/DECOWA/       # 30 adversarial images
├── GhostFaceNet/DECOWA/     # 30 adversarial images
├── VGG-Face/DECOWA/         # 30 adversarial images
├── ArcFace_subset_adv_paths.csv
├── Facenet512_subset_adv_paths.csv
├── GhostFaceNet_subset_adv_paths.csv
└── VGG-Face_subset_adv_paths.csv
```

---

## 2. Evaluation Pipeline

### Split Evaluation Strategy

Since adversarial images were generated in two batches, evaluation was also performed in two separate runs:

1. **Run 1:** Evaluated adversarial images from ArcFace & GhostFaceNet attackers against all 5 victim models → `subset_raw_similarities_long.csv`
2. **Run 2:** Evaluated adversarial images from Facenet512 & VGG-Face attackers against all 5 victim models → `subset_raw_similarities_long_v.csv`

### Merging the Results

The two raw CSV files were merged by appending the data rows from the second file into the first:

```bash
# Skip the header of the second CSV and append to the first
tail -n +2 results_decowa_colab/subset_raw_similarities_long_v.csv >> results_decowa_colab/subset_raw_similarities_long.csv
```

The merged file was then fed into `scripts/build_subset_baselines.py` to compute the final breach rates.

### Important Note: Why Each Run Evaluated All 5 Victims

Even though only 2 attacker models were used per run, the evaluation script tested those adversarial images against **all 5 victim models** (Facenet512, ArcFace, GhostFaceNet, VGG-Face, IR152). This is by design — the entire purpose of a **transferability** study is to measure whether adversarial images crafted on one model can fool a completely different model.

---

## 3. Final Summary Generation

The final summaries were generated using:

```bash
python3 scripts/build_subset_baselines.py \
    --raw-long-csv results_decowa_colab/subset_raw_similarities_long.csv \
    --input-csv docs/subset_input_pairs.csv \
    --thresholds-json core/verification_thresholds.json \
    --output-dir final_check_decowa
```

All results are stored in `final_check_decowa/`.

---

## 🏆 Final Results

### Overall DECOWA Breach Rate

| Attack Method | Num Rows | Breach Rate (%) | Impact Mean |
|:---|:---:|:---:|:---:|
| **DECOWA** | 480 | **33.13%** | 0.1854 |

### Breach Rate by Attack Goal

| Attack Type | Num Rows | Breach Rate (%) | Impact Mean |
|:---|:---:|:---:|:---:|
| Dodging Attack | 240 | **43.75%** | 0.2386 |
| Impersonation Attack | 240 | **22.50%** | 0.1323 |

### Breach Rate by Victim Model (Averaged Across All Attackers)

| Victim Model | Avg Breach Rate (%) | Avg Impact Mean |
|:---|:---:|:---:|
| ArcFace | 44.44% | 0.194 |
| GhostFaceNet | 44.44% | 0.115 |
| VGG-Face | 34.44% | 0.143 |
| Facenet512 | 31.11% | 0.278 |
| IR152 | 16.67% | 0.194 |

> **Note:** IR152 shows significantly lower breach rates because it is a 152-layer PyTorch model with fundamentally different architecture from the Keras-based attacker models. This cross-framework transfer gap is well-documented in the literature and consistent with expected behavior.

### Full Attacker → Victim Breakdown

| Attacker | Victim | Breach Rate (%) | Impact Mean |
|:---|:---|:---:|:---:|
| **VGG-Face** | Facenet512 | **60.00%** | 0.4616 |
| **Facenet512** | VGG-Face | **60.00%** | 0.2356 |
| **Facenet512** | ArcFace | **56.67%** | 0.2467 |
| **VGG-Face** | ArcFace | **56.67%** | 0.2506 |
| **Facenet512** | GhostFaceNet | 50.00% | 0.1299 |
| **VGG-Face** | GhostFaceNet | 43.33% | 0.1130 |
| **Facenet512** | IR152 | 40.00% | 0.2697 |
| **ArcFace** | GhostFaceNet | 40.00% | 0.1032 |
| **ArcFace** | VGG-Face | 33.33% | 0.1421 |
| **ArcFace** | Facenet512 | 30.00% | 0.2565 |
| **GhostFaceNet** | ArcFace | 20.00% | 0.0843 |
| **VGG-Face** | IR152 | 16.67% | 0.2247 |
| **ArcFace** | IR152 | 10.00% | 0.1923 |
| **GhostFaceNet** | VGG-Face | 10.00% | 0.0524 |
| **GhostFaceNet** | Facenet512 | 3.33% | 0.1151 |
| **GhostFaceNet** | IR152 | 0.00% | 0.0892 |

### Key Observations

- **VGG-Face and Facenet512 are the strongest DECOWA attackers**, achieving 60% breach rates against each other. Their adversarial images transfer very effectively.
- **GhostFaceNet is the weakest attacker** for DECOWA, with breach rates between 0–20%. Its internal representations do not produce adversarial perturbations that generalize well across models.
- **IR152 is the most robust victim** across all attackers, consistent with its role as the deep, cross-framework benchmark.

---

## 📊 Combined Leaderboard: All Attacks

After resolving all evaluation issues and running a strict, fair, apples-to-apples comparison across all 5 standard victims:

| Rank | Attack Method | Breach Rate (%) | Impact Mean |
|:---:|:---|:---:|:---:|
| 🥇 1 | **DECOWA** | **33.13%** | 0.1854 |
| 🥈 2 | BPA\_CNN (Ours) | 29.58% | 0.1743 |
| 🥉 3 | SI\_NI\_FGSM | 28.33% | 0.1697 |
| 4 | MI\_FGSM | 25.62% | 0.1576 |
| 5 | MI\_ADMIX\_DI\_TI | 22.92% | 0.1434 |
| 6 | TI\_FGSM | 20.62% | 0.1267 |
| 7 | PGD | 16.67% | 0.0962 |

### Conclusion

**DECOWA achieves the highest overall breach rate of 33.13%**, outperforming all vanilla baselines and our custom BPA\_CNN implementation by a significant margin (+3.55% over BPA\_CNN, +4.80% over SI\_NI\_FGSM). This confirms DECOWA as a state-of-the-art transferable adversarial attack method for face recognition systems.

---

## 📁 File Structure

```
transferattack/
├── outputs_decowa_colab/              # Generated adversarial images
│   ├── ArcFace/DECOWA/
│   ├── Facenet512/DECOWA/
│   ├── GhostFaceNet/DECOWA/
│   └── VGG-Face/DECOWA/
├── results_decowa_colab/              # Raw evaluation CSVs
│   ├── subset_raw_similarities_long.csv    (merged, all 4 attackers)
│   └── subset_raw_similarities_long_v.csv  (Facenet512 & VGG-Face only)
├── final_check_decowa/                # Final summary tables
│   ├── subset_attack_summary.csv
│   ├── subset_attack_summary_by_goal.csv
│   ├── subset_attacker_victim_summary.csv
│   ├── subset_attack_eval_long.csv
│   └── subset_input_pairs.csv
└── README_DECOWA.md                   # This file
```
