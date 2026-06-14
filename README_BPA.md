# Slide 1: Title Slide
**Title:** BPA-CNN: Backward Propagation Attack on Face Verification
**Author:** Om Singh Rawat
**Date:** June 14, 2026

---
# Slide 2: Selecting BPA-CNN
**Why BPA-CNN?**
* Recent NeurIPS 2023 publication
* Highly effective transferability across CNN-based models
* Present in recent literature but not implemented in the intern baseline

**Code**
* Implemented natively in the TransferAttack repository

---
# Slide 3: Understanding BPA-CNN
**BPA = Backward Propagation Attack**
* Targets the sharp operations (ReLU, MaxPool) that cause gradient masking in CNNs.

**SiLU-Derivative Scaling**
* Replaces sharp ReLU gradients with smooth SiLU-inspired gradients.
* Counteracts binary gradient masking.

**Gaussian Spatial Smoothing**
* Replaces MaxPool's winner-take-all gradient concentration with a spatial Gaussian blur.
* Spreads the gradient information across adjacent pixels.

**Momentum Update**
* Like MI-FGSM, maintains accumulated gradient
* `g = decay*g + grad`
* `adv = adv + alpha*sign(g)`

**Attack Objectives (unchanged)**
* Impersonation: maximize cosine similarity
* Dodging: minimize cosine similarity

---
# Slide 4: Adapting BPA to Face Verification
**Problem**
* Official BPA relies on modifying internal CNN layers (white-box access).
* Baseline uses pre-trained, encapsulated DeepFace models (ArcFace, Facenet512, GhostFaceNet, VGG-Face).
* Dynamically unpacking and modifying internal layers of these frozen models was impractical. Direct copy-paste was impossible.

**Adapted Core Ideas**
* We brought the BPA principles to the **input level**:
  * Gradient normalization
  * SiLU-inspired scaling applied directly to the raw image gradients.
  * Spatial Gaussian convolution applied to the gradients.
  * Momentum accumulation
  * Epsilon-ball projection

---
# Slide 5: Code Integration
**Step 1 — Register BPA_CNN**
```python
ALL_ATTACKS = [
    'PGD', 'MI_FGSM',
    'TI_FGSM', 'SI_NI_FGSM',
    'MI_ADMIX_DI_TI',
    'BPA_CNN' # <-- added
]
```

**BPA-CNN function Pseudo Code**
```python
1. Initialize adversarial image
2. Compute embedding similarity & loss
3. Compute gradient
4. Apply SiLU-derivative smoothing
5. Apply Gaussian spatial smoothing
6. Normalize gradient & Update momentum
7. Project into ε-ball & Clip image
8. Repeat for NUM_ITER
```

**Step 3 — Dispatcher**
```python
if attack_name == "BPA_CNN":
    return bpa_cnn(...)
```

---
# Slide 6: Key Challenges Encountered
**Missing Evaluation Pipeline**
* No script existed to compute similarity scores.
* Only the summarization script was present.
* Evaluation pipeline had been intentionally removed.
* Had to reverse-engineer from baseline CSV structure.

**Dataset Path Mismatch**
* Professor’s CSV used `/content/face_module/...`
* My local WSL used `dataset_extractedfaces/...`
* Solution: Custom path conversion function (`resolve_image_path`).

**Model Input Size Mismatch**
* ArcFace, GhostFaceNet: 112 × 112
* Facenet512: 160 × 160
* VGG-Face: 224 × 224
* Fixed via dynamic per-model resizing.

---
# Slide 7: Reconstructed Evaluation Pipeline
**Pipeline Flow**
1. Load adversarial image
2. Load victim model
3. Compute embedding
4. Load target image
5. Compute target embedding
6. Compute cosine similarity
7. Store CSV row

**Output Files**
* `subset_raw_similarities_long.csv`
* Processed by `build_subset_baselines.py`
* Aggregated into `final_check/` folders

**CSV Schema**
```python
 row_id,
 attacker_model,
 img1,
 img2,
 dataset,
 attack_type,
 victim_model,
 attack_method,
 variant,
 similarity
```

---
# Slide 8: Final Results

### 🏆 Attack Performance Comparison

| Rank | Attack Method | Overall Breach Rate | Dodging Success | Impersonation Success | Impact Mean |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **BPA_CNN** | **29.58%** | **39.17%** | 20.00% | **0.174** |
| 🥈 | **SI_NI_FGSM** | 28.33% | 34.58% | **22.08%** | 0.170 |
| 🥉 | **MI_FGSM** | 25.62% | 32.92% | 18.33% | 0.158 |
| 4 | **MI_ADMIX_DI_TI** | 22.92% | 27.08% | 18.75% | 0.143 |
| 5 | **TI_FGSM** | 20.63% | 24.58% | 16.67% | 0.127 |
| 6 | **PGD** | 16.67% | 20.83% | 12.50% | 0.096 |

> **Note:** The percentages indicate the successful transfer rate across all tested victim models. Impact Mean measures the average magnitude of cosine similarity shift.

**Analysis**
* BPA-CNN outperformed the best baseline (SI-NI-FGSM) in overall transferability by **+1.25%**.
* It achieved the **highest Dodging success rate** (+4.59% over SI_NI_FGSM).
* **Why so high?**
  * BPA was explicitly designed to bypass gradient masking and concentration inherent in CNNs.
  * Since all baseline models (VGG, ResNet-based ArcFace) are CNNs, the input-level smoothing heavily disrupted their shared structural vulnerabilities, leading to massive transferability.

---
# Slide 9: What I Learned
**Technical**
* Transfer attack mechanics & Face verification pipelines
* Reading and adapting research code into input-level equivalents
* Reverse engineering incomplete repositories
* Building missing evaluation infrastructure
* GPU optimization (Rewriting core logic for massive `batch_size=32` parallelism)

**Process**
* Integrating new research into existing codebases
* Quantitative attack comparison
* Debugging silent failures (missing rows, path mismatches)
* Extending baselines without redesigning the system

---
# Slide 10: Thank You
**Thank You**
* Om Singh Rawat
