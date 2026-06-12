---
# Slide 1: Title Slide
**Title:** Transfer Attacks in Face Recognition
**Subtitle:** Backward Propagation Attack (BPA-CNN)
**Author:** Om Singh Rawat
**Date:** June 12, 2026

---
# Slide 2: Attacks Considered
* PGD (baseline)
* MI-FGSM (momentum-based)
* TI-FGSM (translation-invariant)
* SI-NI-FGSM (scale-invariant Nesterov)
* MI-ADMIX-DI-TI (admixed input diversity)
* **BPA-CNN (backward propagation adaptation)**

---
# Slide 3: BPA-CNN (Backward Propagation Attack)
**Definition & Nature**
* White-box transfer-based adversarial attack
* Gradient-based optimization method
* Adapted for black-box/pre-trained CNN models
* Based on NeurIPS 2023 paper implementation

**Basic Idea**
* Replaces sharp backward pass operations (ReLU, MaxPool) with smooth alternatives
* Smooths gradients at the input level to prevent overfitting to the surrogate model
* Counteracts ReLU binary gradient masking and MaxPool winner-take-all gradient concentration

---
# Slide 4: BPA-CNN - Goal
**Operates in embedding space**
* **Impersonation:** increase cosine similarity
* **Dodging:** decrease cosine similarity
* Optimizes embedding distance directly

**Working of BPA-CNN attack**
1. Initialize adversarial image as original input
2. Define attack objective (based on attack type)
3. For T iterations:
   * Compute gradient of objective w.r.t input
   * Scale gradient and apply SiLU-derivative smoothing
   * Apply Gaussian spatial smoothing to the gradient
   * Normalize gradient and update accumulated gradient (momentum)
   * Update image using sign of accumulated gradient
   * Project image into ε bounded region
   * Clip to valid pixel range
4. Return final adversarial image

---
# Slide 5: Attack Implementation
**BPA-CNN in Our Implementation**
1. Initializes momentum variable `g`
2. Uses SiLU-derivative gradient scaling (`silu_deriv`) with Temperature = 3.0
3. Applies spatial Gaussian smoothing with Kernel = 5, Sigma = 1.0
4. Normalizes and updates accumulated gradient: `g = DECAY * g + grad`
5. Standard ε-projection via clipping (ε = 0.062)
6. Fixed number of iterations (`NUM_ITER = 5`, `DECAY = 1.0`)

---
# Slide 6: Performance Results on Subset
**Overall Breach Rate (Transferability)**
* **BPA_CNN: 34.42% (Highest)**
* SI_NI_FGSM: 30.98%
* MI_FGSM: 25.82%
* MI_ADMIX_DI_TI: 24.18%
* TI_FGSM: 20.92%
* PGD: 17.39%

**Performance by Goal**
* **Dodging Success Rate:** BPA_CNN achieves **48.08%** (vs SI_NI_FGSM: 39.90%)
* **Impersonation Success Rate:** BPA_CNN achieves **16.67%** (vs SI_NI_FGSM: 19.38%)

**Why BPA-CNN Performs Well**
* Smoother gradients reduce model overfitting
* Enhances cross-model transferability specifically on CNN architectures

---
# Slide 7: Thank You
**Thank You**
