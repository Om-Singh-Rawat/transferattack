# DECOWA — Presentation Slides README

**Om Singh Rawat** · Transfer Attacks in Face Recognition · June 16, 2026

---

## Slide 1 — Attacks Considered

- PGD (baseline)
- MI-FGSM (momentum-based)
- TI-FGSM (translation-invariant)
- SI-NI-FGSM (scale-invariant Nesterov)
- MI-ADMIX-DI-TI (admixed input diversity)
- BPA-CNN (backward propagation adaptation)
- **DECOWA (deformation-constrained warping attack) ← NEW**

---

## Slide 2 — Selecting DECOWA

### Why DECOWA?

- ❑ Recent publication from adversarial robustness literature (AAAI 2024)
- ❑ Achieves state-of-the-art transferability via geometric image warping
- ❑ Fundamentally different approach from gradient-only methods — applies spatial deformation
- ❑ Not implemented in the intern baseline
- ❑ Code reference — [GitHub Repo](https://github.com/lzj-isee/DECOWA)

---

## Slide 3 — DECOWA (Deformation-Constrained Warping Attack)

### Definition & Nature

- ❑ White-box transfer-based adversarial attack
- ❑ Gradient-based optimization method enhanced with geometric warping
- ❑ Adapted for black-box / pre-trained CNN face recognition models
- ❑ Uses Thin-Plate-Spline (TPS) deformation to augment inputs during gradient computation

### Basic Idea

- ❑ Instead of only optimizing pixel perturbations, DECOWA also optimizes **spatial warping** of the input
- ❑ Generates multiple deformed copies of the adversarial image per iteration
- ❑ Averages gradients across all deformed copies to produce a more transferable perturbation
- ❑ The warping is constrained via a deformation step (ρ) to prevent extreme distortion
- ❑ This "input diversity via geometric deformation" avoids overfitting to the surrogate model's decision boundary

---

## Slide 4 — Working of DECOWA Attack

### Algorithm Steps

1. Initialize adversarial image as original input
2. Define attack objective (based on attack type — impersonation or dodging)
3. **For T iterations:**
   1. Initialize gradient accumulator to zero
   2. **For K warping samples:**
      - Generate a random noise map for the TPS control grid
      - Compute deformation-update step: descend the attack loss w.r.t. noise map (ρ = 0.01)
      - Warp the adversarial image using the updated TPS noise map
      - Compute gradient of attack loss w.r.t. the adversarial image through the warped copy
      - Accumulate the gradient
   3. Average gradients across all K warping samples
   4. Normalize gradient by mean absolute value
   5. Update momentum: g = DECAY × g + normalized\_gradient
   6. Update image: adv = adv + α × sign(g)
   7. Project image into ε-bounded region
   8. Clip to valid pixel range \[-1, 1\]
4. Return final adversarial image

---

## Slide 5 — DECOWA Goal

- Operates in **embedding space** (cosine similarity)
- **Impersonation:** increase cosine similarity between adversarial face and target face → make the system think two different people are the same
- **Dodging:** decrease cosine similarity between adversarial face and its genuine pair → make the system fail to recognize the same person
- Optimizes embedding distance directly through differentiable warping

---

## Slide 6 — Understanding DECOWA

### DECOWA = Deformation-Constrained Warping Attack

> Targets the **model overfitting problem** in transfer attacks by augmenting inputs with geometric deformations.

### Thin-Plate-Spline (TPS) Warping

- Defines a 3×3 control mesh over the image
- Interior control points are displaced by a learnable noise map
- Edge points remain fixed to prevent extreme distortion
- Produces smooth, differentiable spatial deformations

### Deformation-Constrained Update

- For each warping sample, the noise map is initialized randomly
- A single gradient descent step on the noise map finds a "hard" warp — a deformation that makes the attack objective harder
- This forces the gradient to account for geometric variations, improving transferability

### Gradient Averaging Over Warps

- 20 warping samples per iteration (DECOWA\_NUM\_WARPING = 20)
- Each sample produces a different deformed view of the adversarial image
- Averaging gradients across all views smooths out model-specific features
- Similar philosophy to DI-FGSM (input diversity) but using geometric deformation instead of random resizing

### Momentum Update

- Same as MI-FGSM: maintains accumulated gradient momentum
- g = DECAY × g + avg\_gradient
- adv = adv + α × sign(g)

---

## Slide 7 — Attack Implementation

### DECOWA in Our Implementation

| Parameter | Value |
|:---|:---:|
| ε (epsilon) | 0.062 |
| NUM\_ITER | 5 |
| DECAY (momentum) | 1.0 |
| DECOWA\_MESH | 3 (3×3 TPS control grid) |
| DECOWA\_NUM\_WARPING | 20 (warping samples per iteration) |
| DECOWA\_NOISE\_SCALE | 2.0 (initial noise magnitude) |
| DECOWA\_RHO | 0.01 (deformation update step size) |

### Implementation Details

1. Initializes momentum variable `g`
2. For each iteration, performs **20 warping samples**:
   - Random TPS noise map → deformation-constrained update (ρ = 0.01)
   - Warp adversarial image via TPS grid → bilinear sampling
   - Compute gradient through warped image
3. Average gradients across all 20 warps
4. Normalize by mean absolute value
5. Momentum update: `g = DECAY * g + normalized_grad`
6. Standard ε-projection via clipping (ε = 0.062)
7. Full implementation → `core/transfer_attack_core.py`

---

## Slide 8 — Results: Overall DECOWA Performance

### Overall Breach Rate

| Attack Method | Num Rows | Breach Rate (%) | Impact Mean |
|:---|:---:|:---:|:---:|
| **DECOWA** | 480 | **33.13%** | 0.1854 |

### Breach Rate by Attack Goal

| Attack Type | Breach Rate (%) | Impact Mean |
|:---|:---:|:---:|
| Dodging Attack | **43.75%** | 0.2386 |
| Impersonation Attack | **22.50%** | 0.1323 |

---

## Slide 9 — Results: Per-Victim Breach Rates

### Average Breach Rate by Victim Model

| Victim Model | Avg Breach Rate (%) | Avg Impact Mean |
|:---|:---:|:---:|
| ArcFace | 44.44% | 0.194 |
| GhostFaceNet | 44.44% | 0.115 |
| VGG-Face | 34.44% | 0.143 |
| Facenet512 | 31.11% | 0.278 |
| IR152 | 16.67% | 0.194 |

> **Note:** IR152 shows significantly lower breach rates because it is a 152-layer PyTorch model with a fundamentally different architecture from the Keras-based surrogate models. This cross-framework transfer gap is well-documented in the literature.

---

## Slide 10 — Results: Full Attacker → Victim Breakdown

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
| GhostFaceNet | ArcFace | 20.00% | 0.0843 |
| VGG-Face | IR152 | 16.67% | 0.2247 |
| ArcFace | IR152 | 10.00% | 0.1923 |
| GhostFaceNet | VGG-Face | 10.00% | 0.0524 |
| GhostFaceNet | Facenet512 | 3.33% | 0.1151 |
| GhostFaceNet | IR152 | 0.00% | 0.0892 |

### Key Observations

- **VGG-Face and Facenet512 are the strongest DECOWA attackers** — achieving 60% breach rates against each other
- **GhostFaceNet is the weakest attacker** — its perturbations do not generalize well across architectures
- **IR152 is the most robust victim** — consistent with its role as the deep, cross-framework benchmark

---

## Slide 11 — Combined Leaderboard: All Attacks

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

**DECOWA achieves the highest overall breach rate of 33.13%**, outperforming all vanilla baselines and our custom BPA\_CNN implementation by a significant margin (+3.55% over BPA\_CNN, +4.80% over SI\_NI\_FGSM).

DECOWA's strength lies in its **geometric input diversity** — by averaging gradients across 20 different spatially-warped copies per iteration, it produces perturbations that generalize across model architectures far better than purely gradient-based methods.
