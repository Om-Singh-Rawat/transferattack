# Baseline Adversarial Images Output

This directory contains the generated adversarial images for the original baseline attack methods:
- `SI_NI_FGSM`
- `MI_FGSM`
- `MI_ADMIX_DI_TI`
- `TI_FGSM`
- `PGD`

## How these were generated
These images were generated using the standard `run_vanilla_subset_generation.py` script for each of the 4 Keras attacker models (`Facenet512`, `ArcFace`, `GhostFaceNet`, and `VGG-Face`).
