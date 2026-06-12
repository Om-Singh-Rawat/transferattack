#!/usr/bin/env python3
"""Sanity check for the BPA backward-gradient patch.

BPA leaves the forward pass unchanged and only replaces the *backward*
(gradient) functions of ReLU / MaxPool. This script verifies, for a given
surrogate model, that:

  1. The forward embedding is unchanged under the patch (it must be).
  2. The input gradient *does* change under the patch (otherwise the patch
     intercepted nothing on this architecture and BPA is a no-op there).

Run from the repo root, e.g.:
    python scripts/check_bpa_patch.py VGG-Face
    python scripts/check_bpa_patch.py Facenet512
"""
from __future__ import annotations

import sys
import contextlib

import numpy as np
import tensorflow as tf

from core.transfer_attack_core import (
    ATTACKER_MODELS,
    build_attacker,
    compute_embedding,
    attack_loss,
    bpa_backward_patch,
    configure_cpu_runtime,
)


def _grad(model, x, tgt_emb, patched):
    ctx = bpa_backward_patch() if patched else contextlib.nullcontext()
    with ctx:
        with tf.GradientTape() as tape:
            tape.watch(x)
            emb = compute_embedding(model, x)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, 'impersonation_attack')
        return tape.gradient(loss, x), emb


def main():
    configure_cpu_runtime(1)
    name = sys.argv[1] if len(sys.argv) > 1 else 'VGG-Face'
    if name not in ATTACKER_MODELS:
        raise SystemExit(f'Unknown model {name}. Choose from {list(ATTACKER_MODELS)}')

    size = ATTACKER_MODELS[name][0]
    model = build_attacker(name)

    rng = np.random.default_rng(0)
    x = tf.constant(rng.uniform(-1, 1, (1, size, size, 3)).astype('float32'))
    t = tf.constant(rng.uniform(-1, 1, (1, size, size, 3)).astype('float32'))
    tgt_emb = compute_embedding(model, t)

    g0, e0 = _grad(model, x, tgt_emb, patched=False)
    g1, e1 = _grad(model, x, tgt_emb, patched=True)

    fwd_diff = float(tf.norm(e1 - e0))
    rel_change = float(tf.norm(g1 - g0) / (tf.norm(g0) + 1e-12))
    f0 = tf.nn.l2_normalize(tf.reshape(g0, [-1]), 0)
    f1 = tf.nn.l2_normalize(tf.reshape(g1, [-1]), 0)
    grad_cos = float(tf.reduce_sum(f0 * f1))

    print(f'model            : {name}')
    print(f'forward diff     : {fwd_diff:.3e}   (should be ~0: BPA must not change the forward pass)')
    print(f'grad rel. change : {rel_change:.4f}')
    print(f'grad cosine sim  : {grad_cos:.4f}')
    if rel_change > 1e-4:
        print('=> BPA backward IS active on this model.')
    else:
        print('=> WARNING: patch had no measurable effect — ReLU/MaxPool ops were '
              'not intercepted on this architecture. BPA would behave like MI-FGSM here.')


if __name__ == '__main__':
    main()
