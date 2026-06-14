import os
import uuid
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from PIL import Image
from deepface import DeepFace

ATTACKER_MODELS = {
    'Facenet512': (160, 160),
    'ArcFace': (112, 112),
    'GhostFaceNet': (112, 112),
    'VGG-Face': (224, 224),
}

VICTIM_MODELS = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face', 'IR152']

ALL_ATTACKS = [
    'PGD',
    'MI_FGSM',
    'TI_FGSM',
    'SI_NI_FGSM',
    'MI_ADMIX_DI_TI',
    'DECOWA',
]

ATTACK_COLS = {
    'PGD': 'pgd_path',
    'MI_FGSM': 'mi_fgsm_path',
    'TI_FGSM': 'ti_fgsm_path',
    'SI_NI_FGSM': 'si_ni_fgsm_path',
    'MI_ADMIX_DI_TI': 'mi_admix_di_ti_path',
    'DECOWA': 'decowa_path',
}

EPSILON = 0.062
NUM_ITER = 5
DECAY = 1.0

# DeCowA (Deformation-Constrained Warping Attack) hyper-parameters.
# Defaults follow the paper: a 3x3 thin-plate-spline control mesh, 20 warping
# samples per iteration, noise strength 2.0, and a single deformation-update
# step with rho=0.01. num_warping is the main cost driver (each sample runs two
# forward/backward passes); lower it (e.g. 8-10) to trade some accuracy for speed.
DECOWA_MESH = 3
DECOWA_NUM_WARPING = 20
DECOWA_NOISE_SCALE = 2.0
DECOWA_RHO = 0.01


def configure_cpu_runtime(tf_threads: int = 1) -> None:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    except Exception:
        pass


def resolve_image_path(path: str, dataset_root: str) -> str:
    value = str(path)
    if os.path.exists(value):
        return value
    marker = 'dataset_extractedfaces/'
    if marker in value:
        rel = value.split(marker, 1)[1]
        candidate = os.path.join(dataset_root, rel)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(dataset_root, value.lstrip('/'))


def load_and_preprocess(path: str, input_size):
    img = Image.open(path).convert('RGB').resize(input_size)
    arr = np.array(img).astype('float32') / 255.0
    return (arr - 0.5) * 2.0


def denormalize(x: np.ndarray) -> np.ndarray:
    x = (x + 1.0) / 2.0
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def compute_embedding(model, x):
    out = model(x, training=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return tf.nn.l2_normalize(out, axis=1)


def attack_loss(cos, attack_type: str):
    return tf.reduce_mean(cos if str(attack_type).strip().lower() == 'impersonation_attack' else (1 - cos))


def save_adv(img_uint8: np.ndarray, attack_name: str, src: str, tgt: str, attack_type: str, model_name: str, row_id: int, adv_root: str) -> str:
    out_dir = Path(adv_root) / model_name / attack_name
    out_dir.mkdir(parents=True, exist_ok=True)
    s = Path(src).stem.replace(' ', '_')
    t = Path(tgt).stem.replace(' ', '_')
    rand = uuid.uuid4().hex[:8]
    name = f'adv_r{row_id}_{s}_to_{t}_{attack_type}_{rand}.png'
    path = out_dir / name
    Image.fromarray(img_uint8).save(path)
    return str(path.resolve())


def gaussian_kernel(k=15, sigma=3.0, ch=3):
    x = tf.range(-k // 2 + 1, k // 2 + 1, dtype=tf.float32)
    g = tf.exp(-tf.square(x) / (2 * sigma**2))
    g /= tf.reduce_sum(g)
    kernel = tf.tensordot(g, g, axes=0)
    kernel = kernel[:, :, None, None]
    return tf.tile(kernel, [1, 1, ch, 1])


def input_diversity(x, input_size, prob=0.7):
    if tf.random.uniform([]) > prob:
        return x
    img_size = input_size[0]
    rnd = tf.random.uniform([], int(0.9 * img_size), img_size, dtype=tf.int32)
    x_resized = tf.image.resize(x, (rnd, rnd))
    pad_total = img_size - rnd
    pad_top = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_bottom = pad_total - pad_top
    pad_left = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_right = pad_total - pad_left
    x_padded = tf.pad(x_resized, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return tf.image.resize(x_padded, input_size)


def pgd_attack(model, x, tgt_emb, attack_type, random_start=True):
    if random_start:
        noise = tf.random.uniform(tf.shape(x), minval=-EPSILON, maxval=EPSILON, dtype=x.dtype)
        adv = tf.clip_by_value(x + noise, -1.0, 1.0)
    else:
        adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def ti_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    kernel = gaussian_kernel()
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def si_ni_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
    for _ in range(NUM_ITER):
        nes = adv + DECAY * alpha * g
        grad_sum = tf.zeros_like(x)
        for s in scales:
            with tf.GradientTape() as tape:
                tape.watch(nes)
                emb = compute_embedding(model, nes * s)
                cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                loss = attack_loss(cos, attack_type)
            grad_sum += tape.gradient(loss, nes)
        grad = grad_sum / len(scales)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_admix_di_ti(model, x, tgt_emb, attack_type, pool_imgs, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel()
    n_pool = tf.shape(pool_imgs)[0]
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            idx = tf.random.uniform([3], 0, n_pool, dtype=tf.int32)
            others = tf.gather(pool_imgs, idx)
            adv_rep = tf.repeat(adv, 3, axis=0)
            mixed = adv_rep + 0.2 * (others - adv_rep)
            batch = input_diversity(mixed, input_size)
            emb = compute_embedding(model, batch)
            tgt_rep = tf.repeat(tgt_emb, 3, axis=0)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


# ----------------------------------------------------------------------------
# DeCowA: Deformation-Constrained Warping Attack
# "Boosting Adversarial Transferability across Model Genus by
#  Deformation-Constrained Warping" (Lin et al., AAAI 2024,
#  https://arxiv.org/abs/2402.03951)
#
# Idea: instead of pixel/scale augmentations, augment each input with an elastic
# thin-plate-spline (TPS) deformation driven by a small grid of control points.
# For every warp, one gradient step first optimizes the control-point noise to
# be a *hard* deformation (deformation-constrained inner step), then the attack
# gradient is taken on the warped image. Averaging over many warps yields a
# perturbation robust to local geometric deformation, which transfers well
# across architectures. This is a TF port of the official PyTorch implementation
# (transferattack/input_transformation/decowa.py), adapted to the embedding-
# cosine objective used by this face-verification pipeline.
# ----------------------------------------------------------------------------


def _decowa_grid_points_2d(width, height):
    """Uniform control-point grid in (x, y) order, flattened to [k, 2]."""
    a = tf.linspace(-1.0, 1.0, height)
    b = tf.linspace(-1.0, 1.0, width)
    xx, yy = tf.meshgrid(a, b, indexing='ij')   # xx <- height axis, yy <- width axis
    pts = tf.stack([yy, xx], axis=-1)            # (x from width, y from height)
    return tf.reshape(pts, [-1, 2])


def _decowa_noisy_grid(width, height, noise_map):
    """Control grid with interior points displaced by noise_map; edges fixed."""
    grid = _decowa_grid_points_2d(width, height)
    mod = tf.pad(noise_map, [[1, 1], [1, 1], [0, 0]])   # interior <- noise, border 0
    return grid + tf.reshape(mod, [-1, 2])


def _decowa_K(X, Y):
    eps = 1e-9
    d2 = tf.reduce_sum(tf.square(X[:, :, None, :] - Y[:, None, :, :]), axis=-1)
    return d2 * tf.math.log(d2 + eps)


def _decowa_P(X):
    n = tf.shape(X)[0]
    k = tf.shape(X)[1]
    return tf.concat([tf.ones([n, k, 1]), X], axis=-1)


def _decowa_tps_coeffs(X, Y):
    """Solve thin-plate-spline coefficients (W non-affine, A affine)."""
    k = tf.shape(X)[1]
    n = tf.shape(X)[0]
    K = _decowa_K(X, X)                                  # [n, k, k]
    P = _decowa_P(X)                                     # [n, k, 3]
    top = tf.concat([K, P], axis=-1)                     # [n, k, k+3]
    bottom = tf.concat([tf.transpose(P, [0, 2, 1]), tf.zeros([n, 3, 3])], axis=-1)
    L = tf.concat([top, bottom], axis=1)                 # [n, k+3, k+3]
    Z = tf.concat([Y, tf.zeros([n, 3, 2])], axis=1)      # [n, k+3, 2]
    Q = tf.linalg.solve(L, Z)
    return Q[:, :k], Q[:, k:]


def _decowa_dense_grid(h, w):
    gx = tf.linspace(-1.0, 1.0, w)
    gy = tf.linspace(-1.0, 1.0, h)
    X0 = tf.tile(gx[None, None, :], [1, h, 1])
    Y0 = tf.tile(gy[None, :, None], [1, 1, w])
    grid = tf.stack([X0, Y0], axis=-1)                   # [1, h, w, 2]
    return tf.reshape(grid, [1, h * w, 2])


def _decowa_tps_grid(X, Y, h, w):
    """Dense sampling grid [1, h, w, 2] produced by the TPS warp X -> Y."""
    W, A = _decowa_tps_coeffs(X, Y)
    base = _decowa_dense_grid(h, w)
    U = _decowa_K(base, X)
    P = _decowa_P(base)
    grid = tf.matmul(P, A) + tf.matmul(U, W)
    return tf.reshape(grid, [1, h, w, 2])


def _decowa_grid_sample(img, grid):
    """Bilinear sampler matching torch.grid_sample(align_corners=False, zeros).

    img:  [N, H, W, C];  grid: [N, H, W, 2] with (x, y) normalized to [-1, 1].
    """
    N = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    Hf = tf.cast(H, tf.float32)
    Wf = tf.cast(W, tf.float32)
    x = grid[..., 0]
    y = grid[..., 1]
    ix = ((x + 1.0) * Wf - 1.0) / 2.0
    iy = ((y + 1.0) * Hf - 1.0) / 2.0
    ix0 = tf.floor(ix)
    iy0 = tf.floor(iy)
    wx1 = ix - ix0
    wy1 = iy - iy0
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    def sample(ixc, iyc):
        in_x = tf.logical_and(ixc >= 0.0, ixc <= Wf - 1.0)
        in_y = tf.logical_and(iyc >= 0.0, iyc <= Hf - 1.0)
        mask = tf.cast(tf.logical_and(in_x, in_y), tf.float32)[..., None]
        xc = tf.clip_by_value(tf.cast(ixc, tf.int32), 0, W - 1)
        yc = tf.clip_by_value(tf.cast(iyc, tf.int32), 0, H - 1)
        bidx = tf.broadcast_to(tf.reshape(tf.range(N), [N, 1, 1]), tf.shape(xc))
        idx = tf.stack([bidx, yc, xc], axis=-1)
        return tf.gather_nd(img, idx) * mask

    v00 = sample(ix0, iy0)
    v01 = sample(ix0, iy0 + 1.0)
    v10 = sample(ix0 + 1.0, iy0)
    v11 = sample(ix0 + 1.0, iy0 + 1.0)
    return (v00 * (wx0 * wy0)[..., None] + v10 * (wx1 * wy0)[..., None]
            + v01 * (wx0 * wy1)[..., None] + v11 * (wx1 * wy1)[..., None])


def _decowa_warp(adv, noise_map, h, w):
    X = _decowa_grid_points_2d(DECOWA_MESH, DECOWA_MESH)[None, ...]
    Y = _decowa_noisy_grid(DECOWA_MESH, DECOWA_MESH, noise_map)[None, ...]
    grid = _decowa_tps_grid(X, Y, h, w)
    grid = tf.tile(grid, [tf.shape(adv)[0], 1, 1, 1])
    return _decowa_grid_sample(adv, grid)


def _decowa_update_noise_map(model, adv, tgt_emb, attack_type, h, w):
    """One deformation-constrained step: find a hard warp (descend the loss)."""
    nm = (tf.random.uniform([DECOWA_MESH - 2, DECOWA_MESH - 2, 2]) - 0.5) * DECOWA_NOISE_SCALE
    with tf.GradientTape() as tape:
        tape.watch(nm)
        warped = _decowa_warp(adv, nm, h, w)
        emb = compute_embedding(model, warped)
        cos = tf.reduce_sum(emb * tgt_emb, axis=1)
        loss = attack_loss(cos, attack_type)
    grad = tape.gradient(loss, nm)
    if grad is None:
        return nm
    grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
    return nm - DECOWA_RHO * grad


def decowa(model, x, tgt_emb, attack_type, input_size):
    """DeCowA attack on the embedding-cosine objective (MI-FGSM base)."""
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    h, w = int(input_size[1]), int(input_size[0])  # input_size is (width, height)
    for _ in range(NUM_ITER):
        grads = tf.zeros_like(x)
        for _ in range(DECOWA_NUM_WARPING):
            noise_map = _decowa_update_noise_map(model, tf.stop_gradient(adv), tgt_emb, attack_type, h, w)
            with tf.GradientTape() as tape:
                tape.watch(adv)
                warped = _decowa_warp(adv, noise_map, h, w)
                emb = compute_embedding(model, warped)
                cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                loss = attack_loss(cos, attack_type)
            grad = tape.gradient(loss, adv)
            grads += tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        grads = grads / DECOWA_NUM_WARPING
        grads = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)
        g = DECAY * g + grads
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def build_attacker(model_name: str):
    return DeepFace.build_model(model_name).model


def run_attack(attack_name: str, model, src, tgt, attack_type: str, input_size):
    tgt_emb = compute_embedding(model, tgt)
    if attack_name == 'PGD':
        return pgd_attack(model, src, tgt_emb, attack_type)
    if attack_name == 'MI_FGSM':
        return mi_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'TI_FGSM':
        return ti_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'SI_NI_FGSM':
        return si_ni_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'MI_ADMIX_DI_TI':
        pool_imgs = tf.concat([src, tgt, src], axis=0)
        return mi_admix_di_ti(model, src, tgt_emb, attack_type, pool_imgs, input_size)
    if attack_name == 'DECOWA':
        return decowa(model, src, tgt_emb, attack_type, input_size)
    raise ValueError(f'Unsupported attack: {attack_name}')
