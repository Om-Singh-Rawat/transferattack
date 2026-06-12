import os
import uuid
import contextlib
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
    'BPA_CNN',
]

ATTACK_COLS = {
    'PGD': 'pgd_path',
    'MI_FGSM': 'mi_fgsm_path',
    'TI_FGSM': 'ti_fgsm_path',
    'SI_NI_FGSM': 'si_ni_fgsm_path',
    'MI_ADMIX_DI_TI': 'mi_admix_di_ti_path',
    'BPA_CNN': 'bpa_cnn_path',
}

EPSILON = 0.062
NUM_ITER = 5
DECAY = 1.0

# BPA (Backward Propagation Attack) surrogate-gradient temperatures.
# Higher temperature -> sharper (closer to the original truncating backward);
# lower -> smoother gradient flow. 1.0 matches the SiLU-derivative form used
# in the reference implementation.
BPA_RELU_TEMP = 1.0
BPA_POOL_TEMP = 1.0


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
# BPA: Backward Propagation Attack
# "Rethinking the Backward Propagation for Adversarial Transferability"
# (Wang et al., NeurIPS 2023, https://arxiv.org/abs/2306.12685)
#
# Core idea: ReLU and MaxPool *truncate* gradients in the backward pass
# (ReLU zeroes the gradient for negative pre-activations; MaxPool routes the
# gradient only to the single argmax element). This truncation hurts black-box
# transferability. BPA keeps the FORWARD pass identical but replaces the
# BACKWARD function of these ops with smooth surrogates so richer gradient
# information reaches the input:
#   * ReLU   -> backward uses the derivative of SiLU (x * sigmoid(x)).
#   * MaxPool-> backward distributes the gradient across the pooling window
#              via a temperature softmax instead of a hard one-hot routing.
#
# The reference code rewires torchvision ResNet layers directly. Since the
# DeepFace surrogates here are pre-built Keras models we cannot edit, we inject
# the same surrogate backward functions by temporarily patching the ReLU/MaxPool
# ops the models call, using tf.custom_gradient (forward unchanged, backward
# replaced). Run scripts/check_bpa_patch.py to confirm the patch is active on a
# given surrogate.
# ----------------------------------------------------------------------------


@tf.custom_gradient
def _bpa_relu(x):
    """ReLU forward, SiLU-derivative backward (BPA ReLU surrogate).

    Forward must use tf.math.maximum, NOT tf.nn.relu: under bpa_backward_patch()
    tf.nn.relu is rebound to this wrapper, so calling it here would recurse
    infinitely. tf.math.maximum(x, 0.0) equals relu and is not patched.
    """
    y = tf.math.maximum(x, 0.0)

    def grad(dy):
        s = tf.sigmoid(BPA_RELU_TEMP * x)
        # d/dx [ x * sigmoid(t*x) ] = sigmoid(t*x) + t*x*sigmoid(t*x)*(1-sigmoid(t*x))
        silu_grad = s + BPA_RELU_TEMP * x * s * (1.0 - s)
        return dy * silu_grad

    return y, grad


def _parse_pool_params(ksize, strides):
    def _two(v):
        if isinstance(v, int):
            return v, v
        v = list(v)
        if len(v) == 1:
            return v[0], v[0]
        if len(v) == 2:
            return v[0], v[1]
        if len(v) == 4:  # NHWC -> [1, kh, kw, 1]
            return v[1], v[2]
        return v[0], v[1]
    kh, kw = _two(ksize)
    sh, sw = _two(strides)
    return kh, kw, sh, sw


def _bpa_maxpool_nonoverlap(x, kh, kw):
    """MaxPool forward, softmax-weighted backward (BPA MaxPool surrogate).

    Implemented for the non-overlapping VALID case (stride == kernel), which
    covers the standard pooling layers in these FR backbones (e.g. VGG-Face's
    2x2/stride-2 pools). The forward output is identical to tf.nn.max_pool2d;
    only the backward routing is softened. Overlapping / SAME-padded pools fall
    back to the standard op (see the patch wrappers below).
    """
    shp = tf.shape(x)
    B, H, W, C = shp[0], shp[1], shp[2], shp[3]
    Ho = H // kh
    Wo = W // kw
    x = x[:, :Ho * kh, :Wo * kw, :]

    @tf.custom_gradient
    def op(xx):
        r = tf.reshape(xx, tf.stack([B, Ho, kh, Wo, kw, C]))
        r = tf.transpose(r, [0, 1, 3, 5, 2, 4])              # B, Ho, Wo, C, kh, kw
        rf = tf.reshape(r, tf.stack([B, Ho, Wo, C, kh * kw]))
        y = tf.reduce_max(rf, axis=-1)                        # B, Ho, Wo, C

        def grad(dy):
            w = tf.nn.softmax(BPA_POOL_TEMP * rf, axis=-1)    # soft routing weights
            din = w * tf.expand_dims(dy, -1)
            din = tf.reshape(din, tf.stack([B, Ho, Wo, C, kh, kw]))
            din = tf.transpose(din, [0, 1, 4, 2, 5, 3])       # B, Ho, kh, Wo, kw, C
            din = tf.reshape(din, tf.stack([B, Ho * kh, Wo * kw, C]))
            return din

        return y, grad

    return op(x)


@contextlib.contextmanager
def bpa_backward_patch():
    """Temporarily route ReLU/MaxPool through BPA surrogate-gradient ops.

    Patches the functions the Keras surrogate actually calls (tf.nn.relu /
    tf.keras.backend.relu and the max-pool ops). The forward result is
    unchanged; only the backward (gradient) computation is replaced. Anything
    not matched (e.g. fused or non-standard activations) simply keeps its normal
    gradient, so the attack always runs.
    """
    K = tf.keras.backend
    nn = tf.nn
    o = {
        'nn_relu': getattr(nn, 'relu', None),
        'nn_mp2': getattr(nn, 'max_pool2d', None),
        'nn_mp': getattr(nn, 'max_pool', None),
        'k_relu': getattr(K, 'relu', None),
        'k_pool': getattr(K, 'pool2d', None),
    }

    def p_nn_relu(features, name=None):
        return _bpa_relu(features)

    def p_k_relu(x, alpha=0.0, max_value=None, threshold=0.0):
        if (alpha in (0.0, 0)) and (max_value is None) and (threshold in (0.0, 0)):
            return _bpa_relu(x)
        return o['k_relu'](x, alpha=alpha, max_value=max_value, threshold=threshold)

    def _try_soft_maxpool(x, ksize, strides, padding, data_format):
        try:
            kh, kw, sh, sw = _parse_pool_params(ksize, strides)
            ok_df = data_format in (None, 'NHWC', 'channels_last')
            if str(padding).upper() == 'VALID' and kh == sh and kw == sw and ok_df:
                return _bpa_maxpool_nonoverlap(x, kh, kw)
        except Exception:
            pass
        return None

    def p_nn_mp2(input, ksize, strides, padding, data_format=None, name=None):
        r = _try_soft_maxpool(input, ksize, strides, padding, data_format)
        return r if r is not None else o['nn_mp2'](input, ksize, strides, padding, data_format=data_format, name=name)

    def p_nn_mp(input, ksize, strides, padding, data_format=None, name=None):
        r = _try_soft_maxpool(input, ksize, strides, padding, data_format)
        return r if r is not None else o['nn_mp'](input, ksize, strides, padding, data_format=data_format, name=name)

    def p_k_pool(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max'):
        if pool_mode == 'max':
            try:
                kh, kw = pool_size[0], pool_size[1]
                sh, sw = strides[0], strides[1]
                df = data_format or K.image_data_format()
                if str(padding).lower() == 'valid' and kh == sh and kw == sw and df == 'channels_last':
                    return _bpa_maxpool_nonoverlap(x, kh, kw)
            except Exception:
                pass
        return o['k_pool'](x, pool_size, strides=strides, padding=padding, data_format=data_format, pool_mode=pool_mode)

    try:
        if o['nn_relu'] is not None:
            nn.relu = p_nn_relu
        if o['nn_mp2'] is not None:
            nn.max_pool2d = p_nn_mp2
        if o['nn_mp'] is not None:
            nn.max_pool = p_nn_mp
        if o['k_relu'] is not None:
            K.relu = p_k_relu
        if o['k_pool'] is not None:
            K.pool2d = p_k_pool
        yield
    finally:
        if o['nn_relu'] is not None:
            nn.relu = o['nn_relu']
        if o['nn_mp2'] is not None:
            nn.max_pool2d = o['nn_mp2']
        if o['nn_mp'] is not None:
            nn.max_pool = o['nn_mp']
        if o['k_relu'] is not None:
            K.relu = o['k_relu']
        if o['k_pool'] is not None:
            K.pool2d = o['k_pool']


def bpa_cnn(model, x, tgt_emb, attack_type):
    """BPA applied on a Scale-Invariant Nesterov (SI-NI) base.

    BPA is a backward-pass modification (smoothed ReLU/MaxPool gradients) that
    stacks on top of any gradient attack; the original paper combines it with
    strong transfer methods rather than plain MI-FGSM. We therefore run the
    strongest baseline here -- SI-NI-FGSM (Nesterov lookahead + scale-invariant
    gradient averaging over 5 scaled copies) -- with every forward/backward pass
    executed under bpa_backward_patch(). Iteration / query budget is identical to
    SI_NI_FGSM (NUM_ITER x 5 scales), so the only difference vs SI_NI_FGSM is the
    BPA backward; compare the two to isolate BPA's contribution.
    """
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
    for _ in range(NUM_ITER):
        nes = adv + DECAY * alpha * g                 # Nesterov lookahead
        grad_sum = tf.zeros_like(x)
        with bpa_backward_patch():                    # BPA surrogate backward
            for s in scales:                          # scale-invariance
                with tf.GradientTape() as tape:
                    tape.watch(nes)
                    emb = compute_embedding(model, nes * s)
                    cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                    loss = attack_loss(cos, attack_type)
                grad_sum += tape.gradient(loss, nes)
        grad = grad_sum / len(scales)
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))  # NaN guard
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
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
    if attack_name == 'BPA_CNN':
        return bpa_cnn(model, src, tgt_emb, attack_type)
    raise ValueError(f'Unsupported attack: {attack_name}')
