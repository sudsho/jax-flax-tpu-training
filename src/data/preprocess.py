"""Image preprocessing: resize, crop, RandAugment, mixup, normalize.

RandAugment is implemented via tf.image ops. It's not identical to the
timm python-side impl but close enough for ViT reproductions.
"""
from __future__ import annotations

import tensorflow as tf


IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def resize_and_random_crop(img: tf.Tensor, size: int = 224) -> tf.Tensor:
    img = tf.image.resize(img, (int(size * 1.15), int(size * 1.15)))
    img = tf.image.random_crop(img, (size, size, 3))
    img = tf.image.random_flip_left_right(img)
    return img


def center_crop(img: tf.Tensor, size: int = 224) -> tf.Tensor:
    img = tf.image.resize(img, (int(size * 1.15), int(size * 1.15)))
    h, w = int(size * 1.15), int(size * 1.15)
    off_h = (h - size) // 2
    off_w = (w - size) // 2
    return img[off_h : off_h + size, off_w : off_w + size, :]


def normalize(img: tf.Tensor) -> tf.Tensor:
    return (img - IMAGENET_MEAN) / IMAGENET_STD


# ---- RandAugment (subset of the timm ops) ---------------------------------
_RAND_OPS = (
    "identity",
    "auto_contrast",
    "equalize",
    "rotate",
    "solarize",
    "color",
    "posterize",
    "contrast",
    "brightness",
    "sharpness",
    "shear_x",
    "shear_y",
    "translate_x",
    "translate_y",
)


def _apply_rand_op(img: tf.Tensor, op: str, mag: tf.Tensor) -> tf.Tensor:
    if op == "identity":
        return img
    if op == "auto_contrast":
        return tf.image.adjust_contrast(img, 1.0 + (mag - 5.0) * 0.1)
    if op == "equalize":
        # tf lacks a batched equalize; approximate with hist stretch
        lo = tf.reduce_min(img)
        hi = tf.reduce_max(img)
        return (img - lo) / tf.maximum(hi - lo, 1e-6)
    if op == "rotate":
        return tf.image.rot90(img, k=tf.cast(mag % 4, tf.int32))
    if op == "solarize":
        thresh = 1.0 - mag / 10.0
        return tf.where(img > thresh, 1.0 - img, img)
    if op == "color":
        return tf.image.adjust_saturation(img, 0.1 + mag / 10.0)
    if op == "posterize":
        bits = tf.cast(8 - tf.round(mag * 0.6), tf.uint8)
        scale = tf.cast(2**bits, tf.float32)
        return tf.round(img * scale) / scale
    if op == "contrast":
        return tf.image.adjust_contrast(img, 0.5 + mag / 5.0)
    if op == "brightness":
        return tf.image.adjust_brightness(img, (mag - 5.0) * 0.05)
    if op == "sharpness":
        return tf.image.adjust_contrast(img, 0.8 + mag / 20.0)
    # shears / translates approximated as small random pad+crop
    if op.startswith("shear") or op.startswith("translate"):
        pad = tf.cast(mag, tf.int32)
        img_p = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
        return tf.image.random_crop(img_p, tf.shape(img))
    return img


def rand_augment(img: tf.Tensor, num_ops: int = 2, magnitude: float = 9.0) -> tf.Tensor:
    """Apply `num_ops` random ops with a fixed magnitude."""
    for _ in range(num_ops):
        idx = tf.random.uniform((), 0, len(_RAND_OPS), dtype=tf.int32)
        mag = tf.constant(magnitude, dtype=tf.float32)
        for i, op in enumerate(_RAND_OPS):
            img = tf.cond(tf.equal(idx, i), lambda o=op: _apply_rand_op(img, o, mag), lambda: img)
    return tf.clip_by_value(img, 0.0, 1.0)


# ---- mixup ---------------------------------------------------------------
def mixup(
    images: tf.Tensor,
    labels: tf.Tensor,
    num_classes: int,
    alpha: float = 0.2,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Beta(alpha, alpha) mixup on a batched (images, labels)."""
    lam = tf.compat.v1.distributions.Beta(alpha, alpha).sample()
    idx = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    mixed = lam * images + (1.0 - lam) * tf.gather(images, idx)
    y = tf.one_hot(labels, num_classes)
    y_shuf = tf.gather(y, idx)
    mixed_y = lam * y + (1.0 - lam) * y_shuf
    return mixed, mixed_y


def train_preprocess(num_classes: int = 10, size: int = 224):
    def _fn(img, label):
        img = resize_and_random_crop(img, size)
        img = rand_augment(img)
        img = normalize(img)
        return img, label

    return _fn


def val_preprocess(size: int = 224):
    def _fn(img, label):
        img = center_crop(img, size)
        img = normalize(img)
        return img, label

    return _fn
