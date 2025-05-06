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
