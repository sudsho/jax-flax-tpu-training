"""Imagenette via tensorflow-datasets.

Imagenette is a 10-class subset of ImageNet (n=~13k images). Small enough
to iterate locally, big enough to make TPU throughput measurable.

For the ImageNet-100 setting the loader signature stays identical, only the
tfds builder name and the num_classes differ - see the note in README.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


IMAGENETTE_NUM_CLASSES = 10
IMAGENETTE_TRAIN_SIZE = 9469
IMAGENETTE_VAL_SIZE = 3925


def _decode(example: dict) -> tuple[tf.Tensor, tf.Tensor]:
    img = tf.cast(example["image"], tf.float32) / 255.0
    label = tf.cast(example["label"], tf.int32)
    return img, label


def load_imagenette(
    split: str,
    *,
    data_dir: str | None = None,
    preprocess_fn: Callable | None = None,
    batch_size: int = 256,
    shuffle: bool = True,
    seed: int = 0,
    drop_remainder: bool = True,
) -> tf.data.Dataset:
    """Returns a tf.data.Dataset yielding (images, labels) numpy-ready batches."""
    ds = tfds.load(
        "imagenette/320px-v2",
        split=split,
        data_dir=data_dir,
        as_supervised=False,
        shuffle_files=shuffle,
    )
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    if preprocess_fn is not None:
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(4096, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def as_numpy_iterator(ds: tf.data.Dataset) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for batch in tfds.as_numpy(ds):
        yield batch
