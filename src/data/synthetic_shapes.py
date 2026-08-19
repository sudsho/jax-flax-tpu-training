"""Synthetic colored-shapes image classification.

A dependency-free (numpy only) stand-in for imagenette so the training loop
can be exercised end-to-end on a CPU with no tensorflow, no tfds download,
and no network. Each image is a small RGB canvas with one shape drawn on it;
the label is the shape class. Shapes are drawn in a randomized color and
position with a bit of pixel noise so the task is learnable but not trivial.

Classes (num_classes=4):
    0 = filled square
    1 = filled disc
    2 = horizontal bar
    3 = vertical bar

The signal (shape geometry) is independent of color and position, so a small
ViT has to actually attend to structure rather than memorize a pixel.
"""
from __future__ import annotations

import numpy as np

SHAPE_NAMES = ("square", "disc", "hbar", "vbar")
NUM_SHAPE_CLASSES = len(SHAPE_NAMES)


def _draw_one(rng: np.random.Generator, size: int, label: int) -> np.ndarray:
    """Return an [size, size, 3] float32 image in [0, 1] for the given class."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    # random foreground color, kept reasonably saturated
    color = rng.uniform(0.4, 1.0, size=3).astype(np.float32)

    # a random extent for the shape, leaving a margin
    lo = size // 6
    hi = size - size // 6
    if label == 0:  # filled square
        y0 = rng.integers(0, lo + 1)
        x0 = rng.integers(0, lo + 1)
        side = rng.integers(size // 3, hi - lo + 1)
        y1 = min(size, y0 + side)
        x1 = min(size, x0 + side)
        img[y0:y1, x0:x1, :] = color
    elif label == 1:  # filled disc
        r = rng.integers(size // 6, size // 3 + 1)
        cy = rng.integers(r, size - r + 1)
        cx = rng.integers(r, size - r + 1)
        yy, xx = np.ogrid[:size, :size]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = color
    elif label == 2:  # horizontal bar
        h = rng.integers(size // 8, size // 4 + 1)
        y0 = rng.integers(0, size - h + 1)
        img[y0:y0 + h, :, :] = color
    else:  # vertical bar
        w = rng.integers(size // 8, size // 4 + 1)
        x0 = rng.integers(0, size - w + 1)
        img[:, x0:x0 + w, :] = color

    # light pixel noise
    img = img + rng.normal(0.0, 0.03, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def make_dataset(
    n: int,
    *,
    image_size: int = 32,
    num_classes: int = NUM_SHAPE_CLASSES,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (images, labels).

    images: [n, image_size, image_size, 3] float32 in [0, 1]
    labels: [n] int32 in [0, num_classes)
    """
    if num_classes > NUM_SHAPE_CLASSES:
        raise ValueError(
            f"synthetic_shapes has {NUM_SHAPE_CLASSES} shape classes; "
            f"asked for {num_classes}"
        )
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n).astype(np.int32)
    images = np.stack(
        [_draw_one(rng, image_size, int(y)) for y in labels], axis=0
    ).astype(np.float32)
    return images, labels


def batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
):
    """Yield (image_batch, label_batch) numpy tuples, dropping the remainder."""
    n = images.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n - batch_size + 1, batch_size):
        sel = idx[start:start + batch_size]
        yield images[sel], labels[sel]
