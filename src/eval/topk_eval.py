"""Top-1 / Top-5 evaluation on Imagenette val."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from src.data.imagenette_tfds import (
    IMAGENETTE_NUM_CLASSES,
    load_imagenette,
    as_numpy_iterator,
)
from src.data.preprocess import val_preprocess


def topk_accuracy(logits: jax.Array, labels: jax.Array, k: int) -> jax.Array:
    top = jax.lax.top_k(logits, k)[1]
    match = jnp.any(top == labels[:, None], axis=-1)
    return jnp.mean(match.astype(jnp.float32))


@partial(jax.jit, static_argnums=(0,))
def _eval_batch(graphdef, state, images, labels):
    model = nnx.merge(graphdef, state)
    logits = model(images, deterministic=True)
    return topk_accuracy(logits, labels, 1), topk_accuracy(logits, labels, 5)


def evaluate(
    graphdef,
    state,
    *,
    batch_size: int = 128,
    data_dir: str | None = None,
    num_classes: int = IMAGENETTE_NUM_CLASSES,
) -> dict:
    ds = load_imagenette(
        "validation",
        data_dir=data_dir,
        preprocess_fn=val_preprocess(),
        batch_size=batch_size,
        shuffle=False,
        drop_remainder=False,
    )
    top1s, top5s, n = 0.0, 0.0, 0
    for imgs, labels in as_numpy_iterator(ds):
        t1, t5 = _eval_batch(graphdef, state, jnp.asarray(imgs), jnp.asarray(labels))
        bs = imgs.shape[0]
        top1s += float(t1) * bs
        top5s += float(t5) * bs
        n += bs
    return {
        "top1": top1s / max(n, 1),
        "top5": top5s / max(n, 1),
        "num_examples": n,
        "num_classes": num_classes,
    }
