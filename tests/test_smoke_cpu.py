"""Offline CPU smoke test: a tiny ViT learns synthetic colored shapes.

Mirrors scripts/smoke_cpu.py but shorter, and asserts the two learning signals
so `pytest` alone proves the training stack runs end-to-end on a CPU with a
single-device (data, model) mesh and NamedSharding specs.
"""
from __future__ import annotations

import os
from functools import partial

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from src.data.synthetic_shapes import make_dataset, batches, NUM_SHAPE_CLASSES
from src.model.vit_flax import ViT, ViTConfig
from src.parallelism.mesh import MeshConfig, build_mesh, validate_axes
from src.parallelism.sharding import batch_sharding, replicated
from src.training.optimizer import build_optimizer
from src.training.train_jax import train_step, eval_step


def _tiny_vit(rngs: nnx.Rngs, num_classes: int) -> ViT:
    cfg = ViTConfig(
        image_size=32, patch_size=8, num_classes=num_classes,
        hidden_size=64, num_layers=2, num_heads=4, mlp_dim=128,
        dropout_rate=0.0, attention_dropout_rate=0.0,
    )
    return ViT(cfg, rngs=rngs)


def test_single_device_mesh_builds_on_cpu():
    mesh = build_mesh(MeshConfig(data_parallel=-1, model_parallel=1))
    validate_axes(mesh)
    assert mesh.shape["data"] == jax.device_count()
    assert mesh.shape["model"] == 1


def test_synthetic_shapes_shapes_and_labels():
    imgs, lbls = make_dataset(40, image_size=32, num_classes=4, seed=0)
    assert imgs.shape == (40, 32, 32, 3)
    assert imgs.dtype == np.float32
    assert lbls.min() >= 0 and lbls.max() < 4
    assert set(np.unique(lbls)).issubset(set(range(4)))


def test_tiny_vit_trains_on_cpu():
    num_classes = NUM_SHAPE_CLASSES
    batch_size, num_steps, seed = 32, 40, 0

    key = jax.random.key(seed)
    rngs = nnx.Rngs(params=key, dropout=jax.random.fold_in(key, 1))
    model = _tiny_vit(rngs, num_classes)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    tx = build_optimizer(
        {"base_lr": 3e-3, "weight_decay": 0.02, "warmup_steps": 8, "grad_clip": 1.0},
        total_steps=num_steps,
    )
    opt_state = tx.init(params)

    train_imgs, train_lbls = make_dataset(batch_size * num_steps, image_size=32,
                                          num_classes=num_classes, seed=seed)
    eval_imgs, eval_lbls = make_dataset(256, image_size=32,
                                        num_classes=num_classes, seed=99)

    mesh = build_mesh(MeshConfig(data_parallel=-1, model_parallel=1))
    with mesh:
        rep = replicated(mesh)
        bshard = batch_sharding(mesh)
        lbl_shard = NamedSharding(mesh, P("data"))
        params = jax.tree_util.tree_map(lambda x: jax.device_put(x, rep), params)
        opt_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, rep), opt_state)

        batch_shard = {"image": bshard, "label": lbl_shard}
        jitted_train = jax.jit(
            partial(train_step, graphdef, tx, rest, grad_accum_steps=1,
                    num_classes=num_classes, smoothing=0.1),
            in_shardings=(rep, rep, batch_shard, None),
            out_shardings=(rep, rep, None, None),
        )
        jitted_eval = jax.jit(
            partial(eval_step, graphdef, rest, num_classes=num_classes),
            in_shardings=(rep, batch_shard),
            out_shardings=(None, None),
        )

        def acc() -> float:
            imgs = jax.device_put(jnp.asarray(eval_imgs), bshard)
            lbls = jax.device_put(jnp.asarray(eval_lbls), lbl_shard)
            _, a = jitted_eval(params, {"image": imgs, "label": lbls})
            return float(a)

        acc_before = acc()
        losses = []
        for imgs_np, lbls_np in batches(train_imgs, train_lbls, batch_size,
                                        seed=seed, shuffle=True):
            imgs = jax.device_put(jnp.asarray(imgs_np), bshard)
            lbls = jax.device_put(jnp.asarray(lbls_np), lbl_shard)
            key, sk = jax.random.split(key)
            params, opt_state, loss, _ = jitted_train(
                params, opt_state, {"image": imgs, "label": lbls}, sk
            )
            losses.append(float(loss))
        acc_after = acc()

    chance = 1.0 / num_classes
    w = max(1, len(losses) // 5)
    first = sum(losses[:w]) / w
    last = sum(losses[-w:]) / w

    assert last < first, f"loss did not drop: first={first:.4f} last={last:.4f}"
    assert acc_after > chance + 0.10, (
        f"top-1 {acc_after:.3f} not above chance {chance:.3f}"
    )
    assert acc_after > acc_before, (
        f"top-1 did not improve: before={acc_before:.3f} after={acc_after:.3f}"
    )
