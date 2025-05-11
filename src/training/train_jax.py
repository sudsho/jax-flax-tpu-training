"""Main JAX training loop for ViT-B/16 on TPU.

Puts together: mesh + NamedSharding for params & batches, optax with WD mask,
cosine schedule, EMA, gradient accumulation, orbax async checkpoints, wandb.
"""
from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from src.data.imagenette_tfds import (
    IMAGENETTE_NUM_CLASSES,
    load_imagenette,
    steps_per_epoch,
    as_numpy_iterator,
)
from src.data.preprocess import train_preprocess, val_preprocess
from src.model.vit_flax import ViT, ViTConfig
from src.parallelism.mesh import AXIS_DATA, build_mesh, describe, MeshConfig
from src.parallelism.sharding import replicated, batch_sharding
from src.training.optimizer import build_optimizer, EMA, flatten_grads_norm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("jax-vit-train")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true")
    return p.parse_args(argv)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, rngs: nnx.Rngs) -> ViT:
    m = cfg["model"]
    vcfg = ViTConfig(
        image_size=m["image_size"],
        patch_size=m["patch_size"],
        num_classes=m["num_classes"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        mlp_dim=m["mlp_dim"],
        dropout_rate=m.get("dropout_rate", 0.0),
    )
    return ViT(vcfg, rngs=rngs)


def cross_entropy_with_smoothing(
    logits: jax.Array, labels: jax.Array, num_classes: int, smoothing: float = 0.1
) -> jax.Array:
    y = jax.nn.one_hot(labels, num_classes)
    y = y * (1.0 - smoothing) + smoothing / num_classes
    logp = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(y * logp, axis=-1))


def train_step(
    graphdef,
    state,
    opt_state,
    tx,
    batch,
    key,
    grad_accum_steps: int,
    num_classes: int,
    smoothing: float,
):
    model = nnx.merge(graphdef, state)
    dropout_key = jax.random.fold_in(key, 0)

    def _loss(m, imgs, labels):
        logits = m(imgs, deterministic=False)
        return cross_entropy_with_smoothing(logits, labels, num_classes, smoothing)

    if grad_accum_steps > 1:
        imgs = batch["image"].reshape((grad_accum_steps, -1, *batch["image"].shape[1:]))
        labels = batch["label"].reshape((grad_accum_steps, -1))

        def scan_step(carry, micro):
            g_acc, l_acc = carry
            mi, ml = micro
            l, g = nnx.value_and_grad(_loss)(model, mi, ml)
            g_acc = jax.tree_util.tree_map(lambda a, b: a + b, g_acc, g)
            return (g_acc, l_acc + l), None

        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, nnx.state(model))
        (grads, loss_sum), _ = jax.lax.scan(
            scan_step, (zero_grads, 0.0), (imgs, labels)
        )
        loss = loss_sum / grad_accum_steps
        grads = jax.tree_util.tree_map(lambda g: g / grad_accum_steps, grads)
    else:
        loss, grads = nnx.value_and_grad(_loss)(
            model, batch["image"], batch["label"]
        )

    updates, new_opt = tx.update(grads, opt_state, nnx.state(model))
    new_state = optax.apply_updates(nnx.state(model), updates)
    _ = dropout_key  # currently unused; wire when we add stochastic depth
    g_norm = flatten_grads_norm(grads)
    return new_state, new_opt, loss, g_norm


def eval_step(graphdef, state, batch, num_classes: int):
    model = nnx.merge(graphdef, state)
    logits = model(batch["image"], deterministic=True)
    pred = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(pred == batch["label"])
    loss = cross_entropy_with_smoothing(logits, batch["label"], num_classes, 0.0)
    return loss, acc


def _prepare_batch(batch, sharding):
    imgs, labels = batch
    imgs = jnp.asarray(imgs)
    labels = jnp.asarray(labels)
    return {
        "image": jax.device_put(imgs, sharding),
        "label": jax.device_put(labels, sharding.mesh and NamedSharding(
            sharding.mesh, P(AXIS_DATA)
        )),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)

    mesh = build_mesh(MeshConfig(
        data_parallel=cfg["mesh"].get("data_parallel", -1),
        model_parallel=cfg["mesh"].get("model_parallel", 1),
    ))
    print(describe(mesh))

    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(params=key, dropout=jax.random.fold_in(key, 1))
    model = build_model(cfg, rngs)

    steps_total = cfg["train"]["num_epochs"] * steps_per_epoch(
        cfg["train"]["batch_size"], split="train"
    )
    tx = build_optimizer(cfg["optim"], steps_total)
    graphdef, state = nnx.split(model)
    opt_state = tx.init(state)

    ema = EMA(decay=cfg["optim"].get("ema_decay", 0.9999))
    ema_state = ema.init(state)

    with mesh:
        params_sharding = replicated(mesh)
        batch_sh = batch_sharding(mesh)
        state = jax.tree_util.tree_map(lambda x: jax.device_put(x, params_sharding), state)
        opt_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, params_sharding), opt_state)

        jitted_train = jax.jit(
            partial(
                train_step,
                graphdef,
                tx=tx,
                grad_accum_steps=cfg["train"].get("grad_accum_steps", 1),
                num_classes=cfg["model"]["num_classes"],
                smoothing=cfg["train"].get("label_smoothing", 0.1),
            ),
            in_shardings=(params_sharding, params_sharding, batch_sh, None),
            out_shardings=(params_sharding, params_sharding, None, None),
        )

        jitted_eval = jax.jit(
            partial(eval_step, graphdef, num_classes=cfg["model"]["num_classes"]),
            in_shardings=(params_sharding, batch_sh),
            out_shardings=(None, None),
        )

        train_ds = load_imagenette(
            "train",
            data_dir=cfg["data"].get("data_dir"),
            preprocess_fn=train_preprocess(cfg["model"]["num_classes"]),
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            seed=args.seed,
        )

        step = 0
        t0 = time.time()
        for epoch in range(cfg["train"]["num_epochs"]):
            for raw in as_numpy_iterator(train_ds):
                batch = _prepare_batch(raw, batch_sh)
                key, sk = jax.random.split(key)
                state, opt_state, loss, gnorm = jitted_train(state, opt_state, batch, sk)
                ema_state = ema.update(ema_state, state, step)
                step += 1
                if step % cfg["train"].get("log_every", 50) == 0:
                    dt = time.time() - t0
                    ips = step * cfg["train"]["batch_size"] / dt
                    print(
                        f"step {step:>6}  loss {float(loss):.4f}  "
                        f"gnorm {float(gnorm):.3f}  ips {ips:.1f}"
                    )

    print("done")


if __name__ == "__main__":
    main()
