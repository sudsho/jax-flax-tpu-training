"""Offline CPU smoke for the JAX/Flax ViT training stack.

Runs a TINY ViT on a small synthetic colored-shapes classification set for a
handful of steps on the CPU. No TPU, no GCS, no tensorflow, no downloads.

What it exercises (the real code paths, just shrunk):
  - the Flax NNX ViT from src/model/vit_flax.py (tiny config)
  - the optax AdamW + weight-decay-mask + cosine schedule from
    src/training/optimizer.py
  - the reusable train_step / eval_step from src/training/train_jax.py
  - a single-device (data=1, model=1) mesh from src/parallelism/mesh.py and the
    NamedSharding batch / replicated specs from src/parallelism/sharding.py, so
    the pjit in_shardings / out_shardings code path runs on one CPU device

Success criteria (asserted at the end, exit code != 0 on failure):
  - training loss decreases from the first window to the last
  - final top-1 on a held-out synthetic eval set is above chance (1/num_classes)

Run:
    python scripts/smoke_cpu.py
"""
from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

# Keep JAX on CPU and single-threaded-ish for a deterministic, quick smoke.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.data.synthetic_shapes import make_dataset, batches, NUM_SHAPE_CLASSES
from src.model.vit_flax import ViT, ViTConfig
from src.parallelism.mesh import MeshConfig, build_mesh, describe, validate_axes
from src.parallelism.sharding import batch_sharding, replicated
from src.training.optimizer import build_optimizer, EMA, flatten_grads_norm
from src.training.train_jax import train_step, eval_step


IMAGE_SIZE = 32
PATCH_SIZE = 8
NUM_CLASSES = NUM_SHAPE_CLASSES  # 4
BATCH_SIZE = 32
NUM_STEPS = 60
SEED = 0


def build_tiny_vit(rngs: nnx.Rngs) -> ViT:
    cfg = ViTConfig(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,   # 32/8 -> 16 patches
        num_classes=NUM_CLASSES,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.0,
        attention_dropout_rate=0.0,
    )
    return ViT(cfg, rngs=rngs)


def main() -> int:
    print(f"jax {jax.__version__} on {jax.devices()[0].platform} "
          f"({jax.device_count()} device)")

    # --- single-device mesh: exercises the (data, model) sharding path on CPU
    mesh = build_mesh(MeshConfig(data_parallel=-1, model_parallel=1))
    validate_axes(mesh)
    print(describe(mesh))

    key = jax.random.key(SEED)
    rngs = nnx.Rngs(params=key, dropout=jax.random.fold_in(key, 1))
    model = build_tiny_vit(rngs)
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    n_params = sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params))
    print(f"tiny ViT: {n_params:,} params  "
          f"(img={IMAGE_SIZE} patch={PATCH_SIZE} dim=64 layers=2 heads=4)")

    # --- optimizer: cosine + warmup + adamw with the WD mask
    tx = build_optimizer(
        {"base_lr": 3e-3, "weight_decay": 0.02, "warmup_steps": 10, "grad_clip": 1.0},
        total_steps=NUM_STEPS,
    )
    opt_state = tx.init(params)
    ema = EMA(decay=0.99)
    ema_state = ema.init(params)

    # --- synthetic data (pure numpy; no tfds / no download)
    train_imgs, train_lbls = make_dataset(BATCH_SIZE * NUM_STEPS, image_size=IMAGE_SIZE,
                                          num_classes=NUM_CLASSES, seed=SEED)
    eval_imgs, eval_lbls = make_dataset(256, image_size=IMAGE_SIZE,
                                        num_classes=NUM_CLASSES, seed=1234)

    with mesh:
        rep = replicated(mesh)
        bshard = batch_sharding(mesh)  # images: [B, H, W, C] with B over data
        lbl_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        # the batch is a dict, so in_shardings for it is a matching pytree
        batch_shard = {"image": bshard, "label": lbl_shard}
        params = jax.tree_util.tree_map(lambda x: jax.device_put(x, rep), params)
        opt_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, rep), opt_state)

        jitted_train = jax.jit(
            partial(train_step, graphdef, tx, rest, grad_accum_steps=1,
                    num_classes=NUM_CLASSES, smoothing=0.1),
            in_shardings=(rep, rep, batch_shard, None),
            out_shardings=(rep, rep, None, None),
        )
        jitted_eval = jax.jit(
            partial(eval_step, graphdef, rest, num_classes=NUM_CLASSES),
            in_shardings=(rep, batch_shard),
            out_shardings=(None, None),
        )

        def evaluate() -> tuple[float, float]:
            imgs = jax.device_put(jnp.asarray(eval_imgs), bshard)
            lbls = jax.device_put(jnp.asarray(eval_lbls), lbl_shard)
            loss, acc = jitted_eval(params, {"image": imgs, "label": lbls})
            return float(loss), float(acc)

        losses: list[float] = []
        data = list(batches(train_imgs, train_lbls, BATCH_SIZE, seed=SEED, shuffle=True))
        init_eval_loss, init_acc = evaluate()
        print(f"step {0:>3}  eval_loss {init_eval_loss:.4f}  top1 {init_acc:.3f}  "
              f"(chance {1.0 / NUM_CLASSES:.3f})")

        step = 0
        for imgs_np, lbls_np in data:
            imgs = jax.device_put(jnp.asarray(imgs_np), bshard)
            lbls = jax.device_put(
                jnp.asarray(lbls_np),
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")),
            )
            key, sk = jax.random.split(key)
            params, opt_state, loss, gnorm = jitted_train(
                params, opt_state, {"image": imgs, "label": lbls}, sk
            )
            ema_state = ema.update(ema_state, params, step)
            losses.append(float(loss))
            step += 1
            if step % 15 == 0 or step == 1:
                el, acc = evaluate()
                print(f"step {step:>3}  train_loss {float(loss):.4f}  "
                      f"gnorm {float(gnorm):.3f}  eval_loss {el:.4f}  top1 {acc:.3f}")

        final_loss, final_acc = evaluate()

    # --- windowed loss comparison (robust to per-step noise)
    w = max(1, len(losses) // 6)
    first = sum(losses[:w]) / w
    last = sum(losses[-w:]) / w
    chance = 1.0 / NUM_CLASSES
    print("-" * 56)
    print(f"first-{w} mean train_loss {first:.4f}  ->  last-{w} mean {last:.4f}")
    print(f"final top-1 {final_acc:.3f}  (chance {chance:.3f})")

    ok = True
    if not last < first:
        print("FAIL: training loss did not decrease")
        ok = False
    if not final_acc > chance + 0.10:
        print(f"FAIL: final top-1 {final_acc:.3f} not clearly above chance {chance:.3f}")
        ok = False
    if ok:
        print("SMOKE OK: loss decreased and top-1 climbed above chance on CPU")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
