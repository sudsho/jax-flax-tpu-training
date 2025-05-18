"""Throughput comparison harness.

Runs a fixed number of forward+backward steps with synthetic data (fixed
shapes so we don't measure the data pipeline) and reports images/sec.

For JAX we build the ViT-B/16, jit the step, and measure after a few warmup
steps to avoid the initial compilation cost.

For torch-xla we spawn `xmp.spawn` and measure similarly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from src.model.vit_flax import ViT, ViTConfig
from src.parallelism.mesh import build_mesh, MeshConfig
from src.parallelism.sharding import batch_sharding, replicated


def _make_jax_step(model, tx):
    graphdef, state = nnx.split(model)
    opt_state = tx.init(state)

    def loss_fn(m, x, y):
        logits = m(x, deterministic=False)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))

    @jax.jit
    def step(state, opt_state, x, y):
        m = nnx.merge(graphdef, state)
        loss, grads = nnx.value_and_grad(loss_fn)(m, x, y)
        updates, opt_state = tx.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss

    return step, state, opt_state


def bench_jax(
    *,
    batch_size: int,
    image_size: int = 224,
    num_classes: int = 10,
    warmup: int = 5,
    steps: int = 100,
    model_parallel: int = 1,
    seed: int = 0,
) -> dict:
    key = jax.random.key(seed)
    rngs = nnx.Rngs(params=key, dropout=jax.random.fold_in(key, 1))
    model = ViT(ViTConfig(image_size=image_size, num_classes=num_classes), rngs=rngs)
    tx = optax.adamw(1e-3, weight_decay=0.05)
    mesh = build_mesh(MeshConfig(model_parallel=model_parallel))
    step_fn, state, opt_state = _make_jax_step(model, tx)

    x = jnp.zeros((batch_size, image_size, image_size, 3), dtype=jnp.float32)
    y = jnp.zeros((batch_size,), dtype=jnp.int32)

    with mesh:
        bs = batch_sharding(mesh)
        rs = replicated(mesh)
        x = jax.device_put(x, bs)
        y = jax.device_put(y, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data")))
        state = jax.tree_util.tree_map(lambda a: jax.device_put(a, rs), state)
        opt_state = jax.tree_util.tree_map(lambda a: jax.device_put(a, rs), opt_state)
        for _ in range(warmup):
            state, opt_state, loss = step_fn(state, opt_state, x, y)
        loss.block_until_ready()
        t0 = time.time()
        for _ in range(steps):
            state, opt_state, loss = step_fn(state, opt_state, x, y)
        loss.block_until_ready()
        dt = time.time() - t0

    ips = steps * batch_size / dt
    return {"framework": "jax", "batch_size": batch_size, "steps": steps, "dt": dt, "ips": ips}


def bench_torch_xla(
    *,
    batch_size_per_core: int,
    num_cores: int = 8,
    image_size: int = 224,
    num_classes: int = 10,
    warmup: int = 5,
    steps: int = 100,
) -> dict:
    """Delegates to src.training.train_torch_xla via a subprocess to keep
    the JAX and torch-xla process spaces separate."""
    import subprocess, sys
    tmp = Path("outputs/torch_xla_bench.yaml")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(f"""\
model:
  image_size: {image_size}
  patch_size: 16
  num_classes: {num_classes}
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  mlp_dim: 3072
optim:
  base_lr: 1.0e-3
  weight_decay: 0.05
train:
  batch_size_per_core: {batch_size_per_core}
  bench_steps: {steps}
  log_every: 50
xla:
  num_cores: {num_cores}
""")
    t0 = time.time()
    subprocess.run(
        [sys.executable, "-m", "src.training.train_torch_xla", "--config", str(tmp)],
        check=True,
    )
    dt = time.time() - t0
    ips = steps * batch_size_per_core * num_cores / dt
    return {
        "framework": "torch-xla",
        "batch_size": batch_size_per_core * num_cores,
        "steps": steps,
        "dt": dt,
        "ips": ips,
    }


def main() -> None:
    p = argparse.ArgumentParser("throughput-bench")
    p.add_argument("--framework", choices=["jax", "torch-xla"], required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--batch-size-per-core", type=int, default=64)
    p.add_argument("--num-cores", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("benchmarks/last_run.json"))
    args = p.parse_args()

    if args.framework == "jax":
        res = bench_jax(batch_size=args.batch_size, steps=args.steps, warmup=args.warmup)
    else:
        res = bench_torch_xla(
            batch_size_per_core=args.batch_size_per_core,
            num_cores=args.num_cores,
            steps=args.steps,
            warmup=args.warmup,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
