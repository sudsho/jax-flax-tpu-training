# jax-flax-tpu-training

Study scaffold for pretraining a ViT in JAX + Flax NNX on a TPU v4 slice.
The repo collects the pieces of a modern JAX training stack (2D device mesh,
NamedSharding rules, optax with a weight-decay mask, an orbax checkpoint
helper, a wandb wrapper) as standalone modules and a PyTorch/XLA baseline
scaffold.

The full TPU headline (ViT-B/16 on imagenette across a v4-8, real throughput)
needs a TPU and tensorflow-datasets. To make the code actually runnable and
testable without any of that, the repo ships a tiny CPU smoke: a small ViT on a
synthetic colored-shapes classification set, trained for a handful of steps on a
single-device mesh so the same mesh, NamedSharding, optax, and jit code paths run
end-to-end on a laptop CPU with no TPU, no GCS, no tensorflow, and no downloads.

## Quick start (runs offline)

Needs the JAX CPU stack only (`jax[cpu]`, `flax`, `optax`). No TPU, no network,
no dataset download. A repo-local venv keeps it off your global Python:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install "jax[cpu]==0.4.35" flax==0.10.2 optax==0.2.4 orbax-checkpoint==0.10.1 pyyaml pytest
# (on Linux/mac use .venv/bin/python)

.venv/Scripts/python scripts/smoke_cpu.py
```

Real output from a CPU run:

```
jax 0.4.35 on cpu (1 device)
Mesh(data=1, model=1) on 1 devices (cpu)
tiny ViT: 80,836 params  (img=32 patch=8 dim=64 layers=2 heads=4)
step   0  eval_loss 1.6879  top1 0.293  (chance 0.250)
step   1  train_loss 1.9272  gnorm 20.545  eval_loss 1.6879  top1 0.293
step  15  train_loss 0.9864  gnorm 5.099  eval_loss 0.7490  top1 0.742
step  30  train_loss 0.4221  gnorm 1.746  eval_loss 0.3327  top1 0.922
step  45  train_loss 0.5481  gnorm 2.467  eval_loss 0.2745  top1 0.938
step  60  train_loss 0.3912  gnorm 0.535  eval_loss 0.1919  top1 0.957
--------------------------------------------------------
first-10 mean train_loss 1.5087  ->  last-10 mean 0.4303
final top-1 0.957  (chance 0.250)
SMOKE OK: loss decreased and top-1 climbed above chance on CPU
```

The tiny ViT drives training loss from 1.51 down to 0.43 and top-1 from chance
(0.25) up to 0.96 in 60 CPU steps. The synthetic dataset lives in
`src/data/synthetic_shapes.py` (numpy only).

Tests (25, all CPU):

```bash
.venv/Scripts/python -m pytest -q
# 25 passed
```

`tests/test_smoke_cpu.py` runs the tiny-ViT training and asserts loss drops and
top-1 climbs above chance; the rest cover the model shapes, optimizer/schedule,
sharding specs, and an orbax save/restore round-trip to a local dir.

## Status

- The tiny CPU smoke (`scripts/smoke_cpu.py`) and the 25 unit tests run offline
  on a plain CPU and are verified green.
- The reusable `train_step` / `eval_step` in `src/training/train_jax.py` are the
  same ones the CPU smoke drives; the only TPU-headline-specific piece the smoke
  swaps out is the data source (synthetic shapes instead of imagenette via tfds)
  and the mesh size (one CPU device instead of a v4-8).
- What the headline still needs a TPU for: the ViT-B/16 config, the real
  imagenette pipeline (`src/data/imagenette_tfds.py`, imports tensorflow), the
  multi-device `(data=4, model=2)` mesh, orbax checkpoints to `gs://`, and any
  throughput numbers. Those paths are present but not exercised by the CPU smoke.
- No large-scale accuracy or throughput numbers are claimed here. The only
  numbers in this repo are the tiny CPU smoke output above.

## What was fixed to make it run

Three integration bugs in the original scaffold blocked the loop from running;
all three are fixed and covered by the smoke and tests:

- The Flax NNX `MultiHeadAttention` call in `EncoderBlock` needed an explicit
  `decode=False`.
- `train_step` had `tx` (a partial-bound argument) sitting between the traced
  call arguments, so `jax.jit` + `functools.partial` collided. The signature now
  leads with the partial-bound args.
- The batch `in_shardings` applied a rank-4 image spec to the rank-1 label array.
  Batches now carry a per-leaf sharding pytree. The optimizer also now trains the
  `nnx.Param` leaves only (params split from dropout rng state), which is what
  lets `optax` update cleanly.

## Why this repo

Vision Transformers are a compact stress test for a TPU training stack:
dense attention, big MLPs, and enough parameters that both tensor-parallel
and data-parallel sharding matter. This repo sketches the pieces you would
assemble on a v4 slice:

- `jax.Array` with `NamedSharding` over a 2D `(data, model)` mesh
- `pjit`-style `jax.jit` with `in_shardings` / `out_shardings`
- `flax.nnx` split/merge for jit-friendly stateful modules
- `optax.adamw` with a weight-decay mask that skips LayerNorm/bias/pos-embed
- Cosine schedule with linear warmup and EMA helpers
- An orbax-checkpoint helper with keep-latest-N
- A wandb wrapper with a no-op fallback

## Repo layout

```
src/
  model/
    vit_flax.py               # ViT in Flax NNX (tiny + B/16 configs)
    vit_torch_baseline.py     # matched PyTorch impl scaffold
  data/
    synthetic_shapes.py       # numpy colored-shapes set for the CPU smoke
    imagenette_tfds.py        # tfds loader for the TPU headline (imports tensorflow)
    preprocess.py             # resize/crop/randaugment/mixup/normalize helpers
  parallelism/
    mesh.py                   # 2D (data, model) device mesh builder
    sharding.py               # NamedSharding rules for params + batches
  training/
    train_jax.py              # JAX loop (reusable train_step/eval_step; tfds import is lazy)
    train_torch_xla.py        # PyTorch/XLA loop scaffold
    optimizer.py              # adamw + WD mask + cosine + EMA
  checkpoint/
    orbax_ckpt.py             # standalone orbax helper (not wired into training)
  logging/
    wandb_hook.py             # wandb wrapper (not wired into training)
  eval/
    topk_eval.py              # standalone eval helper (not wired)
  bench/
    throughput.py             # harness stub; see caveats below
configs/
  jax_tpu_v4_8.yaml
  torch_xla_v4_8.yaml
  single_host_cpu.yaml
scripts/
  smoke_cpu.py                # offline tiny-ViT CPU smoke (no TPU/GCS/tfds)
  setup_tpu_vm.sh
  train_jax.sh
  train_torch_xla.sh
  run_bench.sh
  download_imagenette.sh
docs/
  jax_primer.md
  mesh_and_sharding.md
  gcs_checkpoint_setup.md
  tpu_pod_slice_notes.md
tests/
notebooks/attention_viz.ipynb   # exploratory scaffold; will not run as-is
```

## Mesh and sharding

The intended mesh on a v4-8 is 2D over `(data, model)`. See
[docs/mesh_and_sharding.md](docs/mesh_and_sharding.md) for the layout table
and the `param_sharding_rules(mesh)` dict in `src/parallelism/sharding.py`.
Note that the current `train_jax.py` scaffold replicates every parameter
and does not consume `param_sharding_rules`, so no tensor parallelism is
actually applied in the shipped code.

## Benchmark harness

`src/bench/throughput.py` sketches a JAX side (warmup + timed step loop
that excludes compile time) and a torch-xla side (subprocess wall clock
around interpreter launch, model construction, and XLA compile). The two
paths do not measure the same thing and were never used to produce a
like-for-like comparison. No benchmark numbers are quoted in this repo.

## GCS checkpoint recipe

`src/checkpoint/orbax_ckpt.py` and
[docs/gcs_checkpoint_setup.md](docs/gcs_checkpoint_setup.md) walk through
bucket setup, IAM, and lifecycle rules for orbax against `gs://`. The
helper is standalone; it is not currently invoked from the training loop.

## References

- Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words"
- Steiner et al. 2021, "How to train your ViT?"
- Xu et al. 2023, "GSPMD: General and Scalable Parallelization for ML"
- JAX Sharding: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- Orbax: https://orbax.readthedocs.io/en/latest/

## License

MIT. See [LICENSE](LICENSE).
