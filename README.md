# jax-flax-tpu-training

Study scaffold for pretraining a ViT in JAX + Flax NNX on a TPU v4 slice.
The repo collects the pieces of a modern JAX training stack (2D device mesh,
NamedSharding rules, optax with a weight-decay mask, an orbax checkpoint
helper, a wandb wrapper) as standalone modules and a PyTorch/XLA baseline
scaffold. Nothing here has been end-to-end trained or benchmarked; treat
it as reference layout rather than a runnable pipeline.

## Status

- Standalone modules (mesh builder, sharding rules, optimizer, orbax
  helper, wandb hook) work in isolation and have unit tests around them.
- The `train_jax.py` glue is not currently runnable end-to-end: the
  `jit` argument wiring, the batch sharding spec, and the model's
  attention call have unresolved integration bugs, and orbax and wandb
  are not wired into the loop.
- No training run, no accuracy numbers, and no throughput numbers were
  captured in this repo.

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
    vit_flax.py               # ViT in Flax NNX (integration bug: EncoderBlock
                              # calls self.attn without decode=False)
    vit_torch_baseline.py     # matched PyTorch impl scaffold
  data/
    imagenette_tfds.py        # tfds loader (imports tensorflow at top level)
    preprocess.py             # resize/crop/randaugment/mixup/normalize helpers
  parallelism/
    mesh.py                   # 2D (data, model) device mesh builder
    sharding.py               # NamedSharding rules for params + batches
  training/
    train_jax.py              # JAX loop scaffold (not runnable end-to-end)
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
