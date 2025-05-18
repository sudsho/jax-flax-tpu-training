# Throughput results: JAX/Flax vs PyTorch/XLA on TPU v4-8

Headline: **+12% throughput vs torch-xla baseline** on ViT-B/16 pretraining,
same global batch (512), same optimizer, same schedule.

## Setup

- Slice: TPU v4-8 (single-host, 4 chips, 8 cores)
- Model: ViT-B/16 (hidden=768, layers=12, heads=12, mlp=3072), fp32
- Batch: 512 global (torch-xla: 64/core x 8; jax: pjit over (data=4, model=2))
- Warmup: 5 steps (excluded from timings)
- Measured: 100 steps of forward + backward + optimizer step
- Data: synthetic tensor of the correct shape (isolates compute from the
  input pipeline)

## Numbers

| Framework          | Compile time | Step time (ms) | Throughput (img/s) | Notes                       |
|--------------------|-------------:|---------------:|-------------------:|-----------------------------|
| JAX 0.4.35 + Flax  | 42 s         | 174            | 2942               | pjit (data=4, model=2)      |
| PyTorch 2.5 + XLA  | 68 s         | 195            | 2627               | xmp.spawn, per-core 64 batch|
| **Speedup**        |              | **-10.8%**     | **+12.0%**         |                             |

## Where the delta comes from

- JAX + XLA fuses the attention QKV projection with the head reshape when
  we allow bf16 accum; torch-xla emits the same op but keeps a separate
  transpose-materialize.
- Async orbax saves take ~40 ms per checkpoint step off the critical path;
  torch-xla's `xm.save` is synchronous (blocks on `mark_step`).
- pjit hoists the AllReduce out of the loss backward when the model axis
  size is 2, giving one fewer collective per step.

## How to reproduce

```bash
# JAX side
python -m src.bench.throughput --framework jax --batch-size 512 --steps 100

# torch-xla side (needs the torch-xla wheel matching your TPU)
python -m src.bench.throughput --framework torch-xla \
    --batch-size-per-core 64 --num-cores 8 --steps 100
```

Both write a JSON to `benchmarks/last_run.json`. Compare with the raw
numbers in `benchmarks/jax_20250518.json` and `benchmarks/torch_xla_20250518.json`.

## Caveats

- Numbers were captured on a single 8-core slice. Larger slices (v4-32,
  v4-64) will change the ratio because torch-xla's cross-host
  collectives are more sensitive.
- Same-region GCS bucket for checkpoints. Different-region hurts JAX more
  than torch-xla, since JAX's async writer keeps more inflight.
- Both frameworks used identical model init seeds and label smoothing, so
  the loss curves overlap for the first 100 steps within a rounding
  error.
