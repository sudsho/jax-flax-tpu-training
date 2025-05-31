# jax-flax-tpu-training

ViT-B/16 pretraining in JAX + Flax on TPU v4-8, with pjit mesh sharding
(data, model), async orbax checkpoints to Google Cloud Storage, and a
PyTorch/XLA baseline for a like-for-like throughput comparison.

**Headline result:** +12% throughput vs the torch-xla baseline on the
same v4-8 slice, same global batch, same optimizer, same schedule. See
[benchmarks/results.md](benchmarks/results.md).

## Why this repo

Vision Transformers are the canonical stress test for a TPU training
stack: dense attention, big MLPs, and enough parameters that both
tensor-parallel and data-parallel sharding matter. This repo reproduces ViT-B/16
in the modern Flax NNX API on a v4-8 slice is a compact way to
exercise:

- `jax.Array` with `NamedSharding` over a 2D `(data, model)` mesh
- `pjit`-style `jax.jit` with `in_shardings` / `out_shardings`
- `flax.nnx` split/merge for jit-friendly stateful modules
- `optax.adamw` with a weight-decay mask that skips LayerNorm/bias/pos-embed
- Cosine schedule with linear warmup, EMA of params, gradient accumulation
- Async `orbax-checkpoint` writing to `gs://` with keep-latest-N
- `wandb` logging with a no-op fallback

## Results

| Framework          | Slice | Global batch | Step time (ms) | Throughput (img/s) |
|--------------------|------:|-------------:|---------------:|-------------------:|
| JAX 0.4.35 + Flax  | v4-8  | 512          | 174            | **2942**           |
| PyTorch 2.5 + XLA  | v4-8  | 512          | 195            | 2627               |
| Delta              |       |              | -10.8%         | **+12.0%**         |

Raw JSON in `benchmarks/jax_20250518.json` and `benchmarks/torch_xla_20250518.json`.

## Repo layout

```
src/
  model/
    vit_flax.py               # ViT-B/16 in Flax NNX
    vit_torch_baseline.py     # matched PyTorch impl for torch-xla
  data/
    imagenette_tfds.py        # tfds loader
    preprocess.py             # resize/crop/randaugment/mixup/normalize
  parallelism/
    mesh.py                   # 2D (data, model) device mesh
    sharding.py               # NamedSharding rules for params + batches
  training/
    train_jax.py              # main JAX loop (jit + mesh + orbax + wandb)
    train_torch_xla.py        # baseline (xmp.spawn, xm.optimizer_step)
    optimizer.py              # adamw + WD mask + cosine/wsd + EMA
  checkpoint/
    orbax_ckpt.py             # async orbax to gs:// with keep-latest-N
  logging/
    wandb_hook.py             # wandb wrapper with silent fallback
  eval/
    topk_eval.py              # top-1 / top-5 on Imagenette val
  bench/
    throughput.py             # jax vs torch-xla harness
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
notebooks/attention_viz.ipynb
benchmarks/results.md
```

## Quick start (CPU smoke)

```bash
pip install -e ".[dev]"
bash scripts/download_imagenette.sh
python -m src.training.train_jax --config configs/single_host_cpu.yaml
```

Runs a ViT-S/16 for one epoch on Imagenette. Meant only to verify the
pipeline; useful step time only shows up on TPU.

## TPU v4-8 how-to

1. Provision a TPU-VM:
   ```bash
   export TPU_NAME=vit-jax-v4-8 ZONE=us-central2-b
   bash scripts/setup_tpu_vm.sh
   ```
2. Copy the repo up:
   ```bash
   gcloud compute tpus tpu-vm scp --recurse --zone $ZONE . \
       $TPU_NAME:~/jax-flax-tpu-training
   ```
3. Run the training loop:
   ```bash
   gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --command "
       source ~/venv/bin/activate
       cd ~/jax-flax-tpu-training
       bash scripts/train_jax.sh configs/jax_tpu_v4_8.yaml
   "
   ```
4. Follow along on wandb (`WANDB_PROJECT=jax-vit-tpu`).

## Mesh and sharding

The default mesh is `(data=4, model=2)` on a v4-8. Batches shard along
`data`; attention QKV/out projections and MLP fc1/fc2 shard their
hidden axis along `model`; everything else replicates. Full walkthrough
in [docs/mesh_and_sharding.md](docs/mesh_and_sharding.md).

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = np.asarray(jax.devices()).reshape(4, 2)
mesh = Mesh(devices, axis_names=("data", "model"))
batch_sh = NamedSharding(mesh, P("data", None, None, None))
mlp_fc1_sh = NamedSharding(mesh, P(None, "model"))
```

## GCS checkpoint recipe

```bash
gcloud storage buckets create gs://vit-jax-checkpoints \
    --location=us-central1 --uniform-bucket-level-access

gcloud storage buckets add-iam-policy-binding gs://vit-jax-checkpoints \
    --member=serviceAccount:service-${PROJECT_NUMBER}@cloud-tpus.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin
```

Then point `checkpoint.directory` in `configs/jax_tpu_v4_8.yaml` at
`gs://vit-jax-checkpoints/exp-tpu-v4-8`. Orbax writes asynchronously,
keeps the last 3 checkpoints, and prunes older ones automatically.
Full setup in [docs/gcs_checkpoint_setup.md](docs/gcs_checkpoint_setup.md).

## Running the benchmark

```bash
bash scripts/run_bench.sh
```

Writes JSON under `benchmarks/` and echoes the result. Update
`benchmarks/results.md` if you get different numbers on a different
slice.

## Kaggle / Colab TPU

Both notebook environments expose a v4-like slice via `jax.devices()`.
The training loop runs unchanged; you can skip
`scripts/setup_tpu_vm.sh` and install with:

```python
!pip install "jax[tpu]==0.4.35" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install flax==0.10.2 optax==0.2.4 orbax-checkpoint==0.10.1 \
             tensorflow-datasets==4.9.7 wandb==0.19.6 gcsfs==2025.2.0
```

## ImageNet-100 (or full ImageNet)

The loader shape is identical for larger splits. Point
`data.dataset` at `imagenet2012` (or a custom tfds builder) and bump
`model.num_classes` to 1000 / 100. On a v4-8 with `configs/jax_tpu_v4_8.yaml`
one epoch of ImageNet-1k takes roughly 8 min at bs=512.

## References

- Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words"
- Steiner et al. 2021, "How to train your ViT?"
- Xu et al. 2023, "GSPMD: General and Scalable Parallelization for ML"
- JAX Sharding: https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- Orbax: https://orbax.readthedocs.io/en/latest/

## Environment

Copy `.env.example` to `.env` and fill in your GCS bucket and (optional)
wandb key before running on a TPU-VM. See
[docs/tpu_pod_slice_notes.md](docs/tpu_pod_slice_notes.md) for
`JAX_COMPILATION_CACHE_DIR` and the `libtpu` vs `jax` version pitfall.

## License

MIT. See [LICENSE](LICENSE).
