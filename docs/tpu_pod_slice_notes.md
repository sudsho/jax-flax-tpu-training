# TPU pod slice notes

Notes I hit while getting this repo running on TPU v4 pod slices.

## Slice sizes and what changes

| Slice   | Hosts | Chips | Cores | Notes                                  |
|---------|------:|------:|------:|----------------------------------------|
| v4-8    | 1     | 4     | 8     | single-host, fits in one VM            |
| v4-32   | 4     | 16    | 32    | multi-host, needs `xmp.spawn`/`orbax`  |
| v4-64   | 8     | 32    | 64    | same code path as v4-32, larger mesh   |
| v4-128  | 16    | 64    | 128   | consider TPU Multislice                |

## Single vs multi host

JAX with pjit hides the host distinction as long as every host runs the
same Python script. On multi-host you launch the same script on every
worker (e.g. via `gcloud compute tpus tpu-vm ssh --worker=all
--command="python -m src.training.train_jax ..."`). Each worker sees a
subset of devices; jax's initializer stitches them into the global
mesh.

## Compilation time

For ViT-B/16 the first jit takes ~40 s on a fresh v4-8 with the default
`persistent_cache=None`. To reuse compiled binaries across runs, set:

```bash
export JAX_COMPILATION_CACHE_DIR=$HOME/.jax_cache
```

After the first run subsequent runs finish `jit` in a few hundred ms.

## Memory rules of thumb

- ViT-B/16 fp32 params: ~90M x 4 = 360 MB
- Optimizer state (adamw, m + v): ~720 MB
- Activation memory at bs=64 per core, seq=197 (=196+cls), d=768:
  roughly `bs * L * seq * d * 4 * 2` (fwd+bwd) = ~200 MB per core

Total headroom on a v4 core (~32 GB HBM) is enormous; ViT-B is compute
bound not memory bound at this batch size. Push batch until step time
stops scaling linearly.

## The `libtpu` version has to match jax

`jax[tpu]==0.4.35` pulls a specific `libtpu` build. Do not mix
`jax==0.4.35` with a stray `libtpu` from an earlier install; the symbol
table changed. If you see `TpuPlatformInitialize returned FAILED
PRECONDITION`, this is the reason. Fix: `pip install --force-reinstall
"jax[tpu]==0.4.35" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`.

## Preemptibility

Spot / preemptible TPU VMs are much cheaper but will disappear with
~30s warning. Orbax's async checkpoints combined with `save_interval_steps=1000`
means you rarely lose more than a few minutes of training. Restore on
startup with `mgr.restore_latest(template)` is the pattern.
