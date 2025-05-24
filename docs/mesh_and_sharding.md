# Mesh and sharding

The pjit mesh in this repo is 2D: `(data, model)`. Every leaf in the
param and batch trees is annotated by a `NamedSharding` that says which
axis it splits along.

## v4-8 layouts we support

| Layout                | mesh     | who splits what                              |
|-----------------------|----------|----------------------------------------------|
| Pure data parallel    | (8, 1)   | batch by data; params replicated             |
| 2-way tensor + 4 dp   | (4, 2)   | batch by data; hidden/mlp axes by model      |
| 4-way tensor + 2 dp   | (2, 4)   | good when hidden > 1024 (ViT-L, ViT-H)       |

The default in `configs/jax_tpu_v4_8.yaml` is `(4, 2)`. That splits the
MLP's fc1 output axis and the attention output axis across 2 chips,
keeping activation memory reasonable while leaving 4-way batch dp.

## Rule: batches always shard on `data`

Batches enter the step function through `NamedSharding(mesh, P("data",
None, None, None))`. Never across `model`; the whole point of the model
axis is that the same batch flows through both halves and the halves
disagree on which slice of parameters they own.

## Rule: params replicate unless flagged

Default is `NamedSharding(mesh, P())` (fully replicated). The
`param_sharding_rules(mesh)` dict in `src/parallelism/sharding.py` lists
which leaves shard on `model`:

- `attn.qkv.kernel` -> `P(None, "model")`
- `attn.out.kernel` -> `P("model", None)`
- `mlp.fc1.kernel`  -> `P(None, "model")`
- `mlp.fc2.kernel`  -> `P("model", None)`

Everything else stays replicated. LayerNorm scales are tiny; sharding
them just adds collectives with no memory saving.

## Debugging

- `jax.debug.visualize_array_sharding(x)` prints an ascii picture of
  how an array is laid out.
- `jax.jit(f).lower(...).compile().as_text()` prints the compiled HLO
  so you can spot unwanted all-gathers.
- Set `JAX_LOG_COMPILES=1` to see when jit recompiles.
- The most common gotcha: forgetting to `with mesh:` around the jit
  call. Without it, `in_shardings=NamedSharding(mesh, ...)` still
  works (the mesh is captured), but any *free* `jax.device_put`
  outside the context silently replicates.
