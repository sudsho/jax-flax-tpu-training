# Mesh and sharding

The pjit mesh in this repo is 2D: `(data, model)`. Every leaf in the
param and batch trees is annotated by a `NamedSharding` that says which
axis it splits along.

## v4-8 layouts

`src/parallelism/mesh.py` notes that the addressable device count on a
v4-8 host is 4 (four chips, each pairing two cores under megacore). With
4 addressable devices, the practical layouts are:

| Layout                | mesh     | who splits what                              |
|-----------------------|----------|----------------------------------------------|
| Pure data parallel    | (4, 1)   | batch by data; params replicated             |
| 2-way tensor + 2 dp   | (2, 2)   | batch by data; hidden/mlp axes by model      |

`configs/jax_tpu_v4_8.yaml` sets `model_parallel=2` and lets `data_parallel`
resolve to whatever is left, so with 4 devices this yields `(2, 2)`.

## Rule: batches always shard on `data`

Batches enter the step function through `NamedSharding(mesh, P("data",
None, None, None))`. Never across `model`; the whole point of the model
axis is that the same batch flows through both halves and the halves
disagree on which slice of parameters they own.

## Rule: params replicate unless flagged

Default is `NamedSharding(mesh, P())` (fully replicated). The
`param_sharding_rules(mesh)` dict in `src/parallelism/sharding.py` sketches
which leaves would shard on `model`:

- `attn.out.kernel` -> `P("model", None)`
- `mlp.fc1.kernel`  -> `P(None, "model")`
- `mlp.fc2.kernel`  -> `P("model", None)`

Note: Flax NNX `MultiHeadAttention` uses separate `query`, `key`, `value`
submodules; there is no fused `attn.qkv.kernel` to match. The current
`train_jax.py` scaffold does not consume `param_sharding_rules` and
replicates every parameter, so no tensor parallelism is applied end to
end. Wiring in the rules and expanding them to cover the Q/K/V leaves is
left as a follow-up.

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
