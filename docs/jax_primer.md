# JAX + Flax primer (for this repo)

A short refresher on the pieces used in this codebase.

## jax.Array + NamedSharding + pjit

`jax.Array` is JAX's unified array type. A single `jax.Array` can live
across many devices; the layout is described by a `NamedSharding` that
binds each logical axis to a `Mesh` axis (`data`, `model`, ...).

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax, numpy as np

devices = np.asarray(jax.devices()).reshape(4, 2)
mesh = Mesh(devices, axis_names=("data", "model"))

with mesh:
    # a [B, D] tensor with B split across "data" and D split across "model"
    x_sh = NamedSharding(mesh, P("data", "model"))
    x = jax.device_put(x, x_sh)
```

`jax.jit(f, in_shardings=..., out_shardings=...)` is now the canonical
way to run sharded computations (pjit is folded into jit as of jax 0.4).

## Flax NNX (flax >= 0.10)

Flax added a stateful, PyTorch-flavored API called NNX. Modules hold
state directly (no more `apply`+`params` split); to jit you split into
a static `graphdef` and a mutable `state`.

```python
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, d_in, d_out, *, rngs):
        self.fc = nnx.Linear(d_in, d_out, rngs=rngs)

    def __call__(self, x):
        return self.fc(x)

model = MLP(768, 10, rngs=nnx.Rngs(0))
graphdef, state = nnx.split(model)

@jax.jit
def step(state, x):
    m = nnx.merge(graphdef, state)
    return m(x)
```

We use exactly this split-merge pattern in `src/training/train_jax.py`
so the outer `jax.jit` can enforce `in_shardings` for the mesh.

## optax

`optax` is the standalone JAX optimizer library. We use
`optax.adamw(..., mask=weight_decay_mask)` to keep weight decay off
LayerNorm scales, biases, and the cls/pos embeddings.

## orbax-checkpoint

Async, sharding-aware checkpointing. It knows how to serialize a
`jax.Array` that lives across devices without gathering to the host
first. Ideal for TPU-VM style training where the host has tiny memory
compared to the pod slice.
