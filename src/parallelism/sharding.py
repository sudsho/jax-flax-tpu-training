"""NamedSharding specs.

Two knobs to remember:
- FSDP-lite: shard the largest weight axis across the model axis. For a
  Linear (in, out) that's typically `out`. For MultiHeadAttention params
  that's the head-count-fused axis.
- Data: replicate batches across model, shard across data axis.

For ViT-B/16 with hidden=768 and heads=12 a mesh (data=4, model=2) is a
reasonable pjit config on v4-8; hidden splits evenly and each core owns
half the heads.
"""
from __future__ import annotations

from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .mesh import AXIS_DATA, AXIS_MODEL


def batch_sharding(mesh: Mesh) -> NamedSharding:
    """Batch axis sharded across `data`, rest replicated."""
    return NamedSharding(mesh, P(AXIS_DATA, None, None, None))


def replicated(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def param_sharding_rules(mesh: Mesh) -> dict[str, NamedSharding]:
    """Map parameter-name-suffix patterns to shardings.

    Kept intentionally simple. The train loop walks the param tree and picks
    the first matching pattern.
    """
    return {
        # attention QKV and output projections: shard along output/head axis
        "attn.qkv.kernel": NamedSharding(mesh, P(None, AXIS_MODEL)),
        "attn.out.kernel": NamedSharding(mesh, P(AXIS_MODEL, None)),
        # MLP fc1: [D, mlp_dim] shard mlp axis; fc2: [mlp_dim, D] shard mlp axis
        "mlp.fc1.kernel": NamedSharding(mesh, P(None, AXIS_MODEL)),
        "mlp.fc2.kernel": NamedSharding(mesh, P(AXIS_MODEL, None)),
        # Everything else replicated
        "*": replicated(mesh),
    }


def shard_pytree_like(
    tree: Any, sharding: NamedSharding | dict[str, NamedSharding]
) -> Any:
    """Move a pytree to devices with the given sharding spec.

    If `sharding` is a dict, apply per-leaf using the path of each leaf.
    """
    if isinstance(sharding, NamedSharding):
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), tree)

    def _put(path, x):
        leaf_name = ".".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        for pat, sh in sharding.items():
            if pat == "*":
                continue
            if pat in leaf_name:
                return jax.device_put(x, sh)
        return jax.device_put(x, sharding.get("*"))

    return jax.tree_util.tree_map_with_path(_put, tree)
