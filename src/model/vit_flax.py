"""ViT-B/16 in Flax NNX.

Following Dosovitskiy et al. 2020 (An Image is Worth 16x16 Words).
Uses the new Flax NNX API (flax >= 0.10).

wip: only the encoder block skeleton lives here right now.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    use_bias: bool = True

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class EncoderBlock(nnx.Module):
    """A single transformer encoder block: prenorm attn + prenorm mlp."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        # placeholders - real submodules land in a follow-up commit.
        self.ln1 = nnx.LayerNorm(cfg.hidden_size, rngs=rngs)
        self.ln2 = nnx.LayerNorm(cfg.hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:  # noqa: ARG002
        # todo: attention + mlp
        return x + self.ln2(self.ln1(x))
