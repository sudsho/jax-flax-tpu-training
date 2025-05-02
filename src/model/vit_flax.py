"""ViT-B/16 in Flax NNX.

Following Dosovitskiy et al. 2020 (An Image is Worth 16x16 Words).
Uses the new Flax NNX API (flax >= 0.10).
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


class PatchEmbed(nnx.Module):
    """Cut the image into patches with a strided conv, then linear-project."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        self.proj = nnx.Conv(
            in_features=3,
            out_features=cfg.hidden_size,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=(cfg.patch_size, cfg.patch_size),
            padding="VALID",
            use_bias=cfg.use_bias,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [B, H, W, C]  ->  [B, N, D]
        x = self.proj(x)
        b, h, w, d = x.shape
        return x.reshape(b, h * w, d)


class LearnablePosEmbed(nnx.Module):
    """cls token + learnable 1D position embedding (N+1, D)."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs) -> None:
        self.cls = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, cfg.hidden_size)) * 0.02
        )
        self.pos = nnx.Param(
            jax.random.normal(rngs.params(), (1, cfg.num_patches + 1, cfg.hidden_size))
            * 0.02
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        b = x.shape[0]
        cls = jnp.broadcast_to(self.cls.value, (b, 1, x.shape[-1]))
        x = jnp.concatenate([cls, x], axis=1)
        return x + self.pos.value


class EncoderBlock(nnx.Module):
    """A single transformer encoder block: prenorm attn + prenorm mlp."""

    def __init__(self, cfg: ViTConfig, *, rngs: nnx.Rngs) -> None:
        self.cfg = cfg
        self.ln1 = nnx.LayerNorm(cfg.hidden_size, rngs=rngs)
        self.ln2 = nnx.LayerNorm(cfg.hidden_size, rngs=rngs)

    def __call__(self, x: jax.Array, *, deterministic: bool = True) -> jax.Array:  # noqa: ARG002
        # attention + mlp land in a follow-up commit
        return x + self.ln2(self.ln1(x))
