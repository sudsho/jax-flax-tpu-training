"""Shape sanity for the Flax ViT."""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.model.vit_flax import ViTConfig, ViT, vit_b16, vit_s16


def _rngs(seed: int = 0) -> nnx.Rngs:
    return nnx.Rngs(params=jax.random.key(seed), dropout=jax.random.key(seed + 1))


@pytest.mark.parametrize("num_classes", [10, 100, 1000])
def test_vit_b16_forward_shape(num_classes):
    model = vit_b16(num_classes=num_classes, rngs=_rngs())
    x = jnp.zeros((2, 224, 224, 3), dtype=jnp.float32)
    out = model(x, deterministic=True)
    assert out.shape == (2, num_classes)


def test_vit_s16_forward_shape():
    model = vit_s16(num_classes=10, rngs=_rngs())
    x = jnp.zeros((3, 224, 224, 3), dtype=jnp.float32)
    out = model(x, deterministic=True)
    assert out.shape == (3, 10)


def test_num_patches_matches():
    cfg = ViTConfig(image_size=224, patch_size=16)
    assert cfg.num_patches == 14 * 14


def test_custom_image_size():
    cfg = ViTConfig(image_size=128, patch_size=16, num_classes=5)
    model = ViT(cfg, rngs=_rngs())
    x = jnp.zeros((1, 128, 128, 3))
    out = model(x)
    assert out.shape == (1, 5)
