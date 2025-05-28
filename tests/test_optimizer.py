"""Optimizer + schedule + weight-decay mask tests."""
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from src.model.vit_flax import ViTConfig, ViT
from src.training.optimizer import (
    build_optimizer,
    cosine_schedule_with_warmup,
    wsd_schedule,
    weight_decay_mask,
    EMA,
    flatten_grads_norm,
)


def _rngs():
    return nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1))


def test_cosine_schedule_ramps_up_and_decays():
    sch = cosine_schedule_with_warmup(base_lr=1e-3, warmup_steps=100, total_steps=1000)
    assert float(sch(0)) == pytest.approx(0.0, abs=1e-6)
    assert float(sch(100)) == pytest.approx(1e-3, rel=1e-3)
    end = float(sch(999))
    assert end < 1e-3


def test_wsd_schedule_has_stable_middle():
    sch = wsd_schedule(base_lr=5e-4, warmup_steps=50, stable_steps=200, decay_steps=100)
    assert float(sch(75)) == pytest.approx(5e-4, rel=1e-3)
    assert float(sch(200)) == pytest.approx(5e-4, rel=1e-3)


def test_weight_decay_mask_excludes_layernorm_and_bias():
    cfg = ViTConfig(image_size=32, patch_size=16, num_classes=2, num_layers=1)
    model = ViT(cfg, rngs=_rngs())
    _, state = nnx.split(model)
    mask = weight_decay_mask(state)
    # any leaf with "ln" in path should be False
    def _check(path, m):
        name = ".".join(str(getattr(p, "key", p)) for p in path).lower()
        if "layernorm" in name or ".ln" in name or "ln." in name or name.endswith(".bias"):
            assert m is False, f"{name} should not be weight-decayed but mask={m}"
    jax.tree_util.tree_map_with_path(_check, mask)


def test_ema_step_zero_matches_input():
    ema = EMA(decay=0.9)
    x = {"a": jnp.array([1.0, 2.0])}
    state = ema.init(x)
    new = ema.update(state, {"a": jnp.array([3.0, 4.0])}, step=0)
    # at step 0 effective_decay is very small, so new ~ params
    assert float(new["a"][0]) == pytest.approx(3.0 * (1 - ema.effective_decay(0)) + 1.0 * ema.effective_decay(0), rel=1e-5)


def test_flatten_grads_norm():
    g = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([0.0])}
    assert float(flatten_grads_norm(g)) == pytest.approx(5.0, rel=1e-6)


def test_build_optimizer_runs_a_step():
    cfg = {"base_lr": 1e-3, "weight_decay": 0.01, "warmup_steps": 10, "grad_clip": 1.0}
    tx = build_optimizer(cfg, total_steps=100)
    params = {"w": jnp.zeros((4, 4))}
    opt_state = tx.init(params)
    grads = {"w": jnp.ones((4, 4))}
    updates, _ = tx.update(grads, opt_state, params)
    new = optax.apply_updates(params, updates)
    assert new["w"].shape == (4, 4)
