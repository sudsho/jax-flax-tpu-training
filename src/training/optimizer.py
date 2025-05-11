"""Optax AdamW with a weight-decay mask that excludes LayerNorm / biases.

Also: cosine schedule with warmup, and a simple EMA helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax


def _is_wd_leaf(path, _leaf) -> bool:
    """True if this leaf should get weight decay.

    Excludes:
      - LayerNorm parameters (scale, bias)
      - all bias vectors
      - cls token and pos embed (following the ViT paper's small-vector rule)
    """
    name = ".".join(str(getattr(p, "key", p)) for p in path)
    lower = name.lower()
    if "layernorm" in lower or ".ln" in lower or "ln." in lower:
        return False
    if lower.endswith(".bias") or ".bias." in lower:
        return False
    if "pos_embed" in lower or "cls" in lower:
        return False
    return True


def weight_decay_mask(params) -> Any:
    return jax.tree_util.tree_map_with_path(lambda p, l: _is_wd_leaf(p, l), params)


def cosine_schedule_with_warmup(
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    end_lr: float = 0.0,
) -> optax.Schedule:
    """Standard linear-warmup + cosine-to-end_lr schedule."""
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, base_lr, warmup_steps),
            optax.cosine_decay_schedule(
                base_lr, total_steps - warmup_steps, alpha=end_lr / max(base_lr, 1e-9)
            ),
        ],
        [warmup_steps],
    )


def wsd_schedule(
    *,
    base_lr: float,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    end_lr: float = 0.0,
) -> optax.Schedule:
    """Warmup-stable-decay, sometimes preferred for ViT continual training."""
    return optax.join_schedules(
        [
            optax.linear_schedule(0.0, base_lr, warmup_steps),
            optax.constant_schedule(base_lr),
            optax.linear_schedule(base_lr, end_lr, decay_steps),
        ],
        [warmup_steps, warmup_steps + stable_steps],
    )


def build_optimizer(cfg: dict, total_steps: int) -> optax.GradientTransformation:
    """Build the training optimizer from a config dict.

    Example config:
        base_lr: 1.0e-3
        weight_decay: 0.05
        warmup_steps: 500
        grad_clip: 1.0
        beta1: 0.9
        beta2: 0.999
    """
    lr = cosine_schedule_with_warmup(
        base_lr=cfg["base_lr"],
        warmup_steps=cfg.get("warmup_steps", 500),
        total_steps=total_steps,
    )
    opt = optax.adamw(
        learning_rate=lr,
        b1=cfg.get("beta1", 0.9),
        b2=cfg.get("beta2", 0.999),
        eps=cfg.get("eps", 1e-8),
        weight_decay=cfg.get("weight_decay", 0.05),
        mask=weight_decay_mask,
    )
    grad_clip = cfg.get("grad_clip", 1.0)
    if grad_clip and grad_clip > 0:
        opt = optax.chain(optax.clip_by_global_norm(grad_clip), opt)
    return opt


@dataclass
class EMA:
    """Simple polyak-averaged param tracker."""

    decay: float = 0.9999

    def init(self, params):
        return jax.tree_util.tree_map(lambda x: x, params)

    def update(self, ema_state, new_params):
        d = self.decay
        return jax.tree_util.tree_map(lambda e, p: e * d + p * (1.0 - d), ema_state, new_params)
