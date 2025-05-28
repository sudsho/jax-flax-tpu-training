"""Round-trip an nnx state through orbax to a local dir."""
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.checkpoint.orbax_ckpt import build_checkpoint_manager


class _Toy(nnx.Module):
    def __init__(self, *, rngs):
        self.fc = nnx.Linear(4, 3, rngs=rngs)

    def __call__(self, x):
        return self.fc(x)


@pytest.fixture
def tmp_ckpt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt"
    yield d
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def test_save_and_restore_roundtrip(tmp_ckpt_dir):
    model = _Toy(rngs=nnx.Rngs(0))
    _, state = nnx.split(model)

    mgr = build_checkpoint_manager(tmp_ckpt_dir, keep_last=2, save_interval_steps=1)
    ok = mgr.save(step=0, payload={"state": state}, force=True)
    mgr.wait_until_finished()
    assert ok

    template = {"state": state}
    restored = mgr.restore_latest(template)
    assert restored is not None
    jax.tree_util.tree_map(
        lambda a, b: jnp.testing.assert_allclose(a, b) if hasattr(a, "shape") else None,
        state,
        restored["state"],
    )


def test_keep_last_prunes(tmp_ckpt_dir):
    model = _Toy(rngs=nnx.Rngs(0))
    _, state = nnx.split(model)
    mgr = build_checkpoint_manager(tmp_ckpt_dir, keep_last=2, save_interval_steps=1)
    for step in range(5):
        mgr.save(step=step, payload={"state": state}, force=True)
        mgr.wait_until_finished()
    kept = mgr.all_steps()
    assert len(kept) <= 2
