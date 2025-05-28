"""Sharding spec sanity checks. These run cpu-only (single device mesh)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from src.parallelism.mesh import AXIS_DATA, AXIS_MODEL, MeshConfig, build_mesh, validate_axes
from src.parallelism.sharding import batch_sharding, replicated, param_sharding_rules


def _single_device_mesh():
    devices = np.asarray(jax.devices()).reshape(1, 1)
    return Mesh(devices, axis_names=(AXIS_DATA, AXIS_MODEL))


def test_mesh_config_resolves_default_to_all_devices():
    cfg = MeshConfig(data_parallel=-1, model_parallel=1)
    dp, mp = cfg.resolve(8)
    assert (dp, mp) == (8, 1)


def test_mesh_config_two_way_model_parallel():
    cfg = MeshConfig(data_parallel=-1, model_parallel=2)
    dp, mp = cfg.resolve(8)
    assert (dp, mp) == (4, 2)


def test_mesh_config_rejects_non_divisible():
    cfg = MeshConfig(data_parallel=-1, model_parallel=3)
    with pytest.raises(AssertionError):
        cfg.resolve(8)


def test_batch_sharding_spec():
    mesh = _single_device_mesh()
    sh = batch_sharding(mesh)
    assert sh.spec == P(AXIS_DATA, None, None, None)


def test_replicated_spec():
    mesh = _single_device_mesh()
    sh = replicated(mesh)
    assert sh.spec == P()


def test_param_rules_shape():
    mesh = _single_device_mesh()
    rules = param_sharding_rules(mesh)
    assert "*" in rules
    for key in ("attn.qkv.kernel", "attn.out.kernel", "mlp.fc1.kernel", "mlp.fc2.kernel"):
        assert key in rules
        assert isinstance(rules[key], NamedSharding)


def test_validate_axes_ok():
    mesh = _single_device_mesh()
    validate_axes(mesh)  # should not raise


def test_validate_axes_rejects_reversed():
    devices = np.asarray(jax.devices()).reshape(1, 1)
    bad = Mesh(devices, axis_names=(AXIS_MODEL, AXIS_DATA))
    with pytest.raises(ValueError, match="do not match expected"):
        validate_axes(bad)
