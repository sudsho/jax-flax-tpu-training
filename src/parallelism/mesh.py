"""Device mesh construction.

On a TPU v4-8 the addressable device count is 4 (host sees 4 chips, each
holding 2 cores in the megacore pairing). We build a 2D logical mesh
`(data, model)` and let pjit shard along whichever axis matches the
NamedSharding spec.

For pure data parallel: mesh = (8, 1) or (4, 1) with model axis size 1.
For 2-way tensor parallel + 4-way data parallel: mesh = (4, 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh


AXIS_DATA = "data"
AXIS_MODEL = "model"

# order matters - jax uses it when building NamedSharding partition specs
AXIS_ORDER: tuple[str, ...] = (AXIS_DATA, AXIS_MODEL)


@dataclass
class MeshConfig:
    data_parallel: int = -1  # -1 means "use all devices along data"
    model_parallel: int = 1

    def resolve(self, num_devices: int) -> tuple[int, int]:
        mp = self.model_parallel
        dp = self.data_parallel
        if dp == -1:
            assert num_devices % mp == 0, (
                f"num_devices={num_devices} not divisible by model_parallel={mp}"
            )
            dp = num_devices // mp
        assert dp * mp == num_devices, (
            f"dp*mp ({dp}*{mp}={dp*mp}) must equal num_devices ({num_devices})"
        )
        return dp, mp


def build_mesh(cfg: MeshConfig | None = None) -> Mesh:
    cfg = cfg or MeshConfig()
    devices = jax.devices()
    dp, mp = cfg.resolve(len(devices))
    device_array = np.asarray(devices).reshape(dp, mp)
    return Mesh(device_array, axis_names=(AXIS_DATA, AXIS_MODEL))


def describe(mesh: Mesh) -> str:
    shape = ", ".join(f"{name}={size}" for name, size in mesh.shape.items())
    return f"Mesh({shape}) on {mesh.size} devices ({jax.devices()[0].platform})"


def validate_axes(mesh: Mesh) -> None:
    """Guard against silently getting a mesh with unexpected axis names."""
    got = tuple(mesh.axis_names)
    if got != AXIS_ORDER:
        raise ValueError(
            f"mesh axis names {got} do not match expected {AXIS_ORDER}; "
            "shardings elsewhere in the codebase assume (data, model) order"
        )
