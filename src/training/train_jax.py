"""Main JAX training loop.

wip skeleton: single-host stub for now. pjit + mesh + orbax + wandb hooks
land in follow-ups.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("jax-vit-train")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    print("devices:", jax.devices())
    print("config:", cfg.get("run_name", "unnamed"))
    # todo: build model, mesh, optimizer, loop


if __name__ == "__main__":
    main()
