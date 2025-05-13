"""Async orbax checkpointing to a local dir or a GCS bucket.

Keeps the latest N (default 3) checkpoints and prunes the rest.

Usage:
    mgr = build_checkpoint_manager("gs://my-bucket/vit-runs/exp1", keep_last=3)
    mgr.save(step, {"state": state, "opt": opt_state, "ema": ema_state})
    ...
    restored = mgr.restore_latest()
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp


@dataclass
class CheckpointManager:
    """Thin wrapper around orbax.CheckpointManager with sensible defaults."""

    directory: str
    keep_last: int = 3
    save_interval_steps: int = 1000
    async_options: ocp.AsyncOptions | None = None

    def __post_init__(self):
        # gs:// paths are handled natively by orbax when gcsfs is installed
        options = ocp.CheckpointManagerOptions(
            max_to_keep=self.keep_last,
            save_interval_steps=self.save_interval_steps,
            enable_async_checkpointing=True,
        )
        self._mgr = ocp.CheckpointManager(
            self.directory,
            options=options,
        )

    def save(self, step: int, payload: dict[str, Any], force: bool = False) -> bool:
        args = ocp.args.Composite(**{k: ocp.args.StandardSave(v) for k, v in payload.items()})
        return bool(self._mgr.save(step, args=args, force=force))

    def wait_until_finished(self) -> None:
        self._mgr.wait_until_finished()

    def latest_step(self) -> int | None:
        return self._mgr.latest_step()

    def restore_latest(self, template: dict[str, Any]) -> dict[str, Any] | None:
        step = self.latest_step()
        if step is None:
            return None
        args = ocp.args.Composite(
            **{k: ocp.args.StandardRestore(v) for k, v in template.items()}
        )
        return self._mgr.restore(step, args=args)

    def all_steps(self) -> list[int]:
        return list(self._mgr.all_steps())

    def close(self) -> None:
        self._mgr.close()


def build_checkpoint_manager(
    directory: str | Path,
    *,
    keep_last: int = 3,
    save_interval_steps: int = 1000,
) -> CheckpointManager:
    return CheckpointManager(
        directory=str(directory),
        keep_last=keep_last,
        save_interval_steps=save_interval_steps,
    )
