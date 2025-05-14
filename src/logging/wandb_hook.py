"""Small wandb wrapper that no-ops when wandb is off or not installed.

Usage:
    hook = WandbHook(project="jax-vit", enabled=args.wandb, config=cfg)
    hook.log({"loss": loss, "ips": ips}, step=step)
    hook.finish()
"""
from __future__ import annotations

from typing import Any


class WandbHook:
    def __init__(
        self,
        *,
        project: str,
        run_name: str | None = None,
        config: dict | None = None,
        enabled: bool = True,
        entity: str | None = None,
    ) -> None:
        self.enabled = enabled
        self._run = None
        if not enabled:
            return
        try:
            import wandb  # type: ignore

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                entity=entity,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] disabled: {e}")
            self.enabled = False

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        self._wandb.log(metrics, step=step)

    def summary(self, **kv) -> None:
        if not self.enabled or self._run is None:
            return
        for k, v in kv.items():
            self._run.summary[k] = v

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._wandb.finish()


class NullHook:
    def log(self, *_a, **_k): ...
    def summary(self, **_k): ...
    def finish(self): ...
