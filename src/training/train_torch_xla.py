"""PyTorch/XLA baseline training loop.

Runs the equivalent ViT-B/16 pretraining on TPU v4-8 via torch-xla, used
only for the throughput comparison in benchmarks/results.md.

Install: pip install -e ".[torch-xla]"  (needs the torch-xla wheel for
your TPU platform, see https://github.com/pytorch/xla for exact URL).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("torch-xla-vit-train")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def _cross_entropy_with_smoothing(logits, labels, num_classes, smoothing=0.1):
    import torch
    import torch.nn.functional as F

    y = F.one_hot(labels, num_classes).float()
    y = y * (1.0 - smoothing) + smoothing / num_classes
    logp = F.log_softmax(logits, dim=-1)
    return -(y * logp).sum(dim=-1).mean()


def _mp_fn(rank: int, cfg: dict, args):
    import torch
    import torch.optim as optim
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp  # noqa: F401
    from src.model.vit_torch_baseline import ViT, ViTConfig

    device = xm.xla_device()
    mcfg = ViTConfig(**{k: cfg["model"][k] for k in [
        "image_size", "patch_size", "num_classes",
        "hidden_size", "num_layers", "num_heads", "mlp_dim",
    ]})
    model = ViT(mcfg).to(device)

    opt = optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["base_lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=(0.9, 0.999),
    )

    # For the comparison bench we don't need real data: we just synthesize a
    # constant batch shape so we can measure step time cleanly.
    bs = cfg["train"]["batch_size_per_core"]
    x = torch.randn(bs, 3, mcfg.image_size, mcfg.image_size).to(device)
    y = torch.randint(0, mcfg.num_classes, (bs,)).to(device)

    steps = cfg["train"].get("bench_steps", 200)
    model.train()
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = _cross_entropy_with_smoothing(
            logits, y, mcfg.num_classes, cfg["train"].get("label_smoothing", 0.1)
        )
        loss.backward()
        xm.optimizer_step(opt)
        if step % cfg["train"].get("log_every", 50) == 0 and xm.is_master_ordinal():
            xm.mark_step()
            dt = time.time() - t0
            ips = (step + 1) * bs * cfg["xla"]["num_cores"] / max(dt, 1e-6)
            print(f"step {step:>5}  loss {loss.item():.4f}  ips {ips:.1f}")
    xm.rendezvous("done")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(_mp_fn, args=(cfg, args), nprocs=cfg["xla"]["num_cores"])


if __name__ == "__main__":
    main()
