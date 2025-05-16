"""ViT-B/16 in PyTorch, meant to be traced by torch-xla for the baseline
throughput measurement on TPU v4-8.

Kept intentionally close to the Flax version so the comparison is apples
to apples: same widths, same depths, same GELU-approximate, same
attention layout.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0


class PatchEmbed(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, cfg.hidden_size, cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]  ->  [B, N, D]
        x = self.proj(x)
        b, d, h, w = x.shape
        return x.flatten(2).transpose(1, 2)  # [B, HW, D]


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = nn.MultiheadAttention(
            cfg.hidden_size,
            cfg.num_heads,
            dropout=cfg.attention_dropout_rate,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(cfg.mlp_dim, cfg.hidden_size),
            nn.Dropout(cfg.dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        y = self.ln2(x)
        return x + self.mlp(y)


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.hidden_size))
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(cfg.dropout_rate)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        b = x.size(0)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x[:, 0])


def vit_b16_torch(num_classes: int = 1000) -> ViT:
    return ViT(ViTConfig(num_classes=num_classes))
