"""MLP probe head over pooled Whisper embeddings."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ProbeConfig:
    in_dim: int
    n_classes: int
    hidden_dim: int = 256
    dropout: float = 0.3


class VowelProbe(nn.Module):
    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.in_dim)
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


def save_checkpoint(path, model: VowelProbe, classes: list[str], extra: dict | None = None):
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "in_dim": model.cfg.in_dim,
            "n_classes": model.cfg.n_classes,
            "hidden_dim": model.cfg.hidden_dim,
            "dropout": model.cfg.dropout,
        },
        "classes": classes,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, map_location="cpu") -> tuple[VowelProbe, list[str], dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ProbeConfig(**ckpt["config"])
    model = VowelProbe(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], ckpt.get("extra", {})
