"""CNN models for vowel classification.

  - SmallVowelCNN: 4 conv blocks + GAP + MLP head. Used by v1 (32 ch, MFCC) and
    v2 (64 ch, MFCC + waveform aug).
  - DeepVowelCNN: 1 stem + 5 residual blocks + GAP + linear. Used by v3 with a
    log-mel filterbank input.

Both models output logits over `n_classes` vowels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNConfig:
    n_classes: int = 7
    in_channels: int = 3
    base_channels: int = 32
    dropout: float = 0.3


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, pool: tuple[int, int] | None = (2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class SmallVowelCNN(nn.Module):
    def __init__(self, cfg: CNNConfig | None = None):
        super().__init__()
        cfg = cfg or CNNConfig()
        c = cfg.base_channels
        self.features = nn.Sequential(
            ConvBlock(cfg.in_channels, c, pool=(2, 2)),       # 40x32 -> 20x16
            ConvBlock(c, c * 2, pool=(2, 2)),                 # 20x16 -> 10x 8
            ConvBlock(c * 2, c * 4, pool=(2, 2)),             # 10x 8 ->  5x 4
            ConvBlock(c * 4, c * 4, pool=None),               #  5x 4 ->  5x 4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(cfg.dropout),
            nn.Linear(c * 4, c * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(c * 2, cfg.n_classes),
        )
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.gap(h)
        return self.head(h)


@dataclass
class DeepCNNConfig:
    n_classes: int = 7
    in_channels: int = 3
    base_channels: int = 48          # stem width
    dropout: float = 0.3


class ResBlock(nn.Module):
    """Two 3x3 conv layers + BN + ReLU + skip connection.

    If `stride > 1` or input/output channels differ, the skip uses a 1x1 conv
    to project; otherwise it is identity.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride == 1 and c_in == c_out:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        h = h + self.shortcut(x)
        return F.relu(h, inplace=True)


class DeepVowelCNN(nn.Module):
    """Residual CNN for log-mel-style inputs (~ (3, 64, 24)).

    Stages (channels, stride):
        stem    : c                   3x3 conv stride 1
        block1  : c        s=2        n_mels//2 x T//2
        block2  : 2c       s=2        n_mels//4 x T//4
        block3  : 2c       s=1
        block4  : 3c       s=2        n_mels//8 x T//8
        block5  : 3c       s=1
        GAP -> Linear(3c, n_classes)
    """

    def __init__(self, cfg: DeepCNNConfig | None = None):
        super().__init__()
        cfg = cfg or DeepCNNConfig()
        c = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResBlock(c,         c,         stride=2)
        self.block2 = ResBlock(c,         2 * c,     stride=2)
        self.block3 = ResBlock(2 * c,     2 * c,     stride=1)
        self.block4 = ResBlock(2 * c,     3 * c,     stride=2)
        self.block5 = ResBlock(3 * c,     3 * c,     stride=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(3 * c, cfg.n_classes)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.gap(h).flatten(1)
        h = self.dropout(h)
        return self.head(h)


@dataclass
class MLPHeadConfig:
    in_dim: int = 768                # Whisper-tiny embed (384) * 2 (mean+std)
    hidden_dim: int = 256
    n_classes: int = 7
    dropout: float = 0.3


class VowelMLPHead(nn.Module):
    """2-layer MLP probe over a frozen-encoder pooled vector."""

    def __init__(self, cfg: MLPHeadConfig | None = None):
        super().__init__()
        cfg = cfg or MLPHeadConfig()
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.n_classes),
        )
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def save_checkpoint(path, model: nn.Module, classes: list[str], extra: dict | None = None):
    if isinstance(model, DeepVowelCNN):
        kind = "deep"
        cfg = {
            "n_classes": model.cfg.n_classes,
            "in_channels": model.cfg.in_channels,
            "base_channels": model.cfg.base_channels,
            "dropout": model.cfg.dropout,
        }
    elif isinstance(model, SmallVowelCNN):
        kind = "small"
        cfg = {
            "n_classes": model.cfg.n_classes,
            "in_channels": model.cfg.in_channels,
            "base_channels": model.cfg.base_channels,
            "dropout": model.cfg.dropout,
        }
    elif isinstance(model, VowelMLPHead):
        kind = "mlp"
        cfg = {
            "in_dim": model.cfg.in_dim,
            "hidden_dim": model.cfg.hidden_dim,
            "n_classes": model.cfg.n_classes,
            "dropout": model.cfg.dropout,
        }
    else:
        raise TypeError(f"unknown model type: {type(model)}")
    payload = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "model_kind": kind,
        "cfg": cfg,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path) -> tuple[nn.Module, list[str], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = payload.get("model_kind", "small")
    if kind == "deep":
        cfg = DeepCNNConfig(**payload["cfg"])
        model: nn.Module = DeepVowelCNN(cfg)
    elif kind == "mlp":
        cfg = MLPHeadConfig(**payload["cfg"])
        model = VowelMLPHead(cfg)
    else:
        cfg = CNNConfig(**payload["cfg"])
        model = SmallVowelCNN(cfg)
    model.load_state_dict(payload["state_dict"])
    extra = payload.get("extra", {})
    extra["model_kind"] = kind
    return model, payload["classes"], extra
