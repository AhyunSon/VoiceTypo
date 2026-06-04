"""Manifest loading + speaker-disjoint split.

The split logic mirrors voicetypo_new/voicetypo/data/dataset.py exactly so
methods 2 and 3 train and evaluate on the same speakers.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from voicetypo_light import (
    EXT_MANIFEST,
    SPLIT_SEED,
    SPLIT_TEST_FRAC,
    SPLIT_VAL_FRAC,
    VOWEL_CLASSES,
)


@dataclass
class Sample:
    audio_path: Path
    label: str
    speaker_id: str
    source: str
    duration_ms: int


def load_manifest(path: Path | None = None) -> list[Sample]:
    path = Path(path) if path else EXT_MANIFEST
    out: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(
                Sample(
                    audio_path=Path(r["audio"]),
                    label=r["label"],
                    speaker_id=r["speaker_id"],
                    source=r["source"],
                    duration_ms=int(r.get("duration_ms", 200)),
                )
            )
    return out


def speaker_disjoint_split(
    samples: list[Sample],
    val_frac: float = SPLIT_VAL_FRAC,
    test_frac: float = SPLIT_TEST_FRAC,
    seed: int = SPLIT_SEED,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)
    by_spk: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_spk[s.speaker_id].append(s)
    speakers = list(by_spk.keys())
    rng.shuffle(speakers)
    n = len(speakers)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_spk = set(speakers[:n_test])
    val_spk = set(speakers[n_test : n_test + n_val])
    train_spk = set(speakers[n_test + n_val :])
    train, val, test = [], [], []
    for s in samples:
        if s.speaker_id in test_spk:
            test.append(s)
        elif s.speaker_id in val_spk:
            val.append(s)
        elif s.speaker_id in train_spk:
            train.append(s)
    return train, val, test


class CachedMFCCDataset(Dataset):
    """In-memory MFCC tensor dataset with optional SpecAugment-style masking."""

    def __init__(
        self,
        npz_path: Path,
        time_mask: int = 0,
        freq_mask: int = 0,
        gain_db_range: tuple[float, float] | None = None,
    ):
        d = np.load(npz_path, allow_pickle=True)
        self.X = torch.from_numpy(d["X"]).float()      # (N, 3, n_mfcc, T)
        self.y = torch.from_numpy(d["y"]).long()
        self.spk = d["spk"]
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.gain_db_range = gain_db_range

    def __len__(self) -> int:
        return self.X.shape[0]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (3, n_mfcc, T)
        if self.gain_db_range is not None:
            lo, hi = self.gain_db_range
            db = float(torch.empty(1).uniform_(lo, hi).item())
            x = x + db / 20.0  # rough log-domain shift on the cepstral 0th coef channel
        if self.freq_mask > 0:
            f = int(torch.randint(0, self.freq_mask + 1, (1,)).item())
            if f > 0:
                f0 = int(torch.randint(0, max(1, x.shape[1] - f), (1,)).item())
                x = x.clone()
                x[:, f0 : f0 + f, :] = 0.0
        if self.time_mask > 0:
            t = int(torch.randint(0, self.time_mask + 1, (1,)).item())
            if t > 0:
                t0 = int(torch.randint(0, max(1, x.shape[2] - t), (1,)).item())
                x = x.clone() if not x.is_floating_point() else x
                x[:, :, t0 : t0 + t] = 0.0
        return x

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.time_mask or self.freq_mask or self.gain_db_range is not None:
            x = self._augment(x)
        return x, int(self.y[i]), str(self.spk[i])


def write_cached(npz_path: Path, X: np.ndarray, y: np.ndarray, spk: np.ndarray):
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        spk=spk.astype(object),
    )


def label_to_idx(label: str) -> int:
    return VOWEL_CLASSES.index(label)
