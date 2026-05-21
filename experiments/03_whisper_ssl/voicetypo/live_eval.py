"""Shared helpers for live mic test (04) and offline wav re-eval (06)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from voicetypo.features import pool_mean_std


def trim_silence(
    wav: np.ndarray,
    sr: int = 16000,
    threshold_db: float = -38.0,
    frame_ms: int = 20,
) -> np.ndarray:
    """Drop leading/trailing 20 ms frames whose RMS is below threshold_db (dBFS)."""
    frame_len = sr * frame_ms // 1000
    if len(wav) < frame_len:
        return wav
    n = len(wav) // frame_len
    if n == 0:
        return wav
    frames = wav[: n * frame_len].reshape(n, frame_len)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    voiced = db > threshold_db
    if not voiced.any():
        return wav
    first = int(np.argmax(voiced))
    last = int(n - 1 - np.argmax(voiced[::-1]))
    return wav[first * frame_len : (last + 1) * frame_len]


def vowel_core(wav: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """Trim silence, then take the same 30–90% core the training extractor uses
    so live segments match the training distribution."""
    threshold_db = cfg.get("inference", {}).get("vad_threshold_db", -38.0)
    trimmed = trim_silence(wav, sr=sr, threshold_db=threshold_db)
    if len(trimmed) < int(0.05 * sr):
        trimmed = wav  # last-resort: nothing detected, keep whole recording
    lo = cfg["extraction"]["segment_lo"]
    hi = cfg["extraction"]["segment_hi"]
    n = len(trimmed)
    i0 = int(n * lo)
    i1 = int(n * hi)
    core = trimmed[i0:i1] if i1 > i0 else trimmed
    if len(core) < int(0.04 * sr):
        core = trimmed
    return core


def classify(
    model,
    extractor,
    segment: np.ndarray,
    sr: int,
    classes: list[str],
    device: str,
    calibration: Optional[dict] = None,
) -> Tuple[int, np.ndarray]:
    """Whisper encode + pool + (calibration LR if given, else MLP probe)
    → (pred_idx, probs[len(classes)]).

    When calibration is provided, the MLP probe is bypassed entirely — the
    user-specific logistic regression replaces it. This keeps each tested
    model honest: the saved probe checkpoint stays unchanged, and the
    --calibration switch is the only thing that flips between
    population-trained and user-adapted predictions.
    """
    emb = extractor.encode(segment, sr=sr)
    v = pool_mean_std(emb)
    if calibration is not None:
        probs = apply_calibration(v.numpy().astype(np.float32), calibration)
        return int(np.argmax(probs)), probs
    v = v.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(v)[0].cpu().numpy()
    logits = logits - logits.max()
    e = np.exp(logits)
    probs = e / e.sum()
    return int(np.argmax(probs)), probs


def load_calibration(path: str | Path) -> dict:
    """Load a calibration npz produced by scripts/05_calibrate.py."""
    d = np.load(str(path), allow_pickle=False)
    return {
        "W": d["W"].astype(np.float32),                # (n_classes, D)
        "b": d["b"].astype(np.float32),                # (n_classes,)
        "classes": [str(c) for c in d["classes"].tolist()],
        "in_dim": int(d["in_dim"]),
        "encoder_id": str(d["encoder_id"]),
    }


def apply_calibration(pooled: np.ndarray, calib: dict) -> np.ndarray:
    """pooled: (D,) float32. Returns softmax probs (n_classes,)."""
    z = calib["W"] @ pooled + calib["b"]
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
