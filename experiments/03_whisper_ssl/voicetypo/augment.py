"""Waveform-domain augmentation: noise / pitch / stretch / gain / reverb.
SpecAugment lives near the model since it operates on encoder embeddings."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np


def add_gaussian_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    sig_p = float(np.mean(x ** 2)) + 1e-12
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_p = sig_p / snr_lin
    noise = np.random.randn(len(x)).astype(np.float32) * np.sqrt(noise_p)
    return x + noise


def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if abs(semitones) < 1e-3:
        return x
    import librosa
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=semitones).astype(np.float32)


def time_stretch(x: np.ndarray, rate: float) -> np.ndarray:
    if abs(rate - 1.0) < 1e-3:
        return x
    import librosa
    return librosa.effects.time_stretch(x, rate=rate).astype(np.float32)


def convolve_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if rir is None or len(rir) == 0:
        return x
    out = np.convolve(x, rir, mode="full")[: len(x)]
    # normalize peak so reverb doesn't blow up gain
    peak = float(np.max(np.abs(out))) + 1e-9
    return (out / peak * float(np.max(np.abs(x)) + 1e-9)).astype(np.float32)


class WaveformAugmenter:
    """Random per-call augmentation. Pulls knobs from config['augment']."""

    def __init__(self, cfg: dict, noise_pool: Optional[list[Path]] = None,
                 rir_pool: Optional[list[Path]] = None, sample_rate: int = 16000):
        self.cfg = cfg
        self.sr = sample_rate
        self.noise_pool = noise_pool or []
        self.rir_pool = rir_pool or []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        # pitch
        lo, hi = cfg["pitch_semitones"]
        x = pitch_shift(x, self.sr, random.uniform(lo, hi))
        # time stretch
        lo, hi = cfg["time_stretch"]
        x = time_stretch(x, random.uniform(lo, hi))
        # noise
        snr_lo, snr_hi = cfg["noise_snr_db"]
        snr = random.uniform(snr_lo, snr_hi)
        if self.noise_pool:
            x = self._mix_noise_clip(x, snr)
        else:
            x = add_gaussian_noise(x, snr)
        # reverb
        if self.rir_pool and random.random() < cfg["reverb_prob"]:
            rir_path = random.choice(self.rir_pool)
            try:
                import soundfile as sf
                rir, sr = sf.read(str(rir_path))
                if rir.ndim > 1:
                    rir = rir.mean(axis=1)
                if sr != self.sr:
                    import librosa
                    rir = librosa.resample(rir.astype(np.float32), orig_sr=sr, target_sr=self.sr)
                x = convolve_rir(x, rir.astype(np.float32))
            except Exception:
                pass
        # gain
        lo, hi = cfg["gain_db"]
        x = apply_gain(x, random.uniform(lo, hi))
        # safety clip
        peak = float(np.max(np.abs(x)) + 1e-9)
        if peak > 1.0:
            x = (x / peak).astype(np.float32)
        return x

    def _mix_noise_clip(self, x: np.ndarray, snr_db: float) -> np.ndarray:
        import soundfile as sf
        path = random.choice(self.noise_pool)
        try:
            n, sr = sf.read(str(path))
        except Exception:
            return add_gaussian_noise(x, snr_db)
        if n.ndim > 1:
            n = n.mean(axis=1)
        if sr != self.sr:
            import librosa
            n = librosa.resample(n.astype(np.float32), orig_sr=sr, target_sr=self.sr)
        if len(n) < len(x):
            reps = int(np.ceil(len(x) / max(1, len(n))))
            n = np.tile(n, reps)
        start = random.randint(0, max(0, len(n) - len(x)))
        n = n[start: start + len(x)].astype(np.float32)
        sig_p = float(np.mean(x ** 2)) + 1e-12
        noi_p = float(np.mean(n ** 2)) + 1e-12
        scale = float(np.sqrt(sig_p / (noi_p * 10.0 ** (snr_db / 10.0))))
        return (x + scale * n).astype(np.float32)


def specaugment_embeddings(emb: "torch.Tensor", time_mask: int = 16, freq_mask: int = 8) -> "torch.Tensor":
    """Apply SpecAugment-style masking to a (T, D) encoder embedding tensor."""
    import torch
    out = emb.clone()
    T, D = out.shape
    if time_mask > 0 and T > time_mask:
        t0 = int(torch.randint(0, T - time_mask, (1,)))
        out[t0:t0 + time_mask] = 0.0
    if freq_mask > 0 and D > freq_mask:
        f0 = int(torch.randint(0, D - freq_mask, (1,)))
        out[:, f0:f0 + freq_mask] = 0.0
    return out
