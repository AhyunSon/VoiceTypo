"""Waveform-level augmentation for v2.

Applied while pre-computing the *training* MFCC cache (val/test stay clean).
Each augmented pass produces a new training example, so with aug_passes=2 the
training set grows 3x (original + 2 augmented copies per source clip) and the
model sees pitch / speed / gain / noise variation it would otherwise have to
generalise to from scratch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class AugmentConfig:
    sample_rate: int = 16000
    p_pitch: float = 0.7
    p_stretch: float = 0.5
    p_noise: float = 0.7
    pitch_semitones: float = 4.0          # ±4 semitones (covers child voices)
    stretch_range: tuple[float, float] = (0.9, 1.1)
    gain_db_range: tuple[float, float] = (-10.0, 10.0)
    snr_db_range: tuple[float, float] = (5.0, 30.0)


class WaveformAugmenter:
    def __init__(self, cfg: AugmentConfig | None = None):
        self.cfg = cfg or AugmentConfig()

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        c = self.cfg
        x = wav.astype(np.float32, copy=False)
        if random.random() < c.p_pitch:
            n_steps = random.uniform(-c.pitch_semitones, c.pitch_semitones)
            x = librosa.effects.pitch_shift(x, sr=c.sample_rate, n_steps=n_steps)
        if random.random() < c.p_stretch:
            rate = random.uniform(*c.stretch_range)
            try:
                x = librosa.effects.time_stretch(x, rate=rate)
            except Exception:
                pass  # very short clips can fail; skip stretch silently
        # Gain (always)
        gain_db = random.uniform(*c.gain_db_range)
        x = x * float(10.0 ** (gain_db / 20.0))
        # Additive Gaussian noise at random SNR
        if random.random() < c.p_noise:
            snr_db = random.uniform(*c.snr_db_range)
            sig_pow = float(np.mean(x * x)) + 1e-12
            noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
            noise = np.random.randn(*x.shape).astype(np.float32) * float(np.sqrt(noise_pow))
            x = x + noise
        return x.astype(np.float32, copy=False)
