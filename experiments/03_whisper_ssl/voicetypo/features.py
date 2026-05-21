"""Frozen Whisper-base encoder features with mean+std pooling."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class WhisperFeatureExtractor:
    """Wraps HF Whisper-base. We use ONLY the encoder, so the decoder is dropped."""

    def __init__(self, model_id: str = "openai/whisper-base", device: str | None = None):
        from transformers import WhisperFeatureExtractor as HFFeat, WhisperModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feat = HFFeat.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id).to(self.device).eval()
        # encoder output frames are at 50 Hz (Whisper conv stride)
        self.frame_hz = 50
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.config.d_model    # 512 for whisper-base
        self.target_sr = 16000

    @torch.no_grad()
    def encode(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        """Returns frame-level encoder hidden states for the audible portion only.

        Whisper internally pads to 30s. We pad ourselves to 30s so the encoder runs,
        then slice the output back to the original duration in encoder frames.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if sr != self.target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        n_samples = len(audio)
        if n_samples == 0:
            return torch.zeros(0, self.embed_dim)

        n_frames_audio = int(np.ceil(n_samples / self.target_sr * self.frame_hz))
        inputs = self.feat(audio, sampling_rate=self.target_sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        out = self.model.encoder(input_features).last_hidden_state[0]   # (T_pad, D)
        return out[:n_frames_audio].detach().cpu()


def pool_mean_std(emb: torch.Tensor) -> torch.Tensor:
    """(T, D) -> (2D,) concat of mean and std along time."""
    if emb.shape[0] == 0:
        return torch.zeros(emb.shape[-1] * 2)
    mu = emb.mean(dim=0)
    sd = emb.std(dim=0) if emb.shape[0] > 1 else torch.zeros_like(mu)
    return torch.cat([mu, sd], dim=0)


def feature_dim(extractor: WhisperFeatureExtractor) -> int:
    return extractor.embed_dim * 2
