"""MFCC feature extraction with delta/delta-delta channels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

from voicetypo_light import (
    LOGMEL_HOP,
    LOGMEL_N_FFT,
    LOGMEL_N_MELS,
    MFCC_HOP,
    MFCC_N_FFT,
    MFCC_N_MELS,
    MFCC_N_MFCC,
    SAMPLE_RATE,
    TARGET_FRAMES,
)


def _build_mfcc(device: str = "cpu") -> T.MFCC:
    return T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=MFCC_N_MFCC,
        melkwargs={
            "n_fft": MFCC_N_FFT,
            "hop_length": MFCC_HOP,
            "n_mels": MFCC_N_MELS,
            "center": True,
            "power": 2.0,
        },
    ).to(device)


def _build_delta(device: str = "cpu") -> T.ComputeDeltas:
    return T.ComputeDeltas(win_length=5).to(device)


def read_wav_mono(path: Path | str) -> torch.Tensor:
    """Read a wav file to a mono float32 tensor of shape (1, T) at SAMPLE_RATE."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(data.T)                                  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav  # (1, T)


def pad_or_truncate(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """x: (..., F, T). Pad with zeros on the right, or center-crop if too long."""
    t = x.shape[-1]
    if t == n_frames:
        return x
    if t < n_frames:
        pad = n_frames - t
        return torch.nn.functional.pad(x, (0, pad))
    # center-crop — vowel core is most stable in the middle
    start = (t - n_frames) // 2
    return x[..., start:start + n_frames]


class MFCCExtractor:
    """Stateful MFCC + delta + delta-delta extractor.

    Output shape per waveform: (3, n_mfcc, target_frames)
        channel 0 = MFCC
        channel 1 = delta
        channel 2 = delta-delta
    """

    def __init__(self, device: str = "cpu", target_frames: int = TARGET_FRAMES):
        self.device = device
        self.target_frames = target_frames
        self.mfcc = _build_mfcc(device)
        self.delta = _build_delta(device)

    @torch.no_grad()
    def from_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (1, T) or (T,)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)
        m = self.mfcc(wav)              # (1, n_mfcc, T)
        d1 = self.delta(m)
        d2 = self.delta(d1)
        feat = torch.stack([m.squeeze(0), d1.squeeze(0), d2.squeeze(0)], dim=0)
        return pad_or_truncate(feat, self.target_frames)

    @torch.no_grad()
    def from_path(self, path: Path | str) -> torch.Tensor:
        wav = read_wav_mono(path)
        return self.from_waveform(wav)


class LogMelExtractor:
    """Log-mel filterbank + delta + delta-delta extractor (v3 input).

    Output shape per waveform: (3, n_mels, target_frames)
    """

    def __init__(
        self,
        device: str = "cpu",
        target_frames: int = TARGET_FRAMES,
        n_mels: int = LOGMEL_N_MELS,
        n_fft: int = LOGMEL_N_FFT,
        hop: int = LOGMEL_HOP,
    ):
        self.device = device
        self.target_frames = target_frames
        self.n_mels = n_mels
        self.mel = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            center=True,
            power=2.0,
        ).to(device)
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0).to(device)
        self.delta = _build_delta(device)

    @torch.no_grad()
    def from_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)
        mel = self.mel(wav)              # (1, n_mels, T)
        log_mel = self.to_db(mel)        # dB-scaled log-mel
        d1 = self.delta(log_mel)
        d2 = self.delta(d1)
        feat = torch.stack(
            [log_mel.squeeze(0), d1.squeeze(0), d2.squeeze(0)], dim=0
        )
        return pad_or_truncate(feat, self.target_frames)

    @torch.no_grad()
    def from_path(self, path: Path | str) -> torch.Tensor:
        wav = read_wav_mono(path)
        return self.from_waveform(wav)


class WhisperTinyExtractor:
    """Frozen Whisper-tiny encoder + mean/std pooling -> 768-d vector per clip.

    Mirrors voicetypo_new's WhisperFeatureExtractor but switches model_id to
    `openai/whisper-tiny` (multilingual, d_model=384).
    """

    def __init__(self, model_id: str = "openai/whisper-tiny", device: str | None = None):
        from transformers import WhisperFeatureExtractor as HFFeat
        from transformers import WhisperModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feat = HFFeat.from_pretrained(model_id)
        self.model = WhisperModel.from_pretrained(model_id).to(self.device).eval()
        self.frame_hz = 50
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.config.d_model      # 384 for whisper-tiny
        self.target_sr = SAMPLE_RATE

    @torch.no_grad()
    def encode_frames(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if sr != self.target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        n_samples = len(audio)
        if n_samples == 0:
            return torch.zeros(0, self.embed_dim)
        n_frames = int(np.ceil(n_samples / self.target_sr * self.frame_hz))
        inputs = self.feat(audio, sampling_rate=self.target_sr, return_tensors="pt")
        x = inputs.input_features.to(self.device)
        out = self.model.encoder(x).last_hidden_state[0]   # (T_pad, D)
        return out[:n_frames].detach().cpu()

    @torch.no_grad()
    def pool_mean_std(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.shape[0] == 0:
            return torch.zeros(self.embed_dim * 2)
        mu = emb.mean(dim=0)
        sd = emb.std(dim=0) if emb.shape[0] > 1 else torch.zeros_like(mu)
        return torch.cat([mu, sd], dim=0)

    @torch.no_grad()
    def from_waveform_np(self, audio: np.ndarray) -> np.ndarray:
        emb = self.encode_frames(audio, sr=self.target_sr)
        v = self.pool_mean_std(emb)
        return v.numpy().astype(np.float32)

    @torch.no_grad()
    def from_path(self, path: Path | str) -> np.ndarray:
        wav = read_wav_mono(path).squeeze(0).cpu().numpy()
        return self.from_waveform_np(wav)


def extract_batch(paths: list[Path], device: str = "cpu") -> np.ndarray:
    """Extract MFCC tensors for many wav files. Returns (N, 3, n_mfcc, T) float32."""
    ext = MFCCExtractor(device=device)
    feats = []
    for p in paths:
        feats.append(ext.from_path(p).cpu().numpy().astype(np.float32))
    return np.stack(feats, axis=0) if feats else np.zeros(
        (0, 3, MFCC_N_MFCC, TARGET_FRAMES), dtype=np.float32
    )
