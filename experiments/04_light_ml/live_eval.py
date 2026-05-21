"""Live-test helpers shared by 04_live_test_v3.py and 06_evaluate_wav_folder_v3.py.

UnifiedClassifier dispatches over checkpoint kind so the SAME wav file can be
classified by any of v1 / v2 / v3 (MFCC or log-mel + CNN) or v4 (Whisper-tiny
encoder + MLP probe). Filename of the wav still encodes the target label, so
methods can be swapped without re-recording.

vowel_core mirrors voicetypo_new/voicetypo/live_eval.py: trim leading/trailing
silence by an RMS threshold, then take the 30–90% middle slice — the same span
the training extractor used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from voicetypo_light import (
    DATA_DIR,
    MFCC_N_MFCC,
    SAMPLE_RATE,
    TARGET_FRAMES,
    VOWEL_CLASSES,
)
from voicetypo_light.features import (
    LogMelExtractor,
    MFCCExtractor,
    WhisperTinyExtractor,
)
from voicetypo_light.model import load_checkpoint


# ---------------------------------------------------------------------------
# segment selection
# ---------------------------------------------------------------------------

def trim_silence(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    threshold_db: float = -38.0,
    frame_ms: int = 20,
) -> np.ndarray:
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


def vowel_core(
    wav: np.ndarray,
    sr: int = SAMPLE_RATE,
    segment_lo: float = 0.30,
    segment_hi: float = 0.90,
    threshold_db: float = -38.0,
) -> np.ndarray:
    """Match the training-time extractor: trim silence, then 30–90% middle slice."""
    trimmed = trim_silence(wav, sr=sr, threshold_db=threshold_db)
    if len(trimmed) < int(0.05 * sr):
        trimmed = wav
    n = len(trimmed)
    i0 = int(n * segment_lo)
    i1 = int(n * segment_hi)
    core = trimmed[i0:i1] if i1 > i0 else trimmed
    if len(core) < int(0.04 * sr):
        core = trimmed
    return core.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# unified classifier
# ---------------------------------------------------------------------------

def detect_version(extra: dict, ckpt_path: str | Path) -> str:
    if "version" in extra:
        return str(extra["version"])
    name = Path(ckpt_path).name
    m = re.search(r"v(\d+)", name)
    if m:
        return f"v{m.group(1)}"
    if name.startswith("small_cnn"):
        return "v1"
    return "v1"


class UnifiedClassifier:
    """Loads any voicetypo_light checkpoint and runs end-to-end inference on
    a raw waveform (np.float32 array at SAMPLE_RATE)."""

    def __init__(self, ckpt_path: str | Path, device: str | None = None):
        self.ckpt_path = Path(ckpt_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.classes, self.extra = load_checkpoint(self.ckpt_path)
        self.model = self.model.to(self.device).eval()
        self.kind = self.extra.get("model_kind", "small")
        self.version = detect_version(self.extra, self.ckpt_path)

        if self.kind == "mlp":
            encoder_id = self.extra.get("encoder_id", "openai/whisper-tiny")
            self.extractor = WhisperTinyExtractor(model_id=encoder_id, device=self.device)
            self.norm = None
            self.target_frames = None
            self.feature_kind = "whisper"
        else:
            self.feature_kind = self.extra.get("feature_kind", "mfcc")
            self.target_frames = int(self.extra.get("target_frames", TARGET_FRAMES))
            if self.feature_kind == "logmel":
                self.extractor = LogMelExtractor(
                    device=self.device, target_frames=self.target_frames
                )
            else:
                self.extractor = MFCCExtractor(
                    device=self.device, target_frames=self.target_frames
                )
            self.norm = self._load_norm()

    def _load_norm(self) -> tuple[np.ndarray, np.ndarray] | None:
        candidates = [
            DATA_DIR / f"features_{self.version}" / "norm_stats.npz",
        ]
        if self.version == "v1":
            candidates.append(DATA_DIR / "features" / "norm_stats.npz")
        for p in candidates:
            if p.exists():
                d = np.load(p)
                return d["mean"].astype(np.float32), d["std"].astype(np.float32)
        print(f"[live] WARNING: norm_stats not found for {self.version} "
              f"(tried: {[str(c) for c in candidates]})")
        return None

    def classify(self, wav: np.ndarray) -> tuple[int, np.ndarray]:
        """wav: 1-D float32 array @ SAMPLE_RATE. Returns (pred_idx, probs[7])."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if self.kind == "mlp":
            v = self.extractor.from_waveform_np(wav)
            x = torch.from_numpy(v).unsqueeze(0).to(self.device)
        else:
            wav_t = torch.from_numpy(wav)
            feat = self.extractor.from_waveform(wav_t)         # (3, F, T)
            if self.norm is not None:
                feat_np = feat.cpu().numpy()
                mean, std = self.norm
                feat_np = (feat_np - mean) / (std + 1e-6)
                feat = torch.from_numpy(feat_np.astype(np.float32))
            x = feat.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)[0].detach().cpu().numpy().astype(np.float64)
        logits = logits - logits.max()
        e = np.exp(logits)
        probs = (e / e.sum()).astype(np.float32)
        return int(np.argmax(probs)), probs


# ---------------------------------------------------------------------------
# label parsing helpers
# ---------------------------------------------------------------------------

DISPLAY_TO_LABEL = {"아": "a", "에": "e", "이": "i", "오": "o",
                    "우": "u", "으": "eu", "어": "eo"}


def parse_target(text: str, classes: list[str] | None = None) -> str | None:
    """Accept either Hangul (아/에/이/오/우/으/어) or latin code (a/e/i/o/u/eu/eo)."""
    classes = classes or VOWEL_CLASSES
    s = text.strip()
    if not s:
        return None
    if s in DISPLAY_TO_LABEL:
        return DISPLAY_TO_LABEL[s]
    if s.lower() in classes:
        return s.lower()
    return None


_FNAME_RE = re.compile(r"^([a-z]+)_(\d+)(?:_.*)?\.wav$", re.IGNORECASE)


def parse_target_from_filename(path: Path, classes: list[str] | None = None) -> str | None:
    classes = classes or VOWEL_CLASSES
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    label = m.group(1).lower()
    return label if label in classes else None
