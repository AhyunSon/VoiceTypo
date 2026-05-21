"""Microphone capture, energy-based VAD, and WAV I/O."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import soundfile as sf


def read_wav(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def write_wav(path: str | Path, audio: np.ndarray, sr: int = 16000) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def dbfs(x: np.ndarray, eps: float = 1e-9) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)) + eps))
    return 20.0 * float(np.log10(rms + eps))


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    threshold_db: float = -38.0
    hangover_ms: int = 250
    min_segment_ms: int = 100
    max_segment_ms: int = 1200


class EnergyVAD:
    """Streaming energy-gate VAD. Yields (audio_segment, sr) on speech end."""

    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.frame_len = cfg.sample_rate * cfg.frame_ms // 1000
        self.hangover_frames = cfg.hangover_ms // cfg.frame_ms
        self.min_frames = cfg.min_segment_ms // cfg.frame_ms
        self.max_frames = cfg.max_segment_ms // cfg.frame_ms
        self._buf: list[np.ndarray] = []
        self._silence_streak = 0
        self._in_voice = False

    def _is_voice(self, frame: np.ndarray) -> bool:
        return dbfs(frame) > self.cfg.threshold_db

    def feed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Push one frame. Returns a finalized segment, or None."""
        if frame.shape[0] != self.frame_len:
            raise ValueError(
                f"VAD frame must be {self.frame_len} samples, got {frame.shape[0]}"
            )
        voiced = self._is_voice(frame)
        if voiced:
            self._buf.append(frame)
            self._silence_streak = 0
            self._in_voice = True
            if len(self._buf) >= self.max_frames:
                return self._flush(force=True)
            return None
        if self._in_voice:
            self._buf.append(frame)
            self._silence_streak += 1
            if self._silence_streak >= self.hangover_frames:
                return self._flush()
        return None

    def _flush(self, force: bool = False) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        seg = np.concatenate(self._buf)
        # trim trailing silence
        trailing = self._silence_streak * self.frame_len
        if trailing and not force:
            seg = seg[: max(0, len(seg) - trailing)]
        self._buf.clear()
        self._silence_streak = 0
        self._in_voice = False
        if seg.shape[0] // self.frame_len < self.min_frames:
            return None
        return seg


class MicStream:
    """Producer thread that pushes fixed-size mono float32 frames into a queue."""

    def __init__(self, sample_rate: int = 16000, frame_ms: int = 20, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.frame_len = sample_rate * frame_ms // 1000
        self.device = device
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._stream = None

    def __enter__(self):
        import sounddevice as sd

        def _cb(indata, frames, time_info, status):
            if status:
                pass  # under/overrun — drop silently
            mono = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            try:
                self._q.put_nowait(mono.astype(np.float32))
            except queue.Full:
                pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_len,
            device=self.device,
            callback=_cb,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._stop.set()

    def frames(self) -> Iterator[np.ndarray]:
        while not self._stop.is_set():
            try:
                yield self._q.get(timeout=0.5)
            except queue.Empty:
                continue


def list_input_devices() -> list[dict]:
    import sounddevice as sd
    devs = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devs)
        if d["max_input_channels"] > 0
    ]
