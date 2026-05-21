"""
audio_stream.py — sounddevice 오디오 캡처 모듈
thread-safe deque 버퍼로 최근 오디오 데이터 유지
"""

import threading
import collections
import numpy as np
import sounddevice as sd

from config import SAMPLE_RATE, CHANNELS, BLOCK_SIZE, ANALYSIS_WIN_SEC


class AudioStream:
    """마이크 입력을 deque 링버퍼에 저장하는 클래스"""

    def __init__(self, device=None):
        # 분석 윈도우 × 4 만큼 버퍼 확보
        buf_cap = int(SAMPLE_RATE * ANALYSIS_WIN_SEC * 4)
        self.buf    = collections.deque(maxlen=buf_cap)
        self.lock   = threading.Lock()
        self._device = device
        self._stream = None
        self._create_stream()

    # ── 장치 목록 ────────────────────────────────────────────────
    @staticmethod
    def get_input_devices():
        """(index, name) 튜플 목록 반환"""
        result = []
        try:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0:
                    result.append((i, d["name"]))
        except Exception:
            pass
        return result

    # ── 스트림 생성 ──────────────────────────────────────────────
    def _create_stream(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._callback,
            device=self._device,
        )

    def _callback(self, indata, frames, time_info, status):
        with self.lock:
            self.buf.extend(indata[:, 0].copy())

    def start(self):
        self._stream.start()

    def stop(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    # ── 장치 교체 (UI에서 호출) ──────────────────────────────────
    def change_device(self, device):
        """스트림을 중지하고 새 장치로 재시작"""
        self.stop()
        with self.lock:
            self.buf.clear()
        self._device = device
        self._create_stream()
        self._stream.start()

    def get_chunk(self, n_samples: int):
        """
        최근 n_samples개의 샘플을 float64 ndarray로 반환.
        샘플이 부족하면 None 반환.
        """
        with self.lock:
            if len(self.buf) < n_samples:
                return None
            return np.array(list(self.buf)[-n_samples:], dtype=np.float64)
