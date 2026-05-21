"""마이크 실시간 오디오 캡처 모듈.

sounddevice 콜백 기반. 오디오 청크가 도착할 때마다
등록된 콜백 함수들을 호출한다.

사용법:
    cap = AudioCapture()
    cap.add_listener(my_callback)  # callback(audio: np.ndarray, sr: int)
    cap.start()
    ...
    cap.stop()
"""

import numpy as np
import sounddevice as sd
from typing import Callable, List, Optional

AudioCallback = Callable[[np.ndarray, int], None]

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BLOCKSIZE = 2048
DEFAULT_CHANNELS = 1


class AudioCapture:
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE,
                 blocksize: int = DEFAULT_BLOCKSIZE,
                 device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.device = device
        self._listeners: List[AudioCallback] = []
        self._stream: Optional[sd.InputStream] = None

    def add_listener(self, callback: AudioCallback):
        """콜백 등록. callback(audio_chunk, sample_rate) 형태."""
        self._listeners.append(callback)

    def remove_listener(self, callback: AudioCallback):
        self._listeners.remove(callback)

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            print(f"[AudioCapture] {status}")
        # indata: (frames, channels) float32 → 모노 1D로 변환
        chunk = indata[:, 0].copy()
        for cb in self._listeners:
            cb(chunk, self.sample_rate)

    def start(self):
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=DEFAULT_CHANNELS,
            dtype='float32',
            device=self.device,
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    @staticmethod
    def list_devices():
        """사용 가능한 오디오 디바이스 목록."""
        return sd.query_devices()

    @staticmethod
    def default_device():
        """기본 입력 디바이스 정보."""
        return sd.query_devices(kind='input')
