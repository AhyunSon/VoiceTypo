"""마이크 실시간 오디오 캡처.

sounddevice 콜백 기반 (20ms 청크).

사용:
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
DEFAULT_BLOCKSIZE = 882   # 44100 * 0.02 = 882 samples = 20ms
DEFAULT_CHANNELS = 1


class AudioCapture:
    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE,
                 blocksize=DEFAULT_BLOCKSIZE,
                 device=None):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.device = device
        self._listeners: List[AudioCallback] = []
        self._stream: Optional[sd.InputStream] = None

    def add_listener(self, callback: AudioCallback):
        self._listeners.append(callback)

    def remove_listener(self, callback: AudioCallback):
        self._listeners.remove(callback)

    def _on_audio(self, indata, frames, time_info, status):
        if status:
            print(f"[AudioCapture] {status}")
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
    def is_running(self):
        return self._stream is not None and self._stream.active

    @staticmethod
    def list_devices():
        return sd.query_devices()

    @staticmethod
    def default_device():
        return sd.query_devices(kind='input')
