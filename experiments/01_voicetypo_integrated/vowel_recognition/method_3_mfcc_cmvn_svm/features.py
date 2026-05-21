"""MFCC 추출 + CMVN 정규화.

numpy + scipy만으로 MFCC 13계수 추출.
CMVN으로 화자/마이크 특성 정규화.
"""

import numpy as np
from scipy.fftpack import dct

# MFCC 파라미터
N_MFCC = 13
N_MELS = 26
N_FFT = 2048
FMIN = 0.0
FMAX = 8000.0


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """멜 필터뱅크 생성."""
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(min(fmax, sr / 2))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            if center > left:
                fbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fbank[i, j] = (right - j) / (right - center)
    return fbank


# 캐시
_filterbank_cache = {}


def extract_mfcc(audio: np.ndarray, sr: int = 44100) -> np.ndarray:
    """오디오 프레임에서 MFCC 13계수 추출.
    Args:
        audio: float32 배열 (1D, 보통 2048 샘플)
        sr: 샘플레이트
    Returns:
        np.ndarray: shape (13,) MFCC 벡터
    """
    n_fft = min(N_FFT, len(audio))

    # 프리엠퍼시스
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # 해밍 윈도우 + FFT
    windowed = emphasized[:n_fft] * np.hamming(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2

    # 멜 필터뱅크
    cache_key = (sr, n_fft, N_MELS)
    if cache_key not in _filterbank_cache:
        _filterbank_cache[cache_key] = _mel_filterbank(sr, n_fft, N_MELS, FMIN, FMAX)
    fbank = _filterbank_cache[cache_key]

    # 멜 스펙트럼 → 로그 → DCT
    mel_spec = fbank @ spectrum
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)
    mfcc = dct(log_mel, type=2, norm='ortho')[:N_MFCC]

    return mfcc.astype(np.float32)


class CMVN:
    """Cepstral Mean Variance Normalization.

    슬라이딩 윈도우 방식으로 MFCC 정규화.
    화자/마이크 특성, 볼륨 변화를 자동 상쇄.
    """

    def __init__(self, window_size: int = 50):
        self._window_size = window_size
        self._buffer = []
        self._mean = np.zeros(N_MFCC, dtype=np.float64)
        self._var = np.ones(N_MFCC, dtype=np.float64)

    def update(self, mfcc: np.ndarray):
        """MFCC 벡터를 버퍼에 추가하고 통계 갱신."""
        self._buffer.append(mfcc.astype(np.float64))
        if len(self._buffer) > self._window_size:
            self._buffer.pop(0)

        if len(self._buffer) >= 3:
            arr = np.array(self._buffer)
            self._mean = np.mean(arr, axis=0)
            self._var = np.var(arr, axis=0) + 1e-8

    def normalize(self, mfcc: np.ndarray) -> np.ndarray:
        """CMVN 적용."""
        return ((mfcc.astype(np.float64) - self._mean)
                / np.sqrt(self._var)).astype(np.float32)

    def reset(self):
        self._buffer.clear()
        self._mean = np.zeros(N_MFCC, dtype=np.float64)
        self._var = np.ones(N_MFCC, dtype=np.float64)
