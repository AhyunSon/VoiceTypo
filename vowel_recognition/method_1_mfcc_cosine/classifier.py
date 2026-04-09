"""MFCC + 코사인 거리 실시간 모음 분류기.

HTML 버전 (Graph_ver_PSOLA_v2) 로직 그대로 재현.
- MFCC 13계수 (c1~c13, c0 건너뜀)
- FFT 2048, 멜 26밴드, 80~4000Hz
- Pre-emphasis 없음, CMVN 없음
- 캘리브레이션: 모음별 평균 MFCC → 프로토타입
- 분류: 코사인 거리 (최근접 프로토타입)

인터페이스: FormantVowelClassifier와 동일
  classify(chunk, f0=, fft_mag=, fft_freqs=) → (vowel, conf, f1, f2)
  reset()
  calibrate_start() / calibrate_feed(vowel, chunk) / calibrate_end()
"""

import os
import json
import numpy as np

VOWELS = ['아', '어', '오', '우', '으', '이', '에']

FFT_SIZE = 2048
NUM_MEL_BANDS = 26
NUM_COEFFS = 13
MIN_FREQ = 80
MAX_FREQ = 4000
MIN_RMS = 0.005


def _build_mel_filterbank(sr, fft_size, num_bands=NUM_MEL_BANDS,
                          min_freq=MIN_FREQ, max_freq=MAX_FREQ):
    num_bins = fft_size // 2 + 1
    bfw = sr / fft_size
    min_mel = 2595 * np.log10(1 + min_freq / 700)
    max_mel = 2595 * np.log10(1 + max_freq / 700)
    mel_pts = []
    for i in range(num_bands + 2):
        mel = min_mel + (max_mel - min_mel) * i / (num_bands + 1)
        mel_pts.append(700 * (10 ** (mel / 2595) - 1))
    bins = [int(np.floor(f / bfw)) for f in mel_pts]
    fb = np.zeros((num_bands, num_bins), dtype=np.float32)
    for m in range(num_bands):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        for k in range(l, min(c, num_bins)):
            if c != l:
                fb[m, k] = (k - l) / (c - l)
        for k in range(c, min(r + 1, num_bins)):
            if r != c:
                fb[m, k] = (r - k) / (r - c)
    return fb


def _build_dct_matrix(num_coeffs=NUM_COEFFS, num_bands=NUM_MEL_BANDS):
    matrix = np.zeros((num_coeffs + 1, num_bands), dtype=np.float32)
    for k in range(num_coeffs + 1):
        for n in range(num_bands):
            matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / num_bands)
    return matrix


class MfccCosineClassifier:
    """MFCC + 코사인 거리 실시간 모음 분류기."""

    def __init__(self, sample_rate=44100):
        self._sr = sample_rate
        self._prototypes = {}       # {vowel: mfcc_vector}
        self._calibrated = False
        self._is_active = False

        # 캐시
        self._fb = None
        self._dct = None
        self._window = None

        # 캘리브레이션 상태
        self._cal_mfccs = None  # {vowel: [mfcc, ...]}

        # EMA 스무딩 (프레임 간)
        self._smooth_mfcc = None
        self._smooth_alpha = 0.3  # 새 프레임 비중

        self._init_cache()

    def _init_cache(self):
        n_fft = FFT_SIZE
        self._fb = _build_mel_filterbank(self._sr, n_fft)
        self._dct = _build_dct_matrix()
        self._window = np.hanning(n_fft).astype(np.float32)

    def _extract_mfcc(self, audio_frame):
        """단일 프레임에서 MFCC 13계수 추출 (c1~c13)."""
        n = min(FFT_SIZE, len(audio_frame))
        windowed = audio_frame[:n] * self._window[:n]
        spectrum = np.abs(np.fft.rfft(windowed, n=n))

        fb = self._fb
        mel_e = np.zeros(NUM_MEL_BANDS, dtype=np.float32)
        for m in range(NUM_MEL_BANDS):
            mel_e[m] = np.sum(spectrum[:fb.shape[1]] ** 2 * fb[m])
        log_mel = np.log(np.maximum(mel_e, 1e-10))

        dct = self._dct
        mfcc = np.zeros(NUM_COEFFS, dtype=np.float32)
        for k in range(NUM_COEFFS):
            mfcc[k] = np.sum(log_mel * dct[k + 1])  # c1~c13 (c0 건너뜀)
        return mfcc

    @staticmethod
    def _cosine_distance(a, b):
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 2.0
        return 1 - dot / (na * nb)

    def classify(self, chunk, f0=None, fft_mag=None, fft_freqs=None):
        """오디오 청크 분류 → (vowel, confidence, f1, f2).

        FormantVowelClassifier와 동일한 인터페이스.
        f1, f2는 MFCC 방식에서 사용하지 않으므로 0.0 반환.
        """
        if not self._calibrated or not self._prototypes:
            return None, 0.0, 0.0, 0.0

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < MIN_RMS:
            return None, 0.0, 0.0, 0.0

        mfcc = self._extract_mfcc(chunk)

        # EMA 스무딩
        if self._smooth_mfcc is None or not self._is_active:
            self._smooth_mfcc = mfcc
        else:
            self._smooth_mfcc = (self._smooth_alpha * mfcc
                                 + (1 - self._smooth_alpha) * self._smooth_mfcc)
        self._is_active = True

        feat = self._smooth_mfcc

        # 코사인 거리로 최근접 프로토타입 찾기
        min_dist = float('inf')
        second_dist = float('inf')
        best = '아'

        for v, proto in self._prototypes.items():
            d = self._cosine_distance(feat, proto)
            if d < min_dist:
                second_dist = min_dist
                min_dist = d
                best = v
            elif d < second_dist:
                second_dist = d

        # 신뢰도: 1등과 2등 거리 차이 비율
        conf = 0.0
        if second_dist > 0:
            conf = min(1.0, (second_dist - min_dist) / second_dist)

        return best, conf, 0.0, 0.0

    def reset(self):
        """VAD 비활성 시 호출."""
        self._smooth_mfcc = None
        self._is_active = False

    # ── 캘리브레이션 ──

    def calibrate_start(self):
        """캘리브레이션 시작."""
        self._cal_mfccs = {v: [] for v in VOWELS}

    def calibrate_feed(self, vowel, chunk):
        """캘리브레이션용 오디오 프레임 제공."""
        if self._cal_mfccs is None:
            return
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < MIN_RMS:
            return
        mfcc = self._extract_mfcc(chunk)
        self._cal_mfccs[vowel].append(mfcc)

    def calibrate_end(self):
        """캘리브레이션 완료 → 프로토타입 계산."""
        if self._cal_mfccs is None:
            return False

        prototypes = {}
        for v in VOWELS:
            vecs = self._cal_mfccs[v]
            if len(vecs) >= 5:
                prototypes[v] = np.mean(vecs, axis=0).astype(np.float32)

        if len(prototypes) < len(VOWELS):
            missing = [v for v in VOWELS if v not in prototypes]
            print(f'[MFCC 캘리브레이션] 데이터 부족: {missing}')
            if len(prototypes) < 3:
                self._cal_mfccs = None
                return False

        self._prototypes = prototypes
        self._calibrated = True
        self._cal_mfccs = None

        print(f'[MFCC 캘리브레이션 완료] 프로토타입 {len(prototypes)}개')
        return True

    def save_calibration(self, path):
        """캘리브레이션 데이터 저장."""
        if not self._calibrated:
            return
        data = {v: proto.tolist() for v, proto in self._prototypes.items()}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'[MFCC] 캘리브레이션 저장: {path}')

    def load_calibration(self, path):
        """캘리브레이션 데이터 로드."""
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._prototypes = {v: np.array(vec, dtype=np.float32)
                            for v, vec in data.items()}
        self._calibrated = True
        print(f'[MFCC] 캘리브레이션 로드: {path} ({len(self._prototypes)}개 프로토타입)')
        return True

    @property
    def is_calibrated(self):
        return self._calibrated
