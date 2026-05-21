"""
mfcc_svm.py — MFCC+CMVN 추출 + SVM 모음 분류기

역할: 포먼트 기반 분류기의 보조 검증자
  - 보정 데이터로 사용자 목소리 학습
  - 포먼트 기반 결과와 교차 검증
  - 일치 시 신뢰도 상승 / 불일치 시 낮은 쪽 기각

VoiceTypo/vowel_recognition/method_3_mfcc_cmvn_svm 기반,
외부 의존성 없이 numpy+scipy+sklearn만 사용.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.fftpack import dct
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from config import SAMPLE_RATE

SVM_FILE    = Path(__file__).parent / "user_svm_model.pkl"
N_MFCC      = 13
N_MELS      = 26
N_FFT       = 2048
FMIN        = 60.0
FMAX        = min(8000.0, SAMPLE_RATE / 2)
CMVN_WIN    = 60     # 슬라이딩 윈도우 크기 (프레임)


# ── 멜 필터뱅크 캐시 ────────────────────────────────────────
_filterbank_cache: dict = {}


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float, fmax: float) -> np.ndarray:
    key = (sr, n_fft, n_mels)
    if key in _filterbank_cache:
        return _filterbank_cache[key]

    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz2mel(fmin), hz2mel(min(fmax, sr / 2)), n_mels + 2)
    hz_pts  = mel2hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = bins[i], bins[i + 1], bins[i + 2]
        for j in range(lo, c):
            if c > lo:
                fb[i, j] = (j - lo) / (c - lo)
        for j in range(c, hi):
            if hi > c:
                fb[i, j] = (hi - j) / (hi - c)

    _filterbank_cache[key] = fb
    return fb


def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    오디오 청크 → MFCC 13계수 (numpy+scipy, 외부 라이브러리 불필요)
    """
    n_fft = min(N_FFT, len(audio))
    if n_fft < 64:
        return np.zeros(N_MFCC, dtype=np.float32)

    # 프리엠퍼시스
    sig = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # 해밍 윈도우 + 파워 스펙트럼
    win  = np.hamming(n_fft)
    spec = np.abs(np.fft.rfft(sig[:n_fft] * win, n=n_fft)) ** 2

    # 멜 필터뱅크
    fb = _mel_filterbank(sr, n_fft, N_MELS, FMIN, FMAX)

    mel  = fb @ spec
    mel  = np.maximum(mel, 1e-10)
    mfcc = dct(np.log(mel), type=2, norm='ortho')[:N_MFCC]
    return mfcc.astype(np.float32)


# ── CMVN (Cepstral Mean Variance Normalization) ─────────────

class _CMVN:
    def __init__(self, win: int = CMVN_WIN):
        self._win  = win
        self._buf  = []
        self._mean = np.zeros(N_MFCC, dtype=np.float64)
        self._var  = np.ones(N_MFCC,  dtype=np.float64)

    def update(self, mfcc: np.ndarray):
        self._buf.append(mfcc.astype(np.float64))
        if len(self._buf) > self._win:
            self._buf.pop(0)
        if len(self._buf) >= 3:
            a          = np.array(self._buf)
            self._mean = np.mean(a, axis=0)
            self._var  = np.var(a,  axis=0) + 1e-8

    def normalize(self, mfcc: np.ndarray) -> np.ndarray:
        return ((mfcc.astype(np.float64) - self._mean)
                / np.sqrt(self._var)).astype(np.float32)

    def reset(self):
        self._buf  = []
        self._mean = np.zeros(N_MFCC, dtype=np.float64)
        self._var  = np.ones(N_MFCC,  dtype=np.float64)


# ── SVM 분류기 ────────────────────────────────────────────────

class MfccSvmClassifier:
    """
    보정 후 자동 학습되는 MFCC+SVM 모음 분류기.
    포먼트 기반 분류기의 보조 검증자로 동작.
    """

    def __init__(self):
        self._cmvn    = _CMVN()
        self._svm: SVC | None = None
        self._scaler  = StandardScaler()
        self._trained = False

        # 보정 데이터 수집용
        self._cal_buf: dict = {}   # vowel → [mfcc_vec, ...]

    # ── 보정 데이터 수집 ──────────────────────────────────────

    def calib_feed(self, vowel: str, audio: np.ndarray):
        """보정 중 오디오 청크 입력 (모음 레이블 + 오디오)"""
        mfcc = extract_mfcc(audio, SAMPLE_RATE)
        self._cmvn.update(mfcc)
        norm = self._cmvn.normalize(mfcc)
        self._cal_buf.setdefault(vowel, []).append(norm)

    def train(self) -> bool:
        """수집된 보정 데이터로 SVM 학습"""
        X, y = [], []
        for vowel, vecs in self._cal_buf.items():
            for v in vecs:
                X.append(v)
                y.append(vowel)

        if len(set(y)) < 2:
            return False

        X = np.array(X, dtype=np.float32)
        y = np.array(y)

        self._scaler.fit(X)
        Xs = self._scaler.transform(X)

        self._svm = SVC(kernel='rbf', probability=True,
                        C=10.0, gamma='scale')
        self._svm.fit(Xs, y)
        self._trained = True
        return True

    # ── 실시간 추론 ───────────────────────────────────────────

    def predict(self, audio: np.ndarray) -> tuple:
        """
        Returns: (vowel: str or None, confidence: float)
        None = 미학습 상태
        """
        if not self._trained or self._svm is None:
            return None, 0.0

        mfcc = extract_mfcc(audio, SAMPLE_RATE)
        self._cmvn.update(mfcc)
        norm = self._cmvn.normalize(mfcc)
        xs   = self._scaler.transform(norm.reshape(1, -1))

        pred  = self._svm.predict(xs)[0]
        proba = self._svm.predict_proba(xs)[0]
        conf  = float(np.max(proba))
        return str(pred), conf

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── 저장 / 불러오기 ──────────────────────────────────────

    def save(self):
        data = {
            "svm":     pickle.dumps(self._svm).hex() if self._svm else None,
            "scaler":  pickle.dumps(self._scaler).hex(),
            "trained": self._trained,
            "cal_buf": {v: [vec.tolist() for vec in vecs]
                        for v, vecs in self._cal_buf.items()},
        }
        SVM_FILE.write_text(json.dumps(data, ensure_ascii=False),
                            encoding="utf-8")

    def load(self) -> bool:
        if not SVM_FILE.exists():
            return False
        try:
            data = json.loads(SVM_FILE.read_text(encoding="utf-8"))
            if data["svm"]:
                self._svm = pickle.loads(bytes.fromhex(data["svm"]))
            self._scaler  = pickle.loads(bytes.fromhex(data["scaler"]))
            self._trained = data["trained"]
            self._cal_buf = {
                v: [np.array(vec, dtype=np.float32) for vec in vecs]
                for v, vecs in data["cal_buf"].items()
            }
            return self._trained
        except Exception:
            return False


# ── 분류 결합 (포먼트 + SVM) ─────────────────────────────────

def combine_decisions(vowel_formant: str, conf_formant: float,
                      vowel_svm:     str | None,
                      conf_svm:      float) -> tuple:
    """
    포먼트 기반 결과와 SVM 결과를 결합.

    Returns: (final_vowel, final_confidence)
    """
    if vowel_svm is None:
        return vowel_formant, conf_formant

    if vowel_formant == vowel_svm:
        # 두 방법 일치 → 신뢰도 상승 (최고값 + 보너스)
        conf = min(1.0, max(conf_formant, conf_svm) + 0.10)
        return vowel_formant, conf

    # 불일치 → 더 높은 신뢰도 쪽 채택, 전체 신뢰도 20% 페널티
    if conf_svm > conf_formant + 0.15:
        return vowel_svm,     conf_svm     * 0.80
    elif conf_formant > conf_svm + 0.15:
        return vowel_formant, conf_formant * 0.80
    else:
        # 신뢰도가 비슷하면 포먼트 기반 우선
        return vowel_formant, conf_formant * 0.70
