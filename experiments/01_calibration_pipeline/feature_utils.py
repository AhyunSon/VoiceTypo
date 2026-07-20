"""
VoiceTypo - MFCC 특징 추출 모듈 (RE:MARK 논문 방식)
=======================================
remark_exp 전용: log-mel 대신 MFCC를 사용한다.
BTS_6의 log-mel 버전과 비교 실험하기 위한 버전.
"""
import numpy as np
import sounddevice as sd

# ---- 설정 (BTS_6과 동일하게 유지: 공정 비교) ----
SAMPLE_RATE = 16000
RECORD_SEC_TRAIN = 1.0
RECORD_SEC_TEST  = 1.0
FRAME_LEN   = 480           # 30ms
HOP_LEN     = 240           # 15ms
N_MELS      = 40
FMIN_MEL    = 50
FMAX_MEL    = 8000
N_FFT       = FRAME_LEN
N_MFCC      = 13            # [MFCC] 사용할 켑스트럼 계수 개수 (앞 13개)
VOWELS      = ["아", "이", "우", "에", "오", "으", "어"]
TRAIN_FILE  = "train_data.npz"
MODEL_FILE  = "svm_model.pkl"

# ---- 멜 필터뱅크 ----
def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
def make_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_pts) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            if c > l: fb[m - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c: fb[m - 1, k] = (r - k) / (r - c)
    return fb
_MEL_FB = make_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS, FMIN_MEL, FMAX_MEL)
_WINDOW = np.hamming(FRAME_LEN)

# ---- DCT 행렬 (log-mel → MFCC 변환) ----
# [MFCC] 이 DCT 단계가 log-mel과 MFCC를 가르는 핵심 차이
_DCT = np.zeros((N_MFCC, N_MELS))
for i in range(N_MFCC):
    for j in range(N_MELS):
        _DCT[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N_MELS))

# ---- 특징 추출 ----
def mfcc_frame(frame):
    """한 프레임 → MFCC(13차원). log-mel에 DCT를 씌운 것."""
    spec = np.abs(np.fft.rfft(frame * _WINDOW, n=N_FFT)) ** 2
    log_mel = np.log(_MEL_FB @ spec + 1e-10)   # log-mel (40)
    mfcc = _DCT @ log_mel                        # [MFCC] DCT 변환 → (13)
    return mfcc

def trim_silence(signal, frame=FRAME_LEN, hop=HOP_LEN, thresh=0.003):
    """앞뒤 침묵을 잘라내고 소리 나는 구간만 반환. 소리 없으면 None."""
    energies = []
    for s in range(0, len(signal) - frame, hop):
        rms = np.sqrt(np.mean(signal[s:s+frame] ** 2))
        energies.append(rms)
    energies = np.array(energies)
    loud = np.where(energies >= thresh)[0]
    if len(loud) == 0:
        return None
    start = loud[0] * hop
    end   = loud[-1] * hop + frame          # loud[0]~loud[-1] 사이는 통째로 유지
    return signal[start:end]


def extract_features(signal):
    """
    음성 신호 → 26차원 MFCC 특징 벡터.
    [MFCC 평균(13) + MFCC delta 평균(13)]
    (BTS_6의 log-mel 80차원과 대비되는 MFCC 버전)
    """
    # ★ RMS 정규화: 신호 전체를 목표 볼륨(RMS 0.1)으로 맞춤 (학습·실시간 일치)
    rms = np.sqrt(np.mean(signal ** 2))
    if rms > 1e-6:                     # 완전 무음이 아니면
        signal = signal * (0.1 / rms)

    # ★ 앞뒤 침묵 트리밍: 짧은 발화에 섞인 침묵을 제거 (우→오/으 오분류 방지)
    signal = trim_silence(signal)
    if signal is None or len(signal) < FRAME_LEN:
        return None

    frames = []
    for s in range(0, len(signal) - FRAME_LEN, HOP_LEN):
        frame = signal[s:s + FRAME_LEN]
        if np.sqrt(np.mean(frame ** 2)) < 0.003:   # 무음 프레임 제외 (train_compare와 일치)
            continue
        frames.append(mfcc_frame(frame))

    if len(frames) < 3:
        return None

    frames = np.array(frames)                    # (T, 13)
    mean_mfcc = np.mean(frames, axis=0)          # (13,)
    delta = np.diff(frames, axis=0)              # (T-1, 13)
    mean_delta = np.mean(np.abs(delta), axis=0)  # (13,)

    return np.concatenate([mean_mfcc, mean_delta])  # (26,)

def record(seconds):
    """마이크 녹음."""
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]

# ---- CMN: 지수이동평균 기반 (훈련/실시간 공통) ----
CMN_ALPHA  = 0.995
CMN_WARMUP = 30      # 이 프레임 수 전까지는 보정 없이 평균만 학습

class StreamingCMN:
    def __init__(self, alpha=CMN_ALPHA, warmup=CMN_WARMUP):
        self.alpha = alpha
        self.warmup = warmup
        self.mean = None
        self.count = 0

    def reset(self):
        self.mean = None
        self.count = 0

    def apply(self, feat):
        if self.mean is None:
            self.mean = feat.copy()
        else:
            self.mean = self.alpha * self.mean + (1 - self.alpha) * feat
        self.count += 1
        if self.count < self.warmup:
            return feat
        return feat - self.mean