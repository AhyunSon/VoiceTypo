"""LPC 기반 포먼트(F1/F2/F3) 추출.

scipy Levinson-Durbin LPC (22차, 44100Hz).
NaN 발생 시 이전 유효값으로 보간.
시뮬레이션 모드: 마이크 없이 합성 포먼트 궤적 생성.
"""

import numpy as np
from scipy.signal import lfilter

# ─── 상수 ───
LPC_ORDER = 22
DEFAULT_SR = 44100
PRE_EMPHASIS = 0.97

# 포먼트 유효 범위 (Hz)
F1_RANGE = (180, 1050)
F2_RANGE = (400, 3200)
F3_RANGE = (2200, 3700)

# 대역폭 상한 (Hz)
MAX_BANDWIDTH = 500

# 모음 참조 포먼트 (남녀 평균, Hz)
VOWEL_FORMANTS = {
    '아': (798, 1369, 2748),
    '이': (255, 2524, 3366),
    '우': (335,  703, 2579),
    '오': (346,  726, 2593),
    '으': (354, 1485, 2606),
    '어': (511,  903, 2817),
    '에': (480, 2142, 2836),
}


def _levinson_durbin(r, order):
    """Levinson-Durbin 알고리즘 → LPC 계수."""
    a = np.zeros(order + 1)
    e = r[0]
    a[0] = 1.0

    for i in range(1, order + 1):
        acc = np.dot(a[1:i][::-1], r[1:i]) + r[i]
        k = -acc / max(e, 1e-12)
        a[1:i] = a[1:i] + k * a[1:i][::-1].copy()
        a[i] = k
        e *= (1 - k * k)
        if e <= 0:
            break

    return a, e


def extract_formants(audio, sr=DEFAULT_SR, order=LPC_ORDER):
    """오디오 프레임에서 F1, F2, F3 추출.

    Args:
        audio: float32 1D 배열 (한 프레임, ~20ms)
        sr: 샘플레이트
        order: LPC 차수

    Returns:
        (f1, f2, f3) Hz 튜플. 추출 실패 시 (nan, nan, nan).
    """
    NAN3 = (float('nan'), float('nan'), float('nan'))

    if len(audio) < order + 1:
        return NAN3

    # 프리엠퍼시스
    emphasized = np.empty_like(audio)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - PRE_EMPHASIS * audio[:-1]

    # 해밍 윈도우
    windowed = emphasized * np.hamming(len(emphasized))

    # 자기상관
    n = len(windowed)
    r = np.correlate(windowed, windowed, mode='full')[n - 1: n + order]
    if r[0] < 1e-10:
        return NAN3

    # LPC
    a, e = _levinson_durbin(r, order)
    if e <= 0:
        return NAN3

    # 다항식 근
    roots = np.roots(a)

    # 허수부 양수인 근만 (대칭 → 절반)
    roots = roots[np.imag(roots) > 0.01]
    if len(roots) == 0:
        return NAN3

    # 주파수·대역폭
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2.0 * np.pi))
    bandwidths = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.abs(roots) + 1e-12)

    # 대역폭 필터
    valid = bandwidths < MAX_BANDWIDTH
    freqs = freqs[valid]

    if len(freqs) == 0:
        return NAN3

    freqs = np.sort(freqs)

    # F1, F2, F3 순차 할당 — F1 < F2 < F3 강제
    f1 = float('nan')
    f2 = float('nan')
    f3 = float('nan')

    # F1: F1_RANGE 내 가장 낮은 후보
    f1_cands = freqs[(freqs >= F1_RANGE[0]) & (freqs <= F1_RANGE[1])]
    if len(f1_cands) > 0:
        f1 = float(f1_cands[0])

    # F2: F2_RANGE 내에서 F1보다 큰 후보 중 가장 낮은 것
    f2_floor = f1 + 100 if not np.isnan(f1) else F2_RANGE[0]
    f2_cands = freqs[(freqs >= max(F2_RANGE[0], f2_floor)) & (freqs <= F2_RANGE[1])]
    if len(f2_cands) > 0:
        f2 = float(f2_cands[0])

    # F3: F3_RANGE 내에서 F2보다 큰 후보 중 가장 낮은 것
    f3_floor = f2 + 100 if not np.isnan(f2) else F3_RANGE[0]
    f3_cands = freqs[(freqs >= max(F3_RANGE[0], f3_floor)) & (freqs <= F3_RANGE[1])]
    if len(f3_cands) > 0:
        f3 = float(f3_cands[0])

    return (f1, f2, f3)


class FormantTracker:
    """Praat(Parselmouth) 기반 포먼트 추출 + EMA 스무딩.

    사용:
        tracker = FormantTracker()
        f1, f2, f3 = tracker.update(audio_chunk)
    """

    def __init__(self, sr=DEFAULT_SR, order=LPC_ORDER, smooth_alpha=0.2):
        self.sr = sr
        self.order = order
        self._smooth_alpha = smooth_alpha
        self._prev = (float('nan'), float('nan'), float('nan'))
        self._smooth = (float('nan'), float('nan'), float('nan'))
        self._use_praat = False

        try:
            import parselmouth
            self._parselmouth = parselmouth
            self._use_praat = True
        except ImportError:
            print("[FormantTracker] parselmouth not found, falling back to LPC")

    def _extract_praat(self, audio):
        """Parselmouth로 포먼트 추출."""
        snd = self._parselmouth.Sound(audio, sampling_frequency=self.sr)
        formant = snd.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=5500.0,
            window_length=0.025,
            pre_emphasis_from=50.0,
        )
        t = formant.get_time_from_frame_number(1)
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        # Praat returns undefined as nan-like, convert
        f1 = float(f1) if f1 and f1 > 0 else float('nan')
        f2 = float(f2) if f2 and f2 > 0 else float('nan')
        f3 = float(f3) if f3 and f3 > 0 else float('nan')
        return (f1, f2, f3)

    def update(self, audio):
        """한 프레임 처리. Praat 추출 + NaN 보간 + EMA 스무딩."""
        if self._use_praat:
            try:
                f1, f2, f3 = self._extract_praat(audio)
            except Exception:
                f1, f2, f3 = extract_formants(audio, self.sr, self.order)
        else:
            f1, f2, f3 = extract_formants(audio, self.sr, self.order)

        # NaN 보간
        if np.isnan(f1) and not np.isnan(self._prev[0]):
            f1 = self._prev[0]
        if np.isnan(f2) and not np.isnan(self._prev[1]):
            f2 = self._prev[1]
        if np.isnan(f3) and not np.isnan(self._prev[2]):
            f3 = self._prev[2]

        self._prev = (f1, f2, f3)

        # EMA 시간 스무딩
        a = self._smooth_alpha
        sf1 = f1 if np.isnan(self._smooth[0]) else (a * f1 + (1 - a) * self._smooth[0]) if not np.isnan(f1) else self._smooth[0]
        sf2 = f2 if np.isnan(self._smooth[1]) else (a * f2 + (1 - a) * self._smooth[1]) if not np.isnan(f2) else self._smooth[1]
        sf3 = f3 if np.isnan(self._smooth[2]) else (a * f3 + (1 - a) * self._smooth[2]) if not np.isnan(f3) else self._smooth[2]

        self._smooth = (sf1, sf2, sf3)
        return (sf1, sf2, sf3)

    def reset(self):
        self._prev = (float('nan'), float('nan'), float('nan'))
        self._smooth = (float('nan'), float('nan'), float('nan'))


# ─── 시뮬레이션 모드 ───

class FormantSimulator:
    """마이크 없이 포먼트 궤적을 생성하는 시뮬레이터.

    지정된 모음 시퀀스를 순환하며, 각 모음의 참조 포먼트에
    자연스러운 노이즈를 추가하여 시뮬레이션.

    사용:
        sim = FormantSimulator(['아', '이', '우'], hold_frames=25)
        f1, f2, f3 = sim.next_frame()
    """

    def __init__(self, vowel_sequence=None, hold_frames=25,
                 transition_frames=10, noise_hz=15.0):
        if vowel_sequence is None:
            vowel_sequence = list(VOWEL_FORMANTS.keys())
        self.sequence = vowel_sequence
        self.hold_frames = hold_frames
        self.transition_frames = transition_frames
        self.noise_hz = noise_hz

        self._idx = 0
        self._frame = 0
        self._phase = 'hold'  # 'hold' | 'transition'
        self._current = np.array(VOWEL_FORMANTS[self.sequence[0]], dtype=np.float64)
        self._target = self._current.copy()

    def next_frame(self):
        """다음 프레임의 (f1, f2, f3) 반환."""
        if self._phase == 'hold':
            self._frame += 1
            if self._frame >= self.hold_frames:
                self._frame = 0
                self._phase = 'transition'
                self._idx = (self._idx + 1) % len(self.sequence)
                self._target = np.array(
                    VOWEL_FORMANTS[self.sequence[self._idx]], dtype=np.float64)
        else:  # transition
            self._frame += 1
            t = self._frame / self.transition_frames
            self._current = self._current + (self._target - self._current) * min(t, 1.0)
            if self._frame >= self.transition_frames:
                self._frame = 0
                self._phase = 'hold'
                self._current = self._target.copy()

        # 자연스러운 노이즈
        noise = np.random.randn(3) * self.noise_hz
        result = self._current + noise
        return (float(result[0]), float(result[1]), float(result[2]))

    def reset(self):
        self._idx = 0
        self._frame = 0
        self._phase = 'hold'
        self._current = np.array(
            VOWEL_FORMANTS[self.sequence[0]], dtype=np.float64)
        self._target = self._current.copy()
