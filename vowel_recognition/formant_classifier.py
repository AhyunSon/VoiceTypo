"""포먼트 기반 실시간 한국어 모음 분류기.

v3: 하모닉 봉투 포먼트 추출 + LPC 폴백 + 오/우 보조 피처 + Lobanov 정규화.

추출 우선순위:
  1. 하모닉 봉투 (F0 + FFT → 하모닉 위치 진폭 → 봉투 피크)
  2. LPC 폴백 (F0 미감지 시)

사용법:
    clf = FormantVowelClassifier(sample_rate=44100)
    vowel, conf, f1, f2 = clf.classify(chunk, f0=freq, fft_mag=mag, fft_freqs=freqs)
"""

import numpy as np
from scipy.signal import decimate


# ═══════════════════════════════════════════════════
#  한국어 7모음 F1/F2 기본 프로토타입 (Hz)
# ═══════════════════════════════════════════════════

DEFAULT_PROTOTYPES = {
    '아':  (750,  1200),
    '어':  (600,  1050),
    '오':  (450,   850),
    '우':  (350,   750),
    '으':  (400,  1500),
    '이':  (300,  2200),
    '에':  (500,  1800),
}

F1_RANGE = 500.0
F2_RANGE = 1500.0

CONFIDENCE_SCALE = 200.0
MIN_RMS = 0.005


# ═══════════════════════════════════════════════════
#  하모닉 봉투 포먼트 추출
# ═══════════════════════════════════════════════════

def _extract_formants_harmonic(fft_mag, fft_freqs, f0, sr,
                                max_freq=5000.0, n_formants=3):
    """하모닉 봉투에서 포먼트 추출.

    F0의 정수배(하모닉) 위치에서 FFT 진폭을 읽어 봉투를 구성하고,
    봉투의 피크를 포먼트로 반환한다.
    """
    if f0 <= 0 or f0 > 1000:
        return [0.0] * n_formants

    freq_res = fft_freqs[1] - fft_freqs[0] if len(fft_freqs) > 1 else sr / (2 * len(fft_mag))

    # ── F0 에너지 검증: YIN octave error 방지 ──
    f0_bin = int(round(f0 / freq_res))
    if 0 < f0_bin < len(fft_mag):
        local_max = float(np.max(fft_mag[max(0, f0_bin - 2):f0_bin + 3]))
        global_max = float(np.max(fft_mag))
        if global_max > 0 and local_max < global_max * 0.01:
            return [0.0] * n_formants

    # ── 하모닉 위치에서 진폭 읽기 (parabolic interpolation) ──
    harmonic_freqs = []
    harmonic_amps = []

    k = 1
    while k * f0 < max_freq:
        hf = k * f0
        bin_idx = int(round(hf / freq_res))
        if bin_idx <= 0 or bin_idx >= len(fft_mag) - 1:
            k += 1
            continue

        amp_l = fft_mag[bin_idx - 1]
        amp_c = fft_mag[bin_idx]
        amp_r = fft_mag[bin_idx + 1]
        peak_amp = float(max(amp_l, amp_c, amp_r))

        if peak_amp > 0:
            # parabolic interpolation on log-magnitude
            log_l = np.log(amp_l + 1e-10)
            log_c = np.log(amp_c + 1e-10)
            log_r = np.log(amp_r + 1e-10)
            denom = log_l - 2.0 * log_c + log_r
            if abs(denom) > 1e-10:
                delta = 0.5 * (log_l - log_r) / denom
                delta = max(-1.0, min(1.0, delta))
                precise_freq = fft_freqs[bin_idx] + delta * freq_res
            else:
                precise_freq = hf
            harmonic_freqs.append(precise_freq)
            harmonic_amps.append(peak_amp)

        k += 1

    if len(harmonic_freqs) < 4:
        return [0.0] * n_formants

    harmonic_freqs = np.array(harmonic_freqs)
    harmonic_amps = np.array(harmonic_amps)

    # ── dB 스케일 + 3점 스무딩 ──
    harmonic_db = 20.0 * np.log10(harmonic_amps + 1e-10)

    if len(harmonic_db) >= 3:
        smoothed = harmonic_db.copy()
        for i in range(1, len(smoothed) - 1):
            smoothed[i] = (harmonic_db[i - 1] + harmonic_db[i] + harmonic_db[i + 1]) / 3.0
        harmonic_db = smoothed

    # ── 피크 찾기 (prominence 기반) ──
    MIN_PROMINENCE = 1.5  # dB
    peaks = []
    for i in range(1, len(harmonic_db) - 1):
        if harmonic_db[i] > harmonic_db[i - 1] and harmonic_db[i] > harmonic_db[i + 1]:
            prom = min(harmonic_db[i] - harmonic_db[i - 1],
                       harmonic_db[i] - harmonic_db[i + 1])
            if prom >= MIN_PROMINENCE:
                peaks.append((harmonic_freqs[i], harmonic_db[i], prom))

    if not peaks:
        return [0.0] * n_formants

    # ── 포먼트 배정: 주파수 순 ──
    peaks.sort(key=lambda p: p[0])

    formant_ranges = [
        (200, 1000),   # F1
        (600, 2800),   # F2
        (1800, 3500),  # F3
    ]

    formants = [0.0] * n_formants
    assigned = 0
    for freq, db, prom in peaks:
        if assigned >= n_formants:
            break
        f_lo, f_hi = formant_ranges[assigned]
        if f_lo <= freq <= f_hi:
            formants[assigned] = freq
            assigned += 1
        elif freq > f_hi:
            # 현재 포먼트 범위를 넘었으면 다음 포먼트로
            assigned += 1
            if assigned < n_formants:
                f_lo, f_hi = formant_ranges[assigned]
                if f_lo <= freq <= f_hi:
                    formants[assigned] = freq
                    assigned += 1

    return formants[:n_formants]


# ═══════════════════════════════════════════════════
#  LPC 포먼트 추출 (폴백)
# ═══════════════════════════════════════════════════

def _extract_formants_lpc(audio, sr, lpc_order=12, n_formants=3):
    """LPC 기반 포먼트 추출. 다운샘플링된 오디오 필요."""
    n = len(audio)
    if n < lpc_order + 1:
        return [0.0] * n_formants

    emphasized = np.empty(n, dtype=np.float64)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - 0.97 * audio[:-1]
    emphasized *= np.hamming(n)

    full_corr = np.correlate(emphasized, emphasized, mode='full')
    mid = len(full_corr) // 2
    r = full_corr[mid:mid + lpc_order + 1].copy()

    if r[0] < 1e-10:
        return [0.0] * n_formants

    a = np.zeros(lpc_order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, lpc_order + 1):
        lam = 0.0
        for j in range(1, i):
            lam += a[j] * r[i - j]
        lam = -(r[i] + lam) / max(e, 1e-12)
        a_prev = a.copy()
        for j in range(1, i):
            a[j] = a_prev[j] + lam * a_prev[i - j]
        a[i] = lam
        e *= (1.0 - lam * lam)
        if e <= 0:
            break

    poly = a[:lpc_order + 1]
    roots = np.roots(poly)
    roots = roots[np.imag(roots) > 0.01]
    if len(roots) == 0:
        return [0.0] * n_formants

    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2.0 * np.pi))
    bandwidths = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.abs(roots) + 1e-12)

    valid = (bandwidths < 500) & (freqs > 90) & (freqs < sr / 2 - 50)
    freqs = freqs[valid]

    if len(freqs) == 0:
        return [0.0] * n_formants

    freqs = np.sort(freqs)
    result = freqs[:n_formants].tolist()
    while len(result) < n_formants:
        result.append(0.0)
    return result


# ═══════════════════════════════════════════════════
#  Lobanov 정규화
# ═══════════════════════════════════════════════════

class LobanovNormalizer:
    """화자별 F1/F2를 z-score로 변환."""

    def __init__(self):
        self._f1_mean = 0.0
        self._f1_std = 1.0
        self._f2_mean = 0.0
        self._f2_std = 1.0
        self._fitted = False

    def fit(self, all_f1, all_f2):
        if len(all_f1) < 5:
            return
        self._f1_mean = float(np.mean(all_f1))
        self._f1_std = max(float(np.std(all_f1)), 10.0)
        self._f2_mean = float(np.mean(all_f2))
        self._f2_std = max(float(np.std(all_f2)), 10.0)
        self._fitted = True

    def transform(self, f1, f2):
        if not self._fitted:
            return f1, f2
        return ((f1 - self._f1_mean) / self._f1_std,
                (f2 - self._f2_mean) / self._f2_std)

    @property
    def is_fitted(self):
        return self._fitted


# ═══════════════════════════════════════════════════
#  메인 분류기
# ═══════════════════════════════════════════════════

class FormantVowelClassifier:
    """포먼트 기반 실시간 한국어 모음 분류기.

    v3: 하모닉 봉투 우선, LPC 폴백, 오/우 보조 피처, Lobanov 정규화.
    """

    def __init__(self, sample_rate=44100, target_sr=11025,
                 lpc_order=12, smooth_alpha=0.3):
        self._sr = sample_rate
        self._target_sr = target_sr
        self._lpc_order = lpc_order
        self._smooth_alpha = smooth_alpha

        self._decimate_factor = round(sample_rate / target_sr)
        self._actual_target_sr = sample_rate / self._decimate_factor

        self._smooth_f1 = 0.0
        self._smooth_f2 = 0.0
        self._is_active = False

        self._vowel_names = list(DEFAULT_PROTOTYPES.keys())
        self._prototypes_hz = dict(DEFAULT_PROTOTYPES)
        self._prototypes_norm = np.array([
            [f1 / F1_RANGE, f2 / F2_RANGE]
            for f1, f2 in DEFAULT_PROTOTYPES.values()
        ], dtype=np.float64)
        self._calibrated = False
        self._normalizer = LobanovNormalizer()

    @property
    def is_calibrated(self):
        return self._calibrated

    @property
    def prototypes(self):
        return dict(self._prototypes_hz)

    def classify(self, audio_chunk, f0=None, fft_mag=None, fft_freqs=None):
        """오디오 프레임 → (vowel, confidence, f1_hz, f2_hz).

        f0/fft_mag/fft_freqs가 주어지면 하모닉 봉투 사용.
        없으면 LPC 폴백.
        """
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        if rms < MIN_RMS:
            return None, 0.0, 0.0, 0.0

        # ── 포먼트 추출: 하모닉 봉투 우선 ──
        f1_raw, f2_raw, f3_raw = 0.0, 0.0, 0.0
        method = 'none'

        if f0 is not None and f0 > 60 and fft_mag is not None and fft_freqs is not None:
            formants = _extract_formants_harmonic(
                fft_mag, fft_freqs, f0, self._sr, n_formants=3)
            f1_raw, f2_raw = formants[0], formants[1]
            f3_raw = formants[2] if len(formants) > 2 else 0.0
            if f1_raw > 0 and f2_raw > 0:
                method = 'harmonic'

        # LPC 폴백
        if f1_raw <= 0 or f2_raw <= 0:
            if self._decimate_factor > 1:
                downsampled = decimate(audio_chunk.astype(np.float64),
                                       self._decimate_factor, ftype='fir')
            else:
                downsampled = audio_chunk.astype(np.float64)
            formants = _extract_formants_lpc(downsampled, self._actual_target_sr,
                                              self._lpc_order, n_formants=3)
            f1_raw, f2_raw = formants[0], formants[1]
            f3_raw = formants[2] if len(formants) > 2 else 0.0
            if f1_raw > 0 and f2_raw > 0:
                method = 'lpc'

        if f1_raw <= 0 or f2_raw <= 0:
            return None, 0.0, 0.0, 0.0

        if not (130 < f1_raw < 1100 and 500 < f2_raw < 3000):
            return None, 0.0, 0.0, 0.0
        if f1_raw >= f2_raw:
            return None, 0.0, 0.0, 0.0

        # ── EMA 스무딩 ──
        alpha = self._smooth_alpha
        if self._smooth_f1 <= 0:
            self._smooth_f1 = f1_raw
            self._smooth_f2 = f2_raw
        else:
            self._smooth_f1 += alpha * (f1_raw - self._smooth_f1)
            self._smooth_f2 += alpha * (f2_raw - self._smooth_f2)

        f1 = self._smooth_f1
        f2 = self._smooth_f2
        self._is_active = True

        # ── 분류 ──
        if self._normalizer.is_fitted:
            nf1, nf2 = self._normalizer.transform(f1, f2)
            point = np.array([nf1, nf2])
            norm_scale = 1.0  # z-score 공간
        else:
            point = np.array([f1 / F1_RANGE, f2 / F2_RANGE])
            norm_scale = CONFIDENCE_SCALE / max(F1_RANGE, F2_RANGE)

        dists = np.linalg.norm(self._prototypes_norm - point, axis=1)
        best_idx = np.argmin(dists)
        best_dist = dists[best_idx]
        vowel = self._vowel_names[best_idx]

        confidence = 1.0 / (1.0 + (best_dist / max(norm_scale, 0.01)) ** 2)

        # ── 오/우 보조 판단 ──
        if vowel in ('오', '우') and fft_mag is not None and fft_freqs is not None:
            vowel, confidence = self._resolve_ou(
                vowel, confidence, f1, f2, f3_raw,
                fft_mag, fft_freqs, dists)

        return vowel, float(confidence), float(f1), float(f2)

    def _resolve_ou(self, initial_vowel, initial_conf, f1, f2, f3,
                    fft_mag, fft_freqs, dists):
        """오/우 경계에서 F1 + F3 + 스펙트럴 틸트로 보조 판단."""
        o_idx = self._vowel_names.index('오')
        u_idx = self._vowel_names.index('우')

        ou_margin = abs(dists[o_idx] - dists[u_idx])
        total_ou = dists[o_idx] + dists[u_idx]
        margin_ratio = ou_margin / (total_ou + 1e-6)

        if margin_ratio > 0.3:
            return initial_vowel, initial_conf

        # 스펙트럴 틸트: 저주파(0~1kHz) vs 고주파(1~4kHz)
        spectral_tilt = 0.0
        if fft_freqs is not None and len(fft_freqs) > 0:
            low_mask = fft_freqs < 1000
            high_mask = (fft_freqs >= 1000) & (fft_freqs < 4000)
            low_e = np.sum(fft_mag[low_mask] ** 2) if np.any(low_mask) else 1e-10
            high_e = np.sum(fft_mag[high_mask] ** 2) if np.any(high_mask) else 1e-10
            spectral_tilt = 10.0 * np.log10(low_e / max(high_e, 1e-10))

        # F3: 우가 오보다 낮은 경향
        f3_score = 0.0
        if f3 > 0:
            f3_center = 2600.0
            f3_score = (f3_center - f3) / 300.0
            f3_score = max(-1.0, min(1.0, f3_score))

        # F1: 오 > 우 (가장 안정적)
        f1_boundary = 400.0
        if self._calibrated:
            o_f1 = self._prototypes_hz.get('오', (450, 850))[0]
            u_f1 = self._prototypes_hz.get('우', (350, 750))[0]
            f1_boundary = (o_f1 + u_f1) / 2.0
        f1_score = (f1 - f1_boundary) / 100.0
        f1_score = max(-1.0, min(1.0, f1_score))

        # 종합: 양수=오, 음수=우
        ou_score = (f1_score * 0.5
                    + f3_score * 0.2
                    - spectral_tilt * 0.03)

        # 비대칭 threshold: 오→우 전환은 더 확실해야 함
        if ou_score > 0.1:
            return '오', initial_conf
        elif ou_score < -0.25:
            return '우', initial_conf
        else:
            return initial_vowel, initial_conf * 0.7

    def reset(self):
        if self._is_active:
            self._smooth_f1 = 0.0
            self._smooth_f2 = 0.0
            self._is_active = False

    def set_prototypes(self, prototypes_hz):
        for i, name in enumerate(self._vowel_names):
            if name in prototypes_hz:
                f1, f2 = prototypes_hz[name]
                self._prototypes_hz[name] = (f1, f2)
                if self._normalizer.is_fitted:
                    nf1, nf2 = self._normalizer.transform(f1, f2)
                    self._prototypes_norm[i] = [nf1, nf2]
                else:
                    self._prototypes_norm[i] = [f1 / F1_RANGE, f2 / F2_RANGE]
        self._calibrated = True


# ═══════════════════════════════════════════════════
#  화자 캘리브레이션
# ═══════════════════════════════════════════════════

class FormantCalibrator:
    """화자별 캘리브레이션 + Lobanov 정규화."""

    CORNER_INTERP = {
        '어': (0.6, 0.1, 0.3),
        '오': (0.3, 0.0, 0.7),
        '으': (0.1, 0.4, 0.5),
        '에': (0.2, 0.7, 0.1),
    }

    def __init__(self, classifier: FormantVowelClassifier):
        self._clf = classifier
        self._samples = {}

    def record_frame(self, vowel, f1, f2):
        if vowel not in self._samples:
            self._samples[vowel] = []
        if f1 > 0 and f2 > 0:
            self._samples[vowel].append((f1, f2))

    def apply(self):
        measured = {}
        for vowel, frames in self._samples.items():
            if len(frames) >= 10:
                f1s = [f[0] for f in frames]
                f2s = [f[1] for f in frames]
                measured[vowel] = (float(np.median(f1s)),
                                    float(np.median(f2s)))

        if len(measured) == 0:
            print('[캘리브레이션] 데이터 부족')
            return False

        new_prototypes = dict(DEFAULT_PROTOTYPES)
        for v, (f1, f2) in measured.items():
            new_prototypes[v] = (f1, f2)

        corners = {'아', '이', '우'}
        if corners.issubset(measured.keys()):
            a_f1, a_f2 = measured['아']
            i_f1, i_f2 = measured['이']
            u_f1, u_f2 = measured['우']
            for vowel, (w_a, w_i, w_u) in self.CORNER_INTERP.items():
                if vowel not in measured:
                    est_f1 = w_a * a_f1 + w_i * i_f1 + w_u * u_f1
                    est_f2 = w_a * a_f2 + w_i * i_f2 + w_u * u_f2
                    new_prototypes[vowel] = (est_f1, est_f2)

        # Lobanov 정규화 피팅
        all_f1, all_f2 = [], []
        for frames in self._samples.values():
            for f1, f2 in frames:
                if f1 > 0 and f2 > 0:
                    all_f1.append(f1)
                    all_f2.append(f2)
        self._clf._normalizer.fit(all_f1, all_f2)

        # 프로토타입 설정 (set_prototypes 내에서 normalizer 적용)
        self._clf.set_prototypes(new_prototypes)

        print(f'[캘리브레이션 완료] 측정 {len(measured)}개, '
              f'Lobanov: {"ON" if self._clf._normalizer.is_fitted else "OFF"}')
        for v in self._clf._vowel_names:
            f1, f2 = new_prototypes.get(v, DEFAULT_PROTOTYPES[v])
            src = '측정' if v in measured else '보간'
            print(f'  {v}: F1={f1:.0f} F2={f2:.0f} ({src})')
        return True

    def get_stats(self):
        return {v: len(frames) for v, frames in self._samples.items()}

    def reset(self):
        self._samples = {}
