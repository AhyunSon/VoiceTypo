"""
formant_engine.py — 포먼트 추출 엔진 (시각화용)

역할 분담:
  pyworld  → F0 + 유성음/숨소리 판단 (게이트)
  Praat LPC→ F1/F2/F3 포먼트 추출 (시각화용)
  Kalman   → 포먼트 안정화

vowel 분류는 wav2vec_classifier.py(wav2vec2)가 담당.
이 엔진은 포먼트 수치와 시각화 데이터만 제공한다.
"""

import numpy as np
import pyworld as pw
import parselmouth
from parselmouth.praat import call

from config import (
    SAMPLE_RATE, PREEMPH_ALPHA,
    FORMANT_CEILINGS, MAX_BW, SAMPLE_POS, ANALYSIS_WIN_SEC,
    KALMAN_PROCESS_NOISE, KALMAN_MEAS_NOISE_DEF,
    HNR_MIN_DB, PARAMS,
    PYWORLD_VOICED_FRAC_MIN, PYWORLD_FRAME_PERIOD,
)


# ══════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════

def preemphasis(signal: np.ndarray, alpha: float = PREEMPH_ALPHA) -> np.ndarray:
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def _weighted_median(values: list, weights: list) -> float:
    arr = np.array(values, dtype=float)
    w   = np.array(weights, dtype=float)
    idx = np.argsort(arr)
    arr, w = arr[idx], w[idx]
    cumw = np.cumsum(w)
    pos  = np.searchsorted(cumw, cumw[-1] / 2.0)
    return float(arr[min(pos, len(arr) - 1)])


# ══════════════════════════════════════════
# 칼만 필터
# ══════════════════════════════════════════

class KalmanFormant:
    def __init__(self,
                 process_noise: float = KALMAN_PROCESS_NOISE,
                 meas_noise:    float = KALMAN_MEAS_NOISE_DEF):
        self.x = None
        self.P = 10000.0
        self.Q = process_noise ** 2
        self.R = meas_noise ** 2

    def update(self, measurement, bandwidth=None):
        P_pred = self.P + self.Q
        if measurement is None:
            if self.x is not None:
                self.P = P_pred
            return self.x
        if self.x is None:
            self.x = float(measurement)
            self.P = self.R
            return self.x
        R = (max((bandwidth * 0.25) ** 2, 35.0 ** 2)
             if bandwidth is not None and bandwidth > 0 else self.R)
        K      = P_pred / (P_pred + R)
        self.x = self.x + K * (float(measurement) - self.x)
        self.P = (1.0 - K) * P_pred
        return self.x

    def reset(self):
        self.x = None
        self.P = 10000.0


# ══════════════════════════════════════════
# 포먼트 추출 엔진
# ══════════════════════════════════════════

class FormantEngine:

    VOWEL_CHANGE_THRESH_F1 = 120   # Hz/frame — Kalman 리셋 기준
    VOWEL_CHANGE_THRESH_F2 = 200

    def __init__(self):
        self.kf = {
            1: KalmanFormant(process_noise=60,  meas_noise=150),
            2: KalmanFormant(process_noise=100, meas_noise=250),
            3: KalmanFormant(process_noise=120, meas_noise=300),
        }
        self._prev_raw_f1 = None
        self._prev_raw_f2 = None

    def reset_kalman(self):
        for kf in self.kf.values():
            kf.reset()
        self._prev_raw_f1 = None
        self._prev_raw_f2 = None

    # ── pyworld: F0 + 유성음 판단 ──────────────────────────────

    def _pyworld_voiced(self, chunk: np.ndarray, gender: str):
        """
        저주파 aperiodicity 기반 HNR 근사 → 안정적 숨소리 필터링.
        Returns: f0, hnr, is_voiced

        DIO F0 탐색 범위: 항상 50-500Hz (성별 무관).
        이유: 초기 gender="female"일 때 pitch_floor=150을 쓰면
              초저음 남성(F0 65-100Hz) 목소리를 아예 감지 못해
              gender가 영원히 "female"에 고정되는 순환 오류가 생긴다.
              F0값 자체로 성별을 판단하므로 탐색 범위는 넓게 유지한다.
        """
        p = PARAMS[gender]
        x = chunk.astype(np.float64)
        try:
            f0_arr, t_arr = pw.dio(
                x, float(SAMPLE_RATE),
                f0_floor=50.0,    # 고정 하한: 초저음 남성 베이스 포착
                f0_ceil=500.0,    # 고정 상한: 전 성별 커버
                frame_period=PYWORLD_FRAME_PERIOD,
            )
            f0_arr = pw.stonemask(x, f0_arr, t_arr, float(SAMPLE_RATE))

            voiced_mask = f0_arr > 0
            voiced_frac = float(voiced_mask.mean()) if len(f0_arr) > 0 else 0.0
            f0 = float(np.mean(f0_arr[voiced_mask])) if voiced_mask.any() else None
            is_voiced = voiced_frac >= PYWORLD_VOICED_FRAC_MIN
            hnr = None

            if is_voiced:
                ap = pw.d4c(x, f0_arr, t_arr, float(SAMPLE_RATE))
                n_bins   = ap.shape[1]
                low_bins = max(1, int(n_bins * 2000.0 / (SAMPLE_RATE / 2.0)))
                ap_low   = ap[voiced_mask, :low_bins]
                mean_ap  = float(np.mean(ap_low))
                hnr = float(-10.0 * np.log10(max(mean_ap, 1e-6)))

            return f0, hnr, is_voiced

        except Exception:
            return None, None, False

    # ── Praat Burg LPC 포먼트 추출 ─────────────────────────────

    def _praat_formants(self, snd, gender: str):
        """멀티-ceiling Praat Burg → BW 가중 중앙값"""
        p = PARAMS[gender]
        best_f, best_bw, best_score = None, None, float('inf')

        for ceiling in FORMANT_CEILINGS:
            try:
                fmt = call(snd, "To Formant (burg)",
                           0.0, p["max_formants"], ceiling,
                           p["window_length"], p["pre_emphasis"])

                f_vals  = {1: [], 2: [], 3: []}
                bw_vals = {1: [], 2: [], 3: []}

                for pos in SAMPLE_POS:
                    t = ANALYSIS_WIN_SEC * pos
                    for fn in [1, 2, 3]:
                        fv = call(fmt, "Get value at time", fn, t, "Hertz", "Linear")
                        bw = call(fmt, "Get bandwidth at time", fn, t, "Hertz", "Linear")
                        if (not np.isnan(fv) and fv > 0
                                and not np.isnan(bw) and 0 < bw < MAX_BW[fn]):
                            f_vals[fn].append(float(fv))
                            bw_vals[fn].append(float(bw))

                for fn, rk in [(1, "f1_range"), (2, "f2_range"), (3, "f3_range")]:
                    lo, hi = p[rk]
                    pairs = [(f, b) for f, b in zip(f_vals[fn], bw_vals[fn])
                             if lo <= f <= hi]
                    if pairs:
                        f_vals[fn], bw_vals[fn] = [list(x) for x in zip(*pairs)]
                    else:
                        f_vals[fn], bw_vals[fn] = [], []

                bw12 = [float(np.mean(bw_vals[fn]))
                        for fn in [1, 2] if bw_vals[fn]]
                if bw12 and f_vals[1] and f_vals[2]:
                    score = float(np.mean(bw12))
                    if score < best_score:
                        best_score = score
                        best_f, best_bw = f_vals, bw_vals
            except Exception:
                continue

        if best_f is None:
            return {1: None, 2: None, 3: None}, {1: None, 2: None, 3: None}

        raw, bw_avg = {}, {}
        for fn in [1, 2, 3]:
            vals = best_f.get(fn, [])
            bws  = best_bw.get(fn, [])
            if vals:
                w = [1.0 / (b + 1e-6) for b in bws]
                raw[fn]    = _weighted_median(vals, w)
                bw_avg[fn] = float(np.mean(bws))
            else:
                raw[fn] = bw_avg[fn] = None
        return raw, bw_avg

    # ── 메인 추출 ──────────────────────────────────────────────

    def extract(self, chunk: np.ndarray, gender: str = "female") -> dict:
        """
        Returns:
          f0, hnr       — pyworld 기반
          f1, f2, f3    — Kalman 안정화 포먼트 (시각화용)
          raw_f1~3      — Praat 원시값
          confidence    — BW 기반 신뢰도
          is_voiced     — pyworld 유성음 판단
        """
        # ── 1. pyworld 유성음 판단 ─────────────────────────────
        f0, hnr, is_voiced = self._pyworld_voiced(chunk, gender)

        if not is_voiced:
            return dict(f0=f0, hnr=hnr,
                        f1=None, f2=None, f3=None,
                        raw_f1=None, raw_f2=None, raw_f3=None,
                        confidence=0.0, is_voiced=False)

        # ── 2. DC제거 + 프리엠퍼시스 ──────────────────────────
        chunk_pe = preemphasis(chunk - np.mean(chunk))
        snd = parselmouth.Sound(
            chunk_pe.astype(np.float64),
            sampling_frequency=float(SAMPLE_RATE),
        )

        # ── 3. Praat Burg 포먼트 추출 ─────────────────────────
        raw, bw_avg = self._praat_formants(snd, gender)

        # ── 4. HNR 저품질 시 BW 증폭 → Kalman 신뢰도 낮춤 ────
        if hnr is not None and hnr < HNR_MIN_DB:
            for fn in [1, 2, 3]:
                if bw_avg[fn] is not None:
                    bw_avg[fn] = bw_avg[fn] * 2.0

        # ── 5. 모음 전환 감지 → Kalman 리셋 ───────────────────
        if raw[1] is not None and raw[2] is not None:
            f1j = abs(raw[1] - self._prev_raw_f1) if self._prev_raw_f1 else 0
            f2j = abs(raw[2] - self._prev_raw_f2) if self._prev_raw_f2 else 0
            if f2j > self.VOWEL_CHANGE_THRESH_F2 or f1j > self.VOWEL_CHANGE_THRESH_F1:
                for kf in self.kf.values():
                    kf.reset()
            self._prev_raw_f1 = raw[1]
            self._prev_raw_f2 = raw[2]

        # ── 6. Kalman 스무딩 ──────────────────────────────────
        kf_f1 = self.kf[1].update(raw[1], bw_avg[1])
        kf_f2 = self.kf[2].update(raw[2], bw_avg[2])
        kf_f3 = self.kf[3].update(raw[3], bw_avg[3])

        # ── 7. 신뢰도 ─────────────────────────────────────────
        confidence = 0.0
        if raw[1] is not None and bw_avg[1] is not None:
            confidence = max(0.0, min(1.0, (450 - bw_avg[1]) / 200))
            if hnr is not None and hnr < HNR_MIN_DB:
                confidence *= 0.5

        return dict(f0=f0, hnr=hnr,
                    f1=kf_f1, f2=kf_f2, f3=kf_f3,
                    raw_f1=raw[1], raw_f2=raw[2], raw_f3=raw[3],
                    confidence=confidence, is_voiced=True)
