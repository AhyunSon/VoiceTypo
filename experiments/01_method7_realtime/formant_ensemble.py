"""
formant_ensemble.py — 3-방법 앙상블 포먼트 추출기

Method 1: Praat Burg LPC (멀티-ceiling, BW 가중 중앙값)
Method 2: pyworld CheapTrick 스펙트럼 피크-피킹 (source-filter 가정 없음)
Method 3: scipy LPC 근 (Levinson-Durbin + 다항식 근)

앙상블:
  - 2개 이상 일치(±AGREE_TOL Hz): 동의값 중앙값 채택
  - 불일치: 세 값의 중앙값 + 신뢰도 낮음
  - 유효값 1개: 그 값 + 신뢰도 최저
  - 유효값 없음: None
"""

import numpy as np
import pyworld as pw
import parselmouth
from parselmouth.praat import call
from scipy.signal import find_peaks

from config import (
    SAMPLE_RATE, PREEMPH_ALPHA,
    FORMANT_CEILINGS, MAX_BW, SAMPLE_POS, ANALYSIS_WIN_SEC,
    PARAMS,
)

# 두 추정값이 "일치"로 간주되는 허용 오차 (Hz)
AGREE_TOL = {1: 80, 2: 120, 3: 160}   # F1 < F2 < F3 순으로 허용 폭 확대


# ══════════════════════════════════════════
# 공통 유틸
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
# Method 1: Praat Burg LPC
# ══════════════════════════════════════════

def praat_burg_formants(chunk_pe: np.ndarray, gender: str):
    """
    멀티-ceiling Praat Burg → BW 가중 중앙값
    Returns: (f1, f2, f3) or (None, None, None)
    """
    p = PARAMS[gender]
    snd = parselmouth.Sound(
        chunk_pe.astype(np.float64),
        sampling_frequency=float(SAMPLE_RATE),
    )

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

            # 유효 범위 필터 (pair-wise)
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
        return None, None, None

    out = []
    for fn in [1, 2, 3]:
        vals = best_f.get(fn, [])
        bws  = best_bw.get(fn, [])
        if vals:
            w = [1.0 / (b + 1e-6) for b in bws]
            out.append(_weighted_median(vals, w))
        else:
            out.append(None)
    return tuple(out)


# ══════════════════════════════════════════
# Method 2: pyworld CheapTrick 스펙트럼 피크
# ══════════════════════════════════════════

def cheaptrick_formants(x_f64: np.ndarray,
                        f0_arr: np.ndarray,
                        t_arr: np.ndarray,
                        gender: str):
    """
    CheapTrick 스펙트럼 포락선에서 피크 = 포먼트
    source-filter 분리 없이 직접 스펙트럼 피크 탐색
    Returns: (f1, f2, f3) or (None, None, None)
    """
    try:
        sp = pw.cheaptrick(x_f64, f0_arr, t_arr, float(SAMPLE_RATE))
        if sp.shape[0] == 0:
            return None, None, None

        # 중간 프레임 사용 (안정적인 steady-state 구간)
        mid = sp.shape[0] // 2
        envelope = sp[mid]                        # (n_fft//2+1,)

        n_half = len(envelope)
        freqs   = np.fft.rfftfreq(2 * (n_half - 1), 1.0 / SAMPLE_RATE)

        # 로그 스펙트럼 포락선 → 피크 탐색
        log_env = np.log(np.maximum(envelope, 1e-10))

        # 최소 피크 간격: 약 100 Hz
        bin_per_hz = n_half / (SAMPLE_RATE / 2.0)
        min_dist   = max(1, int(80 * bin_per_hz))

        peaks, props = find_peaks(
            log_env,
            distance=min_dist,
            prominence=0.3,
        )
        if len(peaks) == 0:
            return None, None, None

        peak_freqs = freqs[peaks]
        prominences = props["prominences"]

        p   = PARAMS[gender]
        out = [None, None, None]
        prev_hi = 0.0   # 순차 선택: F2 > F1, F3 > F2
        for i, (fn, rk) in enumerate([(1, "f1_range"), (2, "f2_range"), (3, "f3_range")]):
            lo, hi = p[rk]
            lo = max(lo, prev_hi)
            mask = (peak_freqs >= lo) & (peak_freqs <= hi)
            if mask.any():
                best_i = int(np.argmax(prominences[mask]))
                val    = float(peak_freqs[mask][best_i])
                out[i] = val
                prev_hi = val + 80.0   # 다음 포먼트는 이 이상
        return tuple(out)

    except Exception:
        return None, None, None


# ══════════════════════════════════════════
# Method 3: scipy LPC 근
# ══════════════════════════════════════════

def _levinson_durbin(r: np.ndarray, order: int):
    """Levinson-Durbin 재귀로 LPC 계수 계산 (표준 구현)"""
    a = np.zeros(order + 1)
    a[0] = 1.0
    E = float(r[0])
    if E < 1e-12:
        return a, E
    for k in range(1, order + 1):
        # 반사 계수
        num = -float(r[k])
        for j in range(1, k):
            num -= a[j] * float(r[k - j])
        km = num / E
        # 계수 갱신
        a_new = a.copy()
        for j in range(1, k):
            a_new[j] = a[j] + km * a[k - j]
        a_new[k] = km
        a = a_new
        E = E * (1.0 - km * km)
        if E <= 0:
            break
    return a, E


def scipy_lpc_formants(chunk_pe: np.ndarray, gender: str):
    """
    Levinson-Durbin LPC → 다항식 근 → 포먼트 주파수
    전통적 수학적 방법, Praat Burg와 상호 보완.

    최적화:
      - LPC 차수: 16 (포먼트 분석 충분, 속도 빠름)
      - FFT 자기상관: O(N log N) vs O(N^2)
    """
    try:
        p   = PARAMS[gender]
        sr  = float(SAMPLE_RATE)
        order = 16   # 포먼트 분석에 충분한 차수 (고정)

        sig = chunk_pe.astype(np.float64)
        sig = sig / (np.max(np.abs(sig)) + 1e-8)
        n   = len(sig)

        # FFT 기반 자기상관 (O(N log N))
        fft_sz = 1 << (2 * n - 1).bit_length()
        X  = np.fft.rfft(sig, n=fft_sz)
        r_full = np.fft.irfft(X * np.conj(X))
        r  = r_full[:order + 1].real

        if r[0] < 1e-12:
            return None, None, None

        # Levinson-Durbin
        a, _ = _levinson_durbin(r, order)

        # 다항식 근 (16차 → 빠름)
        roots = np.roots(a)

        # 단위원 내부 + 양의 허수부만 선택
        roots = roots[np.abs(roots) < 1.0]
        roots = roots[np.imag(roots) >= 0]

        if len(roots) == 0:
            return None, None, None

        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs  = angles * sr / (2.0 * np.pi)
        bws    = -np.log(np.abs(roots)) * sr / np.pi

        # BW 필터 (넓은 대역폭 = 약한 공명 제거)
        valid_mask = bws < 550.0
        freqs = freqs[valid_mask]

        if len(freqs) == 0:
            return None, None, None

        freqs_sorted = np.sort(freqs)

        # 순차 선택: F2 > F1, F3 > F2
        out = [None, None, None]
        prev_hi = 0.0
        for i, (fn, rk) in enumerate([(1, "f1_range"), (2, "f2_range"), (3, "f3_range")]):
            lo, hi = p[rk]
            lo = max(lo, prev_hi)
            cands = freqs_sorted[(freqs_sorted >= lo) & (freqs_sorted <= hi)]
            if len(cands) > 0:
                out[i] = float(cands[0])
                prev_hi = cands[0] + 50.0   # 다음 포먼트는 이 값보다 높아야 함
        return tuple(out)

    except Exception:
        return None, None, None


# ══════════════════════════════════════════
# 앙상블 합산
# ══════════════════════════════════════════

def _ensemble_one(candidates: list, fn: int) -> tuple:
    """
    단일 포먼트(F1/F2/F3)에 대해 3개 후보를 앙상블.
    Returns: (value_or_None, agreement_score 0~1)
    """
    tol   = AGREE_TOL[fn]
    valid = [f for f in candidates if f is not None and f > 50]

    if not valid:
        return None, 0.0
    if len(valid) == 1:
        return valid[0], 0.25   # 단일 방법 = 낮은 신뢰

    # 2-이상 동의 검사
    agreed_vals = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if abs(valid[i] - valid[j]) <= tol:
                agreed_vals.extend([valid[i], valid[j]])

    if agreed_vals:
        # 동의 값들의 중앙값 채택
        value = float(np.median(agreed_vals))
        score = len(agreed_vals) / (len(valid) * 2.0)   # 0.5~1.0
    else:
        # 불일치: 세 값의 중앙값 (Kalman이 흡수)
        value = float(np.median(valid))
        score = 0.1

    return value, score


def ensemble_formants(f_praat:  tuple,
                      f_world:  tuple,
                      f_scipy:  tuple) -> dict:
    """
    세 방법의 (f1, f2, f3) 결과를 앙상블.

    Returns:
        dict with keys f1, f2, f3, agreement (float 0–1)
    """
    result = {}
    scores = []
    for i, fn in enumerate([1, 2, 3]):
        candidates = [f_praat[i], f_world[i], f_scipy[i]]
        val, score = _ensemble_one(candidates, fn)
        result[fn]  = val
        scores.append(score)

    result["agreement"] = float(np.mean(scores))
    return result
