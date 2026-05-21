"""
formant_engine.py — 포먼트 추출 엔진 (단순화 버전)

역할:
  pyworld     → F0 + 유성음 판정 + jitter
  Praat Burg  → F1/F2/F3 (단일 source)

정리 이력 (2026-04-28, Stage 2):
  - cheaptrick + scipy LPC 앙상블 제거
    (method_comparison.md: ensemble 평균 |F2 Δ| = 397 vs Praat 단독 131)
  - Kalman 필터 제거 (단일 청크 평가에서 의미 없음, EMA 가 ui_window 에서 대체)
  - preemphasis 함수 제거 (Praat 자체 pre_emphasis_from=50 으로 충분)
"""

import numpy as np
import pyworld as pw
import parselmouth
from parselmouth.praat import call

from config import (
    SAMPLE_RATE,
    FORMANT_CEILINGS, MAX_BW, SAMPLE_POS, ANALYSIS_WIN_SEC,
    HNR_MIN_DB, PARAMS,
    PYWORLD_VOICED_FRAC_MIN, PYWORLD_FRAME_PERIOD,
)


# ══════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════

def compute_jitter(f0_arr: np.ndarray) -> float:
    """
    Local absolute jitter (%) — 시각화용 떨림 지표.

    Jitter = mean(|T0[i] - T0[i-1]|) / mean(T0) × 100,  T0 = 1/F0
    유효 voiced 프레임 ≥4 필요. 미달 시 0.0.

    범위 가이드:
      <0.5%  안정
      0.5–2  자연 vibrato
      2–6    뚜렷한 떨림
      >6     강한 tremolo
    """
    voiced = f0_arr[f0_arr > 30.0]
    if len(voiced) < 4:
        return 0.0
    periods = 1.0 / voiced
    diffs   = np.abs(np.diff(periods))
    return float(np.mean(diffs) / np.mean(periods) * 100.0)


def _weighted_median(values: list, weights: list) -> float:
    arr = np.array(values, dtype=float)
    w   = np.array(weights, dtype=float)
    idx = np.argsort(arr)
    arr, w = arr[idx], w[idx]
    cumw = np.cumsum(w)
    pos  = np.searchsorted(cumw, cumw[-1] / 2.0)
    return float(arr[min(pos, len(arr) - 1)])


# ══════════════════════════════════════════
# 포먼트 추출 엔진
# ══════════════════════════════════════════

class FormantEngine:
    """
    Praat Burg 단독으로 F1/F2/F3 추출.
    pyworld 는 F0/유성음/jitter 만 제공.
    """

    def __init__(self):
        # Stage 2: Kalman 제거 — 인스턴스 상태 없음.
        pass

    def reset_kalman(self):
        """ui_window 호환용 no-op (Kalman 제거됨)."""
        pass

    # ── pyworld: F0 + 유성음 판단 ──────────────────────────────

    def _pyworld_voiced(self, chunk: np.ndarray, gender: str):
        """
        F0 추출 + HNR 근사 + 유성음 판정.
        Returns: f0, hnr, is_voiced, f0_arr, t_arr

        DIO F0 탐색: 50-500 Hz 고정 (성별 무관).
        이유: gender 기본값 'female' 일 때 pitch_floor=150 이면
              초저음 남성(F0 65-100Hz) 감지 못해 gender 가
              영원히 female 에 고정되는 순환 오류 발생.
        """
        x = chunk.astype(np.float64)
        try:
            f0_arr, t_arr = pw.dio(
                x, float(SAMPLE_RATE),
                f0_floor=50.0,
                f0_ceil=500.0,
                frame_period=PYWORLD_FRAME_PERIOD,
            )
            f0_arr = pw.stonemask(x, f0_arr, t_arr, float(SAMPLE_RATE))

            voiced_mask = f0_arr > 0
            voiced_frac = (float(voiced_mask.mean())
                           if len(f0_arr) > 0 else 0.0)
            f0 = (float(np.mean(f0_arr[voiced_mask]))
                  if voiced_mask.any() else None)
            is_voiced = voiced_frac >= PYWORLD_VOICED_FRAC_MIN
            hnr = None

            if is_voiced:
                ap = pw.d4c(x, f0_arr, t_arr, float(SAMPLE_RATE))
                n_bins   = ap.shape[1]
                low_bins = max(1, int(n_bins * 2000.0 / (SAMPLE_RATE / 2.0)))
                ap_low   = ap[voiced_mask, :low_bins]
                mean_ap  = float(np.mean(ap_low))
                hnr = float(-10.0 * np.log10(max(mean_ap, 1e-6)))

            return f0, hnr, is_voiced, f0_arr, t_arr

        except Exception:
            return None, None, False, np.zeros(1), np.zeros(1)

    # ── Praat Burg LPC 포먼트 추출 ─────────────────────────────

    def _praat_formants(self, snd, gender: str, ceilings: list = None):
        """
        ceilings 의 각 값으로 시도 → BW 가중 중앙값.
        단일 ceiling 이면 한 번만 실행.
        """
        p = PARAMS[gender]
        best_f, best_bw, best_score = None, None, float('inf')
        ceilings = ceilings or FORMANT_CEILINGS

        for ceiling in ceilings:
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

    def extract(self, chunk: np.ndarray, gender: str = "female",
                ceilings: list = None, force_extract: bool = False) -> dict:
        """
        Returns dict:
          f0, hnr, jitter         — pyworld 기반 (시각화용)
          f1, f2, f3              — Praat raw 값
          raw_f1, raw_f2, raw_f3  — f1/f2/f3 와 동일 (ui_window 호환 키)
          confidence              — BW 기반 신뢰도
          is_voiced               — pyworld 유성음 판단

        force_extract=True: pyworld voiced 판정 무시하고 Praat 강제 실행.
          breathy 음성 (whisper, 우 등) 의 포먼트 추출 가능.
          Praat 의 bandwidth 필터가 quality 보장.
        """
        # 1. pyworld 유성음 판단
        f0, hnr, is_voiced, f0_arr, _ = self._pyworld_voiced(chunk, gender)

        if not is_voiced and not force_extract:
            return dict(f0=f0, hnr=hnr,
                        f1=None, f2=None, f3=None,
                        raw_f1=None, raw_f2=None, raw_f3=None,
                        confidence=0.0, jitter=0.0,
                        is_voiced=False)

        # 2. Praat Burg 포먼트 추출 (DC 제거만, Praat 자체 preemphasis 적용)
        chunk_dc = (chunk - np.mean(chunk)).astype(np.float64)
        snd = parselmouth.Sound(
            chunk_dc,
            sampling_frequency=float(SAMPLE_RATE),
        )
        raw, bw_avg = self._praat_formants(snd, gender, ceilings)

        # 3. 신뢰도 (F1 BW 기반; HNR 저품질이면 절반)
        confidence = 0.0
        if raw[1] is not None and bw_avg[1] is not None:
            confidence = max(0.0, min(1.0, (450 - bw_avg[1]) / 200))
            if hnr is not None and hnr < HNR_MIN_DB:
                confidence *= 0.5

        jitter = compute_jitter(f0_arr)

        # force_extract 시에도 실제 voicing 상태는 보존
        return dict(f0=f0, hnr=hnr,
                    f1=raw[1], f2=raw[2], f3=raw[3],
                    raw_f1=raw[1], raw_f2=raw[2], raw_f3=raw[3],
                    confidence=confidence,
                    jitter=jitter, is_voiced=is_voiced)
