"""
vad.py — 적응형 VAD (Voice Activity Detection)

StepByStep 대비 개선:
  - 초기 캘리브레이션 + 침묵 중 노이즈 바닥 지속 적응
  - 3-조건 VAD: RMS + 자기상관 주기성 + ZCR (step06-1)
  - 적응형 업데이트: 침묵이 지속되면 노이즈 추정값 자동 조정
"""

import numpy as np
from config import (
    SAMPLE_RATE, VAD_RMS_MULT,
    AUTOCORR_THRESH, ZCR_THRESH, ADAPT_RATE,
)


class AdaptiveVAD:
    """
    적응형 3-조건 VAD
    - 캘리브레이션으로 초기 임계값 설정
    - 침묵 프레임마다 지수 이동 평균으로 노이즈 바닥 갱신
    - onset/hangover: 연속 2프레임 확인 후 음성 시작, 5프레임 유지 후 침묵 전환
    """

    ONSET_FRAMES    = 2   # 음성 시작 확인에 필요한 연속 voiced 프레임 수
    HANGOVER_FRAMES = 3   # Fix3: 음성 종료 후 유지 프레임 (5→3, ~90ms)

    def __init__(self, initial_noise_rms: float = 0.001):
        self.noise_rms      = initial_noise_rms
        self.threshold      = initial_noise_rms * VAD_RMS_MULT
        self._onset_cnt     = 0   # 연속 voiced 카운터
        self._hangover_cnt  = 0   # 음성 종료 후 유지 카운터
        self._is_voiced     = False

    # ── 캘리브레이션 ────────────────────────────────────────────
    def calibrate(self, rms_list: list):
        """수집된 RMS 목록으로 노이즈 바닥 및 임계값 설정"""
        if rms_list:
            self.noise_rms = float(np.percentile(rms_list, 80))  # 상위 20% 제거
            self.threshold = self.noise_rms * VAD_RMS_MULT
        # 상태 리셋
        self._onset_cnt    = 0
        self._hangover_cnt = 0
        self._is_voiced    = False

    # ── 침묵 중 적응 ────────────────────────────────────────────
    def _adapt(self, rms: float):
        """침묵 프레임에서 노이즈 바닥을 천천히 추적"""
        self.noise_rms = (1 - ADAPT_RATE) * self.noise_rms + ADAPT_RATE * rms
        self.threshold = self.noise_rms * VAD_RMS_MULT

    # ── 메인 VAD 판단 ────────────────────────────────────────────
    def check(self, chunk: np.ndarray,
              pitch_lo: float = 75.0,
              pitch_hi: float = 400.0,
              sr: int = SAMPLE_RATE) -> tuple:
        """
        3조건 VAD 검사.

        Returns:
            is_voice (bool): 유성음 판단
            rms (float): 현재 프레임 RMS
        """
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # ── 1) RMS 에너지 조건 ──
        if rms < self.threshold:
            self._adapt(rms)
            return self._update_state(False), rms

        # ── 2) 자기상관 주기성 (유성음 확인) ──
        lag_min = max(1, int(sr / pitch_hi))
        lag_max = int(sr / pitch_lo)
        if lag_max >= len(chunk):
            return self._update_state(False), rms

        # FFT 기반 자기상관: O(n log n), 직접 계산 O(n²)보다 ~10x 빠름
        n = len(chunk)
        fft_size = 1 << (2 * n - 1).bit_length()   # 다음 2의 거듭제곱
        X = np.fft.rfft(chunk, n=fft_size)
        r = np.fft.irfft(X * np.conj(X))[:n]
        r0 = r[0]
        if r0 <= 1e-12:
            return False, rms

        ratio = float(np.max(r[lag_min:lag_max])) / r0
        if ratio < AUTOCORR_THRESH:
            self._adapt(rms)
            return self._update_state(False), rms

        # ── 3) ZCR (유성음은 영점 교차율이 낮음) ──
        zcr = float(np.sum(np.abs(np.diff(np.sign(chunk)))) / 2) / len(chunk)
        if zcr > ZCR_THRESH:
            self._adapt(rms)
            return self._update_state(False), rms

        return self._update_state(True), rms

    # ── onset / hangover 상태 머신 ────────────────────────────────
    def _update_state(self, raw_voice: bool) -> bool:
        """
        순간 판단(raw_voice)을 onset/hangover 로직으로 안정화.

        - 음성 시작: ONSET_FRAMES 연속 voiced → 음성 ON
        - 음성 종료: raw_voice=False 후 HANGOVER_FRAMES 유지 → 음성 OFF
        """
        if raw_voice:
            self._onset_cnt += 1
            self._hangover_cnt = self.HANGOVER_FRAMES  # 행오버 리셋
            if self._onset_cnt >= self.ONSET_FRAMES:
                self._is_voiced = True
        else:
            self._onset_cnt = 0
            if self._hangover_cnt > 0:
                self._hangover_cnt -= 1
                # 행오버 기간 중에는 이전 상태 유지
            else:
                self._is_voiced = False
        return self._is_voiced
