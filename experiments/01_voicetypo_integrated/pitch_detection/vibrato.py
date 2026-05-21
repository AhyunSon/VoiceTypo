"""비브라토 분석 모듈.

HTML 원본의 analyzeVibrato()를 포팅.
피치 히스토리에서 진동 패턴을 추출하여
rate(진동 빈도)와 extent(진폭)를 계산.

사용법:
    analyzer = VibratoAnalyzer()
    analyzer.push(freq, rms)  # 매 프레임 호출
    rate, extent = analyzer.get()
"""

import numpy as np
from collections import deque

PITCH_HISTORY_SIZE = 30
MIN_VIBRATO_RATE = 1.5    # Hz — HTML 원본 기준
MAX_VIBRATO_RATE = 10.0   # Hz
VIBRATO_SMOOTHING = 0.15  # 스무딩 계수
VIBRATO_DECAY = 0.98      # 음성 없을 때 감쇠


class VibratoAnalyzer:
    def __init__(self, frames_per_sec: float = 21.5):
        """frames_per_sec: 초당 프레임 수 (44100/2048 ≈ 21.5)"""
        self.fps = frames_per_sec
        self._history = deque(maxlen=PITCH_HISTORY_SIZE)
        self._rate = 0.0
        self._extent = 0.0
        self._no_voice_count = 0

    def push(self, freq: float, rms: float = 0.1):
        """유효한 피치가 감지될 때마다 호출."""
        if freq <= 0:
            self._no_voice_count += 1
            if self._no_voice_count > 3:
                # 음성 없으면 감쇠
                self._rate *= VIBRATO_DECAY
                self._extent *= VIBRATO_DECAY
                if self._rate < 0.1:
                    self._rate = 0.0
                if self._extent < 0.01:
                    self._extent = 0.0
            return
        self._no_voice_count = 0
        self._history.append((freq, rms))
        if len(self._history) >= 8:
            self._analyze()

    def get(self) -> tuple:
        """현재 비브라토 상태.
        Returns: (rate, extent)
            rate: 진동 빈도 (Hz). 0이면 비브라토 아님.
            extent: 진폭 (반음 단위).
        """
        return self._rate, self._extent

    def reset(self):
        self._history.clear()
        self._rate = 0.0
        self._extent = 0.0
        self._no_voice_count = 0

    def _analyze(self):
        data = list(self._history)
        pitches = np.array([d[0] for d in data], dtype=np.float64)
        rms_val = np.mean([d[1] for d in data])
        n = len(pitches)

        # 평균 피치
        mean_pitch = np.mean(pitches)
        if mean_pitch < 1.0:
            self._rate *= VIBRATO_DECAY
            self._extent *= VIBRATO_DECAY
            return

        # 평균 대비 편차 (반음 단위)
        deviations = 12.0 * np.log2(pitches / mean_pitch)

        # 진폭: peak-to-peak / 2
        extent = (np.max(deviations) - np.min(deviations)) / 2.0

        # 영점 교차 횟수 → 진동 주파수
        zero_crossings = 0
        for i in range(1, n):
            if deviations[i - 1] * deviations[i] < 0:
                zero_crossings += 1

        # rate = (교차 횟수 / 2) / 구간 시간
        duration = (n - 1) / self.fps
        if duration > 0:
            rate = (zero_crossings / 2.0) / duration
        else:
            rate = 0.0

        # 동적 진폭 기준: 소리 작으면 기준 높임 (노이즈 오감지 방지)
        # rms 0.1+ → 50cents(0.5반음), rms 0.02 → 82cents(0.82반음)
        extent_threshold = (50 + max(0, (1 - rms_val * 10)) * 40) / 100.0

        # 유효성 검증
        if rate >= MIN_VIBRATO_RATE and rate <= MAX_VIBRATO_RATE and extent >= extent_threshold:
            # 스무딩 적용
            self._rate += VIBRATO_SMOOTHING * (rate - self._rate)
            self._extent += VIBRATO_SMOOTHING * (extent - self._extent)
        else:
            # 감쇠
            self._rate *= VIBRATO_DECAY
            self._extent *= VIBRATO_DECAY
            if self._rate < 0.1:
                self._rate = 0.0
            if self._extent < 0.01:
                self._extent = 0.0
