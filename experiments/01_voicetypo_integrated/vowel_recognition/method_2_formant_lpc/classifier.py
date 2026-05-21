"""포먼트(F1/F2) 기반 모음 분류기.

한국어 8개 단모음의 알려진 F1/F2 위치를 사용.
캘리브레이션 불필요 — 즉시 동작.
"""

import math
import numpy as np
from typing import Tuple, Optional
from collections import deque

from .formant import extract_formants

# 한국어 단모음 F1/F2 기준값 (Hz)
# 성인 남성 평균 기준 (음성학 문헌 참조)
# 화자에 따라 편차가 있으므로 넓은 범위로 매칭
VOWEL_FORMANTS = {
    "아": (800, 1200),   # /a/  F1 high, F2 mid
    "어": (600, 1000),   # /ʌ/  F1 mid, F2 mid-low
    "오": (450, 800),    # /o/  F1 mid-low, F2 low
    "우": (350, 750),    # /u/  F1 low, F2 low
    "으": (400, 1500),   # /ɯ/  F1 low, F2 mid-high
    "이": (300, 2200),   # /i/  F1 low, F2 high
    "에": (500, 1800),   # /e/  F1 mid-low, F2 mid-high
    "애": (650, 1700),   # /æ/  F1 mid, F2 mid-high
}

# F1, F2 가중치 (F1이 모음 구분에 더 중요)
F1_WEIGHT = 1.2
F2_WEIGHT = 1.0

DEBOUNCE_FRAMES = 3
VOWELS = list(VOWEL_FORMANTS.keys())


def _formant_distance(f1, f2, ref_f1, ref_f2):
    """포먼트 거리 (멜 스케일 기반)."""
    def to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    d1 = (to_mel(f1) - to_mel(ref_f1)) * F1_WEIGHT
    d2 = (to_mel(f2) - to_mel(ref_f2)) * F2_WEIGHT
    return math.sqrt(d1 * d1 + d2 * d2)


class FormantVowelClassifier:
    """포먼트 기반 모음 분류기. 캘리브레이션 불필요."""

    def __init__(self):
        self._current_vowel = ""
        self._current_conf = 0.0
        self._candidate = ""
        self._candidate_count = 0
        self._f1 = 0.0
        self._f2 = 0.0

        # 포먼트 스무딩
        self._f1_buf = deque(maxlen=5)
        self._f2_buf = deque(maxlen=5)

    @property
    def vowels(self):
        return VOWELS

    @property
    def is_trained(self):
        return True  # 항상 사용 가능

    def feed(self, audio: np.ndarray, sr: int):
        """오디오 프레임 입력 → 모음 분류."""
        formants = extract_formants(audio, sr, n_formants=2)
        if len(formants) < 2:
            return

        f1, f2 = formants[0], formants[1]

        # 스무딩 (중앙값)
        self._f1_buf.append(f1)
        self._f2_buf.append(f2)
        if len(self._f1_buf) < 3:
            return
        f1 = float(np.median(self._f1_buf))
        f2 = float(np.median(self._f2_buf))
        self._f1 = f1
        self._f2 = f2

        # 각 모음과의 거리 계산
        distances = {}
        for vowel, (ref_f1, ref_f2) in VOWEL_FORMANTS.items():
            distances[vowel] = _formant_distance(f1, f2, ref_f1, ref_f2)

        # 가장 가까운 모음
        best = min(distances, key=distances.get)
        best_dist = distances[best]

        # 신뢰도: 거리 기반 (가까울수록 높음)
        # 1위와 2위 거리 차이도 반영
        sorted_dists = sorted(distances.values())
        margin = sorted_dists[1] - sorted_dists[0] if len(sorted_dists) > 1 else 0
        conf = max(0.0, min(1.0, 1.0 - best_dist / 500.0 + margin / 300.0))

        # 디바운싱
        if best == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = best
            self._candidate_count = 1

        if self._candidate_count >= DEBOUNCE_FRAMES:
            self._current_vowel = self._candidate
            self._current_conf = conf

    def get_result(self) -> Tuple[str, float]:
        return self._current_vowel, self._current_conf

    def get_formants(self) -> Tuple[float, float]:
        """현재 스무딩된 F1, F2."""
        return self._f1, self._f2
