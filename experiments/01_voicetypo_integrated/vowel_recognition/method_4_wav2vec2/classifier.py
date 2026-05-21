"""wav2vec2 CTC 기반 모음 분류기.

한국어 fine-tuned wav2vec2 모델의 프레임별 자모 확률에서
모음을 직접 판별. 캘리브레이션 완전 불필요.
"""

import numpy as np
from typing import Tuple
from collections import deque

from .features import Wav2Vec2KoreanCTC

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]
DEBOUNCE_FRAMES = 2


class Wav2Vec2VowelClassifier:
    """wav2vec2 CTC 기반 모음 분류기. 캘리브레이션 불필요."""

    def __init__(self):
        self._ctc = Wav2Vec2KoreanCTC()
        self._current_vowel = ""
        self._current_conf = 0.0
        self._candidate = ""
        self._candidate_count = 0

        # 확률 스무딩
        self._prob_buf = deque(maxlen=3)

    @property
    def vowels(self):
        return VOWELS

    @property
    def is_trained(self):
        return True  # 항상 사용 가능

    def feed(self, audio: np.ndarray, sr: int):
        """오디오 프레임 입력 → 모음 분류."""
        probs = self._ctc.get_vowel_probs(audio, sr)
        if not probs:
            return

        # 스무딩
        self._prob_buf.append(probs)
        if len(self._prob_buf) < 2:
            return

        # 평균 확률
        avg = {}
        for v in VOWELS:
            vals = [p.get(v, 0.0) for p in self._prob_buf]
            avg[v] = sum(vals) / len(vals)

        # 모음 확률 합 (음성이 모음인지 판단)
        vowel_total = sum(avg.values())
        if vowel_total < 0.01:
            return

        best = max(avg, key=avg.get)
        best_prob = avg[best]

        # 신뢰도: 모음 확률 중 best의 비율
        conf = best_prob / vowel_total if vowel_total > 0 else 0.0

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
