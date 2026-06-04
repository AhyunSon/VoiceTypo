"""VowelTracker — 모음 공간 추적 3레이어.

L1 소프트 위치: 역거리 가중합 → blend_weights + xyz (연속)
L2 속도 벡터:  Δxyz 지수이동평균 → velocity
L3 히스테리시스: N프레임 다수결 → label (확정 레이블)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from ..vowel.space import DEFAULT_CENTROIDS


@dataclass
class VowelPosition:
    """VowelTracker 출력."""
    xyz: np.ndarray                    # 현재 xyz 좌표
    weights: dict = field(default_factory=dict)  # 모음별 소프트 가중치 (합=1)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 속도 벡터
    label: str = ''                    # 히스테리시스 확정 레이블
    speed: float = 0.0                 # 속도 크기 (스칼라)


class VowelTracker:
    """실시간 모음 공간 추적기.

    Args:
        centroids: 캘리브레이션된 모음 중심점 dict.
                   None이면 기본 참조값 사용.
        softness: 역거리 가중 지수. 높을수록 날카로운 분류.
        hyst_frames: 히스테리시스 윈도우 크기.
        hyst_ratio: 다수결 비율 (0.7 = 70%).
        vel_alpha: 속도 지수이동평균 계수.
    """

    def __init__(self, centroids=None, softness=8.0,
                 hyst_frames=6, hyst_ratio=0.7, vel_alpha=0.4):
        self.centroids = dict(centroids) if centroids else dict(DEFAULT_CENTROIDS)
        self.softness = softness
        self.hyst_frames = hyst_frames
        self.hyst_ratio = hyst_ratio
        self.vel_alpha = vel_alpha

        self._prev_xyz = None
        self._velocity = np.zeros(3, dtype=np.float64)
        self._label_history = deque(maxlen=hyst_frames)
        self._current_label = ''

    def update(self, xyz):
        """한 프레임 업데이트.

        Args:
            xyz: np.ndarray shape (3,) — formant_to_xyz() 출력

        Returns:
            VowelPosition
        """
        xyz = np.asarray(xyz, dtype=np.float64)

        # NaN 체크
        if np.any(np.isnan(xyz)):
            return VowelPosition(
                xyz=xyz,
                weights={},
                velocity=self._velocity.copy(),
                label=self._current_label,
                speed=float(np.linalg.norm(self._velocity)),
            )

        # ── L1: 소프트 위치 (역거리 가중합) ──
        weights = self._compute_weights(xyz)

        # ── L2: 속도 벡터 (지수이동평균) ──
        if self._prev_xyz is not None and not np.any(np.isnan(self._prev_xyz)):
            delta = xyz - self._prev_xyz
            self._velocity = (self.vel_alpha * delta
                              + (1.0 - self.vel_alpha) * self._velocity)
        self._prev_xyz = xyz.copy()

        # ── L3: 히스테리시스 (다수결) ──
        top_vowel = max(weights, key=weights.get)
        self._label_history.append(top_vowel)
        self._current_label = self._hysteresis_vote()

        speed = float(np.linalg.norm(self._velocity))

        return VowelPosition(
            xyz=xyz,
            weights=weights,
            velocity=self._velocity.copy(),
            label=self._current_label,
            speed=speed,
        )

    def _compute_weights(self, xyz):
        """역거리 가중합 → 소프트 가중치 (합=1)."""
        dists = {}
        for name, centroid in self.centroids.items():
            d = float(np.linalg.norm(xyz - centroid))
            dists[name] = d

        # 역거리^softness
        inv_weights = {}
        for name, d in dists.items():
            inv_weights[name] = 1.0 / (d ** self.softness + 1e-12)

        total = sum(inv_weights.values())
        return {name: w / total for name, w in inv_weights.items()}

    def _hysteresis_vote(self):
        """최근 N프레임 다수결. ratio 이상이면 레이블 전환."""
        if len(self._label_history) == 0:
            return self._current_label

        # 빈도 카운트
        counts = {}
        for lbl in self._label_history:
            counts[lbl] = counts.get(lbl, 0) + 1

        top = max(counts, key=counts.get)
        ratio = counts[top] / len(self._label_history)

        if ratio >= self.hyst_ratio:
            return top

        # 기준 미달 → 기존 레이블 유지
        return self._current_label if self._current_label else top

    def last_position(self):
        """VAD 비활성 시 마지막 위치를 그대로 반환 (움직임 없음)."""
        xyz = self._prev_xyz if self._prev_xyz is not None else np.zeros(3)
        return VowelPosition(
            xyz=xyz,
            weights={},
            velocity=np.zeros(3),
            label=self._current_label,
            speed=0.0,
        )

    def reset(self):
        """상태 초기화."""
        self._prev_xyz = None
        self._velocity = np.zeros(3, dtype=np.float64)
        self._label_history.clear()
        self._current_label = ''
