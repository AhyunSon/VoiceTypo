"""화자 캘리브레이션 — 모음별 발성으로 개인 모음 공간 구축.

모음 하나씩 발성 → LPC → F1·F2·F3 → xyz 변환 → 3D 궤적 표시
→ 안정 구간 자동 감지 → 중심점 확정.

안정 구간 감지:
  - 최근 25프레임 xyz 표준편차 평균 < 0.022
  - 연속 18프레임 충족 시 확정

저장 형식:
  centroids = { '아': np.ndarray([x,y,z]), ... }
"""

import numpy as np
from collections import deque
from ..vowel.space import formant_to_xyz

VOWELS = ['아', '이', '우', '에', '오', '으', '어']

# 안정 구간 감지 파라미터
STABILITY_WINDOW = 25       # 표준편차 계산 윈도우
STABILITY_THRESHOLD = 0.022  # xyz 표준편차 평균 임계값
STABILITY_FRAMES = 18        # 연속 안정 프레임 수


class CalibrationSession:
    """한 모음에 대한 캘리브레이션 세션.

    사용:
        session = CalibrationSession('아')
        for each frame:
            result = session.feed(f1, f2, f3)
            if result is not None:
                centroid = result  # 확정된 xyz
    """

    def __init__(self, vowel):
        self.vowel = vowel
        self._history = deque(maxlen=STABILITY_WINDOW)
        self._stable_count = 0
        self._confirmed = False
        self._centroid = None
        self._all_xyz = []  # 궤적 전체 기록 (3D 시각화용)

    def feed(self, f1, f2, f3):
        """포먼트 입력. 안정 구간 감지 시 centroid 반환, 아니면 None."""
        if self._confirmed:
            return self._centroid

        if np.isnan(f1) or np.isnan(f2) or np.isnan(f3):
            self._stable_count = 0
            return None

        xyz = formant_to_xyz(f1, f2, f3)

        # 아웃라이어 필터: 이전 점과 너무 멀면 버림
        if len(self._all_xyz) > 0:
            dist = float(np.linalg.norm(xyz - self._all_xyz[-1]))
            if dist > 0.3:
                return None

        self._history.append(xyz)
        self._all_xyz.append(xyz)

        if len(self._history) < STABILITY_WINDOW:
            return None

        # 표준편차 계산
        pts = np.array(list(self._history))  # (N, 3)
        std_mean = float(np.mean(np.std(pts, axis=0)))

        if std_mean < STABILITY_THRESHOLD:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if self._stable_count >= STABILITY_FRAMES:
            self._centroid = np.mean(pts, axis=0)
            self._confirmed = True
            return self._centroid

        return None

    @property
    def is_confirmed(self):
        return self._confirmed

    @property
    def centroid(self):
        return self._centroid

    @property
    def trajectory(self):
        """지금까지의 xyz 궤적."""
        return np.array(self._all_xyz) if self._all_xyz else np.empty((0, 3))

    @property
    def stability_progress(self):
        """안정 진행도 0~1."""
        return min(self._stable_count / STABILITY_FRAMES, 1.0)

    def force_confirm(self, centroid):
        """외부에서 강제 확정 (건너뛰기 시 기본값 사용)."""
        self._centroid = np.asarray(centroid, dtype=np.float64)
        self._confirmed = True

    def reset(self):
        self._history.clear()
        self._stable_count = 0
        self._confirmed = False
        self._centroid = None
        self._all_xyz = []


class Calibrator:
    """전체 캘리브레이션 관리자.

    7개 모음을 순차적으로 캘리브레이션.

    사용:
        cal = Calibrator()
        while not cal.is_complete:
            result = cal.feed(f1, f2, f3)
            if result is not None:
                print(f"{cal.current_vowel} confirmed: {result}")
                cal.advance()
        centroids = cal.centroids
    """

    def __init__(self, vowels=None):
        self.vowels = vowels or list(VOWELS)
        self._sessions = {v: CalibrationSession(v) for v in self.vowels}
        self._idx = -1  # 시작 시 선택된 모음 없음 — 유저가 직접 선택해야 함

    @property
    def current_vowel(self):
        if 0 <= self._idx < len(self.vowels):
            return self.vowels[self._idx]
        return None

    @property
    def current_session(self):
        v = self.current_vowel
        return self._sessions[v] if v else None

    @property
    def is_complete(self):
        return all(s.is_confirmed for s in self._sessions.values())

    @property
    def centroids(self):
        """확정된 중심점 dict. 미확정이면 None."""
        if not self.is_complete:
            return None
        return {v: s.centroid for v, s in self._sessions.items()}

    @property
    def progress(self):
        """(완료 수, 전체 수)."""
        done = sum(1 for s in self._sessions.values() if s.is_confirmed)
        return (done, len(self.vowels))

    def feed(self, f1, f2, f3):
        """현재 모음에 포먼트 입력. 확정 시 centroid 반환."""
        session = self.current_session
        if session is None:
            return None
        return session.feed(f1, f2, f3)

    def advance(self):
        """다음 모음으로 이동."""
        if self._idx < len(self.vowels) - 1:
            self._idx += 1

    def skip(self):
        """현재 모음 건너뛰기."""
        self.advance()

    def retry(self):
        """현재 모음 재시도."""
        session = self.current_session
        if session:
            session.reset()

    def select_vowel(self, vowel):
        """특정 모음으로 이동."""
        if vowel in self.vowels:
            self._idx = self.vowels.index(vowel)
            session = self.current_session
            if session and not session.is_confirmed:
                session.reset()

    def deselect(self):
        """현재 선택 해제 — 유저가 다음 모음을 직접 선택해야 함."""
        self._idx = -1

    def get_session(self, vowel):
        return self._sessions.get(vowel)

    def save(self, path):
        """캘리브레이션 결과 저장."""
        centroids = self.centroids
        if centroids is None:
            raise ValueError("캘리브레이션이 완료되지 않았습니다.")
        np.savez(path, **{v: c for v, c in centroids.items()})

    @staticmethod
    def load(path):
        """저장된 캘리브레이션 결과 로드."""
        data = np.load(path)
        return {k: data[k] for k in data.files}
