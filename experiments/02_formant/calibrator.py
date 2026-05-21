"""
calibrator.py — 화자 self-reference 캘리브레이션 (경로 C)

역할:
  관객 시작 시 7모음 한 번씩 발음 → 그 화자의 user_refs 직접 구성.
  학계 _REFS 미사용. 화자 자기 기준 분류.

설계 원칙:
  - 다화자 환경: 매 사용자마다 새 캘리브레이션 (영구 저장 X)
  - 학계 데이터 의존 0 (검증용 관대 범위만 사용)
  - 캘리브레이션 미완료 시 → vowel_classifier 가 학계 _REFS fallback

출력 형식:
  user_refs = {
      "아": (F1_med, F1_sd, F2_med, F2_sd, F3_med, F3_sd),
      "에": (...), ...
  }
  → vowel_classifier._REFS 와 동일 구조 → 즉시 교체 가능

캘리브레이션 흐름:
  cal = VowelCalibrator()
  cal.start()
  while cal.state == "recording":
      # UI 표시: cal.current_vowel
      while cal.current_sample_count() < _MIN_SAMPLES_PER_VOWEL + buffer:
          f1, f2, f3 = engine.extract(chunk)[...]
          cal.feed_chunk(f1, f2, f3)
      ok, msg = cal.advance_vowel()    # 검증 후 다음 모음
      if not ok: # 재시도 (UI 메시지 표시)
  # 자동 finalize → cal.user_refs 채워짐
  vowel_classifier.set_user_refs(cal.user_refs)
"""

from typing import Optional
import numpy as np


_VOWEL_ORDER = ["아", "에", "이", "오", "우", "으", "어"]

# 모음당 최소 샘플 수 (안정 평균 산출용).
# 청크 = 300ms, 모음당 2초 발화 → 최대 ~6 청크 가능.
_MIN_SAMPLES_PER_VOWEL = 3

# IQR outlier 제거 K (Q1 - K*IQR, Q3 + K*IQR 밖은 제외)
_OUTLIER_K = 1.5

# 측정 SD 가 너무 작을 때 (샘플 적음) 적용할 floor (Hz).
# 학계 _REFS 의 SD 평균 절반 정도.
_SD_FLOOR = {1: 60.0, 2: 100.0, 3: 150.0}

# 발음 실패 검출용 관대 범위.
# 분류에는 사용하지 않음 — 캘리브레이션 단계 발음 검증용으로만 사용.
# (예: "이"를 발음해야 할 때 "아"를 말한 경우 자동 검출)
#
# 설계 원칙: 화자별 발음 변동이 크므로 좁힐수록 false-reject 증가.
# 학계 코퍼스 평균 ± 3SD 보다 넓게 잡고, 인접 모음 분리만 보장.
_VALIDATION_RANGES = {
    # vowel: (F1_lo, F1_hi, F2_lo, F2_hi)
    "아": (500, 1300,  800, 1900),
    "에": (250,  800, 1400, 2700),
    "이": (150,  550, 1800, 3400),
    "오": (250,  700,  400, 1200),
    "우": (150,  550,  330,  950),
    "으": (200,  650,  800, 1900),
    "어": (350,  900,  700, 1600),
}


class VowelCalibrator:
    """경로 C — 화자 self-reference 캘리브레이션 상태 머신.

    States: "ready" → "recording" (× 7 vowels) → "done"
    """

    def __init__(self):
        self._idx: int = -1   # -1 = ready 전, 0~6 = 녹음 중, 7 = done
        self._buffers: dict[str, list] = {v: [] for v in _VOWEL_ORDER}
        self.user_refs: Optional[dict] = None

    # ── lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """캘리브레이션 시작. 첫 모음(아)부터 녹음 대기."""
        self._idx = 0
        self._buffers = {v: [] for v in _VOWEL_ORDER}
        self.user_refs = None

    def reset(self) -> None:
        self._idx = -1
        self._buffers = {v: [] for v in _VOWEL_ORDER}
        self.user_refs = None

    # ── state 조회 ─────────────────────────────────────────────

    @property
    def state(self) -> str:
        if self._idx < 0:
            return "ready"
        if self._idx >= len(_VOWEL_ORDER):
            return "done"
        return "recording"

    @property
    def current_vowel(self) -> Optional[str]:
        if 0 <= self._idx < len(_VOWEL_ORDER):
            return _VOWEL_ORDER[self._idx]
        return None

    @property
    def progress(self) -> tuple:
        """(완료 수, 전체 수). 진행 게이지용."""
        return (max(0, self._idx), len(_VOWEL_ORDER))

    @property
    def is_ready(self) -> bool:
        return self.user_refs is not None

    # ── 측정 수집 ──────────────────────────────────────────────

    def feed_chunk(self, f1: float, f2: float, f3: float) -> None:
        """현재 모음에 대한 측정 1청크 추가. None/비정상 값은 무시."""
        if self.current_vowel is None:
            return
        if f1 is None or f2 is None or f3 is None:
            return
        if not (100 < f1 < 1500 and 200 < f2 < 4000 and 1000 < f3 < 5000):
            return
        self._buffers[self.current_vowel].append((float(f1), float(f2), float(f3)))

    def current_sample_count(self) -> int:
        """현재 모음에 대해 누적된 청크 수."""
        if self.current_vowel is None:
            return 0
        return len(self._buffers[self.current_vowel])

    # ── 진행 / 검증 ────────────────────────────────────────────

    def advance_vowel(self, validate: bool = True) -> tuple:
        """현재 모음 검증 후 다음으로 이동.

        Args:
            validate: True 면 _VALIDATION_RANGES 검사 수행 (기본).
                      False 면 검증 건너뜀 (오프라인 회귀 테스트용 — 사용자가
                      이미 어떤 wav 가 어떤 모음인지 라벨링한 데이터를 다룰 때).

        Returns:
            (ok: bool, msg: str)
              ok = True  → 다음 모음 (또는 7개 완료 시 finalize)
              ok = False → 재시도 (현재 모음 버퍼 초기화). msg 에 사유.
        """
        v = self.current_vowel
        if v is None:
            return False, "진행 중이 아님"

        samples = self._buffers[v]
        if len(samples) < _MIN_SAMPLES_PER_VOWEL:
            return False, f"샘플 부족 ({len(samples)}/{_MIN_SAMPLES_PER_VOWEL})"

        if validate:
            f1_arr = np.array([s[0] for s in samples])
            f2_arr = np.array([s[1] for s in samples])
            f1_med = float(np.median(f1_arr))
            f2_med = float(np.median(f2_arr))

            lo1, hi1, lo2, hi2 = _VALIDATION_RANGES[v]
            if not (lo1 <= f1_med <= hi1 and lo2 <= f2_med <= hi2):
                self._buffers[v] = []
                return False, (f"발음 영역 벗어남 "
                               f"(F1≈{f1_med:.0f}/F2≈{f2_med:.0f})")

        self._idx += 1
        if self._idx >= len(_VOWEL_ORDER):
            self._finalize()
        return True, "OK"

    # ── 최종 user_refs 산출 ────────────────────────────────────

    def _finalize(self) -> None:
        """7모음 수집 완료 시 호출. user_refs dict 구축.

        - median 으로 중심값 (outlier 강건)
        - sample SD (ddof=1), floor 적용
        - IQR outlier 제거 후 통계
        """
        refs = {}
        for v in _VOWEL_ORDER:
            arr = np.array(self._buffers[v])               # shape (N, 3)
            arr = self._reject_outliers(arr)
            f1, f2, f3 = arr[:, 0], arr[:, 1], arr[:, 2]
            refs[v] = (
                float(np.median(f1)),
                self._safe_sd(f1, _SD_FLOOR[1]),
                float(np.median(f2)),
                self._safe_sd(f2, _SD_FLOOR[2]),
                float(np.median(f3)),
                self._safe_sd(f3, _SD_FLOOR[3]),
            )
        self.user_refs = refs

    @staticmethod
    def _safe_sd(arr: np.ndarray, floor: float) -> float:
        """샘플 SD; 샘플이 1개거나 SD 너무 작으면 floor 반환."""
        if len(arr) < 2:
            return floor
        sd = float(np.std(arr, ddof=1))
        return max(sd, floor)

    @staticmethod
    def _reject_outliers(arr: np.ndarray) -> np.ndarray:
        """IQR 기반 outlier 제거 (각 차원 독립). 너무 많이 제거되면 원본 유지."""
        if len(arr) < 4:
            return arr
        q1 = np.percentile(arr, 25, axis=0)
        q3 = np.percentile(arr, 75, axis=0)
        iqr = q3 - q1
        lo = q1 - _OUTLIER_K * iqr
        hi = q3 + _OUTLIER_K * iqr
        mask = np.all((arr >= lo) & (arr <= hi), axis=1)
        kept = arr[mask]
        return kept if len(kept) >= 3 else arr


# ══════════════════════════════════════════
# 단위 테스트 (모듈 직접 실행 시 동작 확인)
# ══════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import random
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(0)

    cal = VowelCalibrator()

    # 학계 _REFS (여성) 중심값 근처에서 샘플 생성 → 정상 진행 시뮬레이션
    fake_means = {
        "아": (978, 1397, 2600),
        "에": (548, 2125, 2980),
        "이": (352, 2787, 3180),
        "오": (487,  840, 2680),
        "우": (367,  660, 2270),
        "으": (435, 1404, 2720),
        "어": (671, 1212, 2640),
    }

    cal.start()
    print(f"[start] state={cal.state}, current={cal.current_vowel}")

    for vowel in _VOWEL_ORDER:
        # 5 청크 시뮬레이션 (±5% 변동)
        m1, m2, m3 = fake_means[vowel]
        for _ in range(5):
            cal.feed_chunk(m1 * (1 + random.uniform(-0.05, 0.05)),
                           m2 * (1 + random.uniform(-0.05, 0.05)),
                           m3 * (1 + random.uniform(-0.05, 0.05)))
        ok, msg = cal.advance_vowel()
        print(f"  {vowel}: {msg} (samples={cal._buffers[vowel].__len__()})")
        assert ok, f"FAIL on {vowel}: {msg}"

    print(f"[done] state={cal.state}, is_ready={cal.is_ready}")
    print("\n[user_refs]")
    for v, ref in cal.user_refs.items():
        print(f"  {v}: F1={ref[0]:.0f}±{ref[1]:.0f}  "
              f"F2={ref[2]:.0f}±{ref[3]:.0f}  "
              f"F3={ref[4]:.0f}±{ref[5]:.0f}")

    # 발음 실패 시뮬레이션
    print("\n[fail-test] '아' 위치에서 '이' 발음")
    cal.reset()
    cal.start()
    m1, m2, m3 = fake_means["이"]
    for _ in range(5):
        cal.feed_chunk(m1, m2, m3)
    ok, msg = cal.advance_vowel()
    print(f"  expected=False got={ok}: {msg}")
    assert not ok, "발음 실패 감지 안 됨"

    print("\n✓ all tests passed")
