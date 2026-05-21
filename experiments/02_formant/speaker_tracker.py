"""
speaker_tracker.py — F0 기반 화자 유형 추적 (간소화 버전)

역할:
  EWM(지수 가중 이동평균) F0 → 목소리 유형 판별 (남성/여성/아동)
  → vowel_classifier 의 _REFS 선택에 사용 (male / female).

정리 이력 (2026-04-28, Stage 2-3):
  - F0 기반 scale 제거 (집단 평균 / 화자 F0 정규화)
    이유: scale 1.50× 가 남성 포먼트를 과보정해 22.9% baseline 의 부수 원인.
  - voice_type 별 ceiling 목록 제거 → 단일 5500 으로 통일.
  - 호환용 scale / clf_gender 속성도 Stage 3 에서 완전 제거.

EWM α = 0.06 (300 ms 청크 기준 ~10초 등가 윈도우).
READY_N = 30 voiced 청크 (~9초) 누적 후 'ready' 플래그.
"""

_CHILD_THRESH  = 280.0   # F0 > 280Hz → 아동
_MALE_THRESH   = 165.0   # F0 < 165Hz → 남성

_EWM_ALPHA = 0.06
_READY_N   = 30


class SpeakerF0Tracker:
    """
    매 voiced 청크에서 F0 받아 voice_type 판별.

    Properties
    ----------
    voice_type : "female" | "male" | "child"
    ready      : bool   충분한 F0 데이터 누적 여부
    praat_gender : str  vowel_classifier 의 _REFS 키 ("male" | "female")
    formant_ceilings : list  Praat 호출용 ceiling 리스트 (현재 [5500] 고정)
    """

    def __init__(self):
        self._ewm_f0:   float | None = None
        self._n:        int          = 0
        self.voice_type: str         = "female"
        self.ready:      bool        = False

    # ── 공개 API ──────────────────────────────────────────────────

    def update(self, f0: float) -> None:
        """voiced 청크 F0 1개 공급. 범위 외 값은 무시."""
        if not (50.0 < f0 < 600.0):
            return

        if self._ewm_f0 is None:
            self._ewm_f0 = f0
        else:
            self._ewm_f0 = (1.0 - _EWM_ALPHA) * self._ewm_f0 + _EWM_ALPHA * f0
        self._n += 1

        f = self._ewm_f0
        if f < _MALE_THRESH:
            self.voice_type = "male"
        elif f > _CHILD_THRESH:
            self.voice_type = "child"
        else:
            self.voice_type = "female"

        if self._n >= _READY_N:
            self.ready = True

    def reset(self) -> None:
        self._ewm_f0   = None
        self._n        = 0
        self.voice_type = "female"
        self.ready     = False

    @property
    def formant_ceilings(self) -> list:
        from config import FORMANT_CEILINGS
        return FORMANT_CEILINGS

    @property
    def praat_gender(self) -> str:
        """vowel_classifier _REFS 키. child → female refs 사용."""
        return "male" if self.voice_type == "male" else "female"

    def status(self) -> str:
        """UI 상태 표시 문자열."""
        vt_label = {"female": "여성", "male": "남성", "child": "아동"}
        label = vt_label.get(self.voice_type, self.voice_type)
        if self._ewm_f0 is None:
            return "화자 분석 대기"
        if not self.ready:
            pct = int(self._n / _READY_N * 100)
            return f"{label} F0≈{self._ewm_f0:.0f}Hz  화자분석 {pct}%"
        return f"{label} F0≈{self._ewm_f0:.0f}Hz"
