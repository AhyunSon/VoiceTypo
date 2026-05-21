"""
voice_data.py — 음향 분석 결과 데이터 인터페이스 (환경 무관)

역할:
  formant_engine + vowel_classifier 출력을 외부 시각화 환경
  (Processing / TouchDesigner / Web 등) 으로 전달하기 위한
  표준 데이터 구조를 정의한다.

설계 원칙:
  - 데이터 채널(OSC / WebSocket / callback) 미정 → 인터페이스만 정의
  - 외부 환경 종속 코드 없음 (직렬화 기본 dict 변환만 제공)
  - 단일 분석 청크(300 ms) 결과 = VoiceFrame 1개

사용 시점:
  방법 1(포먼트) / 방법 2(Whisper) / 방법 3(CNN) 인식 정확도 비교 후,
  작품 디자인 방향이 확정되면 이 인터페이스 위에 시각화 환경에 맞는
  직렬화/전송 레이어를 추가한다.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass
class VoiceFrame:
    """
    300 ms 분석 청크 1개의 음향 분석 + 모음 인식 결과.

    Fields
    ------
    timestamp : float
        청크 분석 종료 시각 (초, time.monotonic 기준).
    is_voiced : bool
        유성음 판정 결과 (3-조건 VAD + pyworld).
    f0 : float
        기본 주파수 (Hz). is_voiced=False 이면 0.0.
    f1, f2, f3 : float
        Praat Burg 포먼트 (Hz). 측정 실패 시 0.0.
    rms : float
        청크 RMS 진폭 (정규화 0~1).
    jitter : float
        F0 local jitter (0~1). 음성 떨림 지표.
    vowel : str
        분류된 모음 ("아"/"에"/"이"/"오"/"우"/"으"/"어") 또는 "" (미인식).
    confidence : float
        분류 신뢰도 (0~1). Bark-Mahalanobis 거리 기반.
    vowel_distances : Dict[str, float]
        7개 모음 각각에 대한 Bark-Mahalanobis 거리.
        시각화 가중치(연속 보간)용 — 인식이 단정적이지 않은 경우에도
        모든 모음에 대한 "거리감"을 시각적으로 표현할 수 있도록.
    """
    timestamp: float
    is_voiced: bool
    f0: float
    f1: float
    f2: float
    f3: float
    rms: float
    jitter: float
    vowel: str
    confidence: float
    vowel_distances: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """직렬화용 dict 변환 (OSC / WebSocket / JSON 어디서든 사용 가능)."""
        return asdict(self)
