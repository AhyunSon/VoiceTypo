"""수동 룰 기반 VAD (Voice Activity Detection).

HTML 원본 방식 포팅:
  1. RMS > max(0.005, noiseFloor * 2.5)
  2. 유효 피치 존재 (80~1100Hz)
  3. 히스테리시스: ON 3프레임 연속, OFF 15프레임 연속

사용법:
    vad = VoiceActivityDetector()
    vad.update(rms, freq)
    if vad.is_active:
        ...
"""

NOISE_FLOOR_ALPHA = 0.02  # EMA 적응 속도
NOISE_FLOOR_MULT = 2.5
MIN_RMS = 0.005

ON_FRAMES = 3     # 연속 N프레임 조건 만족 시 ON
OFF_FRAMES = 15   # 연속 N프레임 조건 불만족 시 OFF


class VoiceActivityDetector:
    def __init__(self):
        self._noise_floor = 0.01
        self._on_count = 0
        self._off_count = 0
        self.is_active = False

    def update(self, rms: float, freq: float):
        """매 프레임 호출. rms와 감지된 주파수를 받아 VAD 상태 갱신."""
        # 노이즈 플로어 적응 (음성 비활성 구간에서만)
        if not self.is_active:
            self._noise_floor += NOISE_FLOOR_ALPHA * (rms - self._noise_floor)

        # 조건: RMS 충분 + 유효 피치
        threshold = max(MIN_RMS, self._noise_floor * NOISE_FLOOR_MULT)
        voice_detected = rms > threshold and freq > 0

        if voice_detected:
            self._on_count += 1
            self._off_count = 0
            if not self.is_active and self._on_count >= ON_FRAMES:
                self.is_active = True
        else:
            self._off_count += 1
            self._on_count = 0
            if self.is_active and self._off_count >= OFF_FRAMES:
                self.is_active = False

    def reset(self):
        self._noise_floor = 0.01
        self._on_count = 0
        self._off_count = 0
        self.is_active = False
