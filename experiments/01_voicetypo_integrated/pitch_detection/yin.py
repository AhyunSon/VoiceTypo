"""YIN 피치 감지 알고리즘.

HTML 원본의 detectPitch()를 numpy로 포팅.
프레임 단위 실시간 처리.

사용법:
    detector = YinDetector(sample_rate=44100)
    freq, rms = detector.detect(audio_chunk)
"""

import numpy as np

VOICE_MIN_FREQ = 80.0
VOICE_MAX_FREQ = 1100.0
SILENCE_THRESHOLD = 0.001
YIN_THRESHOLD = 0.1


class YinDetector:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        # lag 범위: 주파수 범위에 대응하는 샘플 수
        self.tau_min = int(sample_rate / VOICE_MAX_FREQ)
        self.tau_max = int(sample_rate / VOICE_MIN_FREQ)

    def detect(self, audio: np.ndarray) -> tuple:
        """오디오 프레임에서 피치와 RMS를 추출.
        Returns: (frequency_hz, rms)
            frequency_hz: 감지된 주파수 (Hz). 감지 실패 시 0.0
            rms: 볼륨 (Root Mean Square)
        """
        rms = float(np.sqrt(np.mean(audio ** 2)))

        if rms < SILENCE_THRESHOLD:
            return 0.0, rms

        n = len(audio)
        tau_max = min(self.tau_max, n // 2)
        tau_min = self.tau_min

        if tau_max <= tau_min:
            return 0.0, rms

        # 1. 차분 함수 (벡터화)
        d = np.zeros(tau_max, dtype=np.float32)
        for tau in range(1, tau_max):
            diff = audio[:n - tau] - audio[tau:n]
            d[tau] = np.sum(diff ** 2)

        # 2. 누적 평균 정규화 (CMNDF)
        cmndf = np.ones(tau_max, dtype=np.float32)
        running_sum = 0.0
        for tau in range(1, tau_max):
            running_sum += d[tau]
            if running_sum > 0:
                cmndf[tau] = d[tau] * tau / running_sum
            else:
                cmndf[tau] = 1.0

        # 3. 임계값 이하 첫 번째 최솟값 찾기
        tau_est = 0
        for tau in range(tau_min, tau_max - 1):
            if cmndf[tau] < YIN_THRESHOLD:
                # 극소점 찾기: 값이 다시 올라가기 시작하는 지점
                while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                tau_est = tau
                break

        if tau_est == 0:
            return 0.0, rms

        # 4. 포물선 보간
        tau_est = self._parabolic_interpolation(cmndf, tau_est)

        freq = self.sample_rate / tau_est

        if freq < VOICE_MIN_FREQ or freq > VOICE_MAX_FREQ:
            return 0.0, rms

        return float(freq), rms

    @staticmethod
    def _parabolic_interpolation(array, index):
        """포물선 보간으로 정밀 위치 추정."""
        if index <= 0 or index >= len(array) - 1:
            return float(index)

        s0 = array[index - 1]
        s1 = array[index]
        s2 = array[index + 1]

        denom = 2.0 * (2.0 * s1 - s0 - s2)
        if abs(denom) < 1e-12:
            return float(index)

        return index + (s0 - s2) / denom
