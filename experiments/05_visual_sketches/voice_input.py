"""
voice_input.py — 시각 스케치 공통 음성 입력 (마이크 → VoiceSignal)

역할:
  마이크 → 300ms 청크 → Praat Burg 포먼트 + F0 + RMS + jitter 근사
  + 7모음 연속 가중치(vowel_weights) 를 뽑아 VoiceSignal 1개로 묶어준다.

설계:
  - 02_formant/voice_data.py 의 VoiceFrame 과 같은 재료를 제공하되,
    스케치 폴더가 독립 실행되도록 자체 완결(self-contained)로 작성.
  - 각 스케치는 `VoiceListener` 만 쓰면 됨:
        lis = VoiceListener(); lis.start()
        sig = lis.latest()   # VoiceSignal | None
  - vowel_weights 는 단정적 분류가 아니라 "모든 모음에 대한 거리감"(연속 보간) —
    Reas 식 규칙 블렌딩의 입력으로 쓰라고 만든 값.

주의:
  jitter 는 cycle-to-cycle 가 아니라 최근 F0 변동으로 근사한 값(시각용 proxy).
"""

import threading
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import sounddevice as sd
import parselmouth

# ── 상수 (02_formant/config.py · morphing_demo.py 에서 가져온 값) ──
SAMPLE_RATE = 44100
GENDER_THRESH_HZ = 165   # F0 이 이보다 높으면 female 참조 사용

# 한국어 7모음 F1/F2 중심값 (Hz) — Yoon 2015. morphing_demo 와 동일.
VOWEL_REFS_FEMALE = {
    "아": (978, 1397), "에": (548, 2125), "이": (352, 2787),
    "오": (487, 840),  "우": (367, 660),  "으": (435, 1404), "어": (671, 1212),
}
VOWEL_REFS_MALE = {
    "아": (831, 1145), "에": (466, 1743), "이": (299, 2285),
    "오": (414, 689),  "우": (312, 541),  "으": (370, 1151), "어": (570, 994),
}
VOWELS = list(VOWEL_REFS_FEMALE.keys())


def bark(f):
    """Hz → Bark (지각 척도). 화자간 비교에 Hz 보다 안정적."""
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


@dataclass
class VoiceSignal:
    """300ms 청크 1개의 분석 결과 (시각 매핑용 재료)."""
    t: float                      # 측정 시각 (time.monotonic)
    voiced: bool                  # 유성음 여부
    f0: float                     # 기본 주파수 Hz (무성 0)
    f1: float                     # 포먼트 Hz (실패 0)
    f2: float
    f3: float
    rms: float                    # 청크 RMS (0~1 근사)
    jitter: float                 # F0 불안정도 근사 (0~1)
    vowel: str                    # 최댓값 모음 ("" = 무성)
    vowel_weights: Dict[str, float] = field(default_factory=dict)  # 7모음 연속 가중치 합=1


def _vowel_weights(f1, f2, f0, temp=2.0):
    """F1/F2 → 7모음 연속 가중치 (Bark 거리 softmax). 합 = 1."""
    refs = VOWEL_REFS_FEMALE if f0 >= GENDER_THRESH_HZ else VOWEL_REFS_MALE
    b1, b2 = bark(f1), bark(f2)
    dists = np.array([
        np.hypot(b1 - bark(c1), b2 - bark(c2)) for (c1, c2) in refs.values()
    ])
    w = np.exp(-dists / temp)
    w = w / (w.sum() + 1e-9)
    return {v: float(wi) for v, wi in zip(refs.keys(), w)}


class VoiceListener:
    """마이크를 열고 백그라운드에서 분석. latest() 로 최신 신호를 읽는다."""

    def __init__(self, chunk_sec=0.30, hop_sec=0.08, rms_gate=0.004):
        self.chunk_n = int(chunk_sec * SAMPLE_RATE)
        self.hop_n = int(hop_sec * SAMPLE_RATE)
        self.rms_gate = rms_gate
        self._audio_q = queue.Queue(maxsize=40)
        self._latest: Optional[VoiceSignal] = None
        self._lock = threading.Lock()
        self._f0_hist = deque(maxlen=8)
        self._running = False
        self._stream = None
        self._thread = None

    # ── 공개 API ──────────────────────────────────────────
    def start(self):
        block = int(SAMPLE_RATE * 0.05)
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=block, callback=self._on_audio,
        )
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stream.start()
        self._thread.start()
        return self

    def latest(self) -> Optional[VoiceSignal]:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()

    # ── 내부 ──────────────────────────────────────────────
    def _on_audio(self, indata, frames, time_info, status):
        try:
            self._audio_q.put_nowait(indata[:, 0].copy().astype(np.float32))
        except queue.Full:
            pass

    def _run(self):
        buf = np.zeros(0, dtype=np.float32)
        while self._running:
            try:
                block = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            buf = np.concatenate([buf, block])
            while len(buf) >= self.chunk_n:
                chunk = buf[:self.chunk_n]
                buf = buf[self.hop_n:]
                sig = self._analyze(chunk)
                with self._lock:
                    self._latest = sig

    def _analyze(self, audio) -> VoiceSignal:
        now = time.monotonic()
        audio = audio - np.mean(audio)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.rms_gate:
            return VoiceSignal(now, False, 0, 0, 0, 0, rms, 0.0, "", {})

        f0 = f1 = f2 = f3 = 0.0
        try:
            snd = parselmouth.Sound(audio.astype(np.float64),
                                    sampling_frequency=float(SAMPLE_RATE))
            dur = audio.shape[0] / SAMPLE_RATE
            t = dur / 2

            pitch = snd.to_pitch(time_step=None, pitch_floor=70, pitch_ceiling=500)
            v = pitch.get_value_at_time(t)
            f0 = float(v) if v and not np.isnan(v) else 0.0

            fmt = snd.to_formant_burg(
                time_step=None, max_number_of_formants=5,
                maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
            )
            def fval(n, lo, hi):
                x = fmt.get_value_at_time(n, t)
                return float(x) if (x and not np.isnan(x) and lo < x < hi) else 0.0
            f1 = fval(1, 100, 1500)
            f2 = fval(2, 200, 4000)
            f3 = fval(3, 1500, 5000)
        except Exception:
            pass

        # jitter 근사: 최근 F0 의 상대 표준편차
        jitter = 0.0
        if f0 > 0:
            self._f0_hist.append(f0)
            if len(self._f0_hist) >= 3:
                arr = np.array(self._f0_hist)
                jitter = float(min(1.0, np.std(arr) / (np.mean(arr) + 1e-9) * 4))

        weights, vowel = {}, ""
        if f1 > 0 and f2 > 0:
            weights = _vowel_weights(f1, f2, f0 if f0 > 0 else 200)
            vowel = max(weights, key=weights.get)

        return VoiceSignal(now, True, f0, f1, f2, f3, rms, jitter, vowel, weights)


# 단독 실행 시 — 콘솔에 값 출력 (마이크/추출 동작 확인용)
if __name__ == "__main__":
    lis = VoiceListener().start()
    print("말해보세요 (Ctrl+C 종료)")
    try:
        while True:
            s = lis.latest()
            if s and s.voiced:
                top = sorted(s.vowel_weights.items(), key=lambda x: -x[1])[:3]
                top_str = " ".join(f"{v}:{w:.2f}" for v, w in top)
                print(f"f0={s.f0:6.1f} f1={s.f1:6.1f} f2={s.f2:6.1f} "
                      f"rms={s.rms:.3f} jit={s.jitter:.2f} | {top_str}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        lis.stop()
        print("\n종료")
