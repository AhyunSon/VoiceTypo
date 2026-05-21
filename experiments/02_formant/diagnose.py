"""
diagnose.py — 3초 녹음 + 실제 추출 값 + 분류 결과 확인용

사용:
  python diagnose.py
  → 각 모음 발음 안내 → 추출 결과 표시
"""

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SAMPLE_RATE, FORMANT_CEILINGS
from formant_engine import FormantEngine
from vowel_classifier import classify_vowel


def record(seconds: float = 2.0) -> np.ndarray:
    n = int(seconds * SAMPLE_RATE)
    a = sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return a[:, 0]


def diagnose(vowel: str, gender: str = "male"):
    print(f"\n[{vowel}] 발음하세요... ", end="", flush=True)
    for i in range(3, 0, -1):
        print(f"{i}.", end=" ", flush=True)
        time.sleep(0.5)
    print("녹음 중...", end="", flush=True)
    audio = record(2.0)
    print(" 완료")

    rms = float(np.sqrt(np.mean(audio**2)))
    print(f"  RMS = {rms:.4f}")

    engine = FormantEngine()
    res = engine.extract(audio, gender=gender, ceilings=FORMANT_CEILINGS)

    print(f"  is_voiced = {res['is_voiced']}")
    print(f"  F0 = {res['f0']:.0f}" if res.get("f0") else "  F0 = None")
    print(f"  F1 = {res['f1']}")
    print(f"  F2 = {res['f2']}")
    print(f"  F3 = {res['f3']}")
    print(f"  confidence = {res['confidence']:.2f}")

    if res.get("f1") and res.get("f2"):
        v, c = classify_vowel(res["f1"], res["f2"], gender, f3=res.get("f3"))
        ok = "✓" if v == vowel else "✗"
        print(f"  → 분류: {v} (conf={c:.2f}) {ok}")


def main():
    print("=" * 60)
    print("진단 모드 — 3초 카운트다운 후 2초 녹음")
    print("=" * 60)
    print()

    # 사용자 성별 빠르게 결정
    print("성별? male / female / 엔터로 male:")
    ans = input("> ").strip().lower()
    gender = "female" if ans == "female" else "male"
    print(f"  → {gender} 참조값 사용")

    print(f"\nFORMANT_CEILINGS = {FORMANT_CEILINGS}")
    print()

    for v in ["아", "에", "이", "오", "우", "으", "어"]:
        diagnose(v, gender=gender)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
