"""
diagnose_live.py — 6초간 한 모음 sustained 발음 → 매 500ms 청크의 F1/F2/F3 출력

사용:
  python diagnose_live.py
  → 모음 입력 → 6초 발음 → 12 chunks (500ms each) 의 측정값
  → 안정적이면 같은 값, 불안정이면 jump
"""

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SAMPLE_RATE, FORMANT_CEILINGS, ANALYSIS_WIN_SEC
from formant_engine import FormantEngine


CHUNK_SEC = ANALYSIS_WIN_SEC  # 500ms
TOTAL_SEC = 3.0
N_CHUNKS = int(TOTAL_SEC / CHUNK_SEC)


def main():
    print(f"ANALYSIS_WIN_SEC = {CHUNK_SEC*1000:.0f}ms")
    print(f"FORMANT_CEILINGS = {FORMANT_CEILINGS}")
    print()
    print("성별? male / female / 엔터로 female:")
    ans = input("> ").strip().lower()
    gender = "male" if ans == "male" else "female"

    print()
    print(f"sustained 모음 발음할 것 (예: '아'):")
    vowel = input("> ").strip()
    if not vowel:
        return

    print(f"\n{TOTAL_SEC:.0f}초 간 [{vowel}] 일정하게 발음:")
    for i in range(3, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(0.7)
    print("녹음 시작 — 발음하세요")

    full = sd.rec(int(TOTAL_SEC * SAMPLE_RATE),
                  samplerate=SAMPLE_RATE,
                  channels=1, dtype="float32")
    sd.wait()
    full = full[:, 0]
    print("녹음 종료\n")

    engine = FormantEngine()
    n_per = int(CHUNK_SEC * SAMPLE_RATE)

    print(f"{'chunk':<6}{'RMS':>7}{'iv':>4}{'F0':>5}{'F1':>6}{'F2':>6}{'F3':>6}")
    print("─" * 42)

    f1_list, f2_list = [], []
    for i in range(N_CHUNKS):
        chunk = full[i*n_per:(i+1)*n_per]
        rms = float(np.sqrt(np.mean(chunk**2)))
        if rms < 0.005:
            print(f"{i:<6}{rms:>7.4f}  -    -    -    -    -  (음량↓)")
            continue
        res = engine.extract(chunk - np.mean(chunk),
                             gender=gender,
                             ceilings=FORMANT_CEILINGS)
        iv = "T" if res.get("is_voiced") else "F"
        f0 = res.get("f0")
        f1, f2, f3 = res.get("f1"), res.get("f2"), res.get("f3")
        f0s = f"{f0:.0f}" if f0 else "  - "
        f1s = f"{f1:.0f}" if f1 else "  - "
        f2s = f"{f2:.0f}" if f2 else "  - "
        f3s = f"{f3:.0f}" if f3 else "  - "
        print(f"{i:<6}{rms:>7.4f}{iv:>4}{f0s:>5}{f1s:>6}{f2s:>6}{f3s:>6}")
        if f1 and f2:
            f1_list.append(f1)
            f2_list.append(f2)

    if f1_list:
        print()
        print(f"F1 통계: 평균={np.mean(f1_list):.0f}  "
              f"std={np.std(f1_list):.0f}  "
              f"min-max=[{min(f1_list):.0f}, {max(f1_list):.0f}]")
        print(f"F2 통계: 평균={np.mean(f2_list):.0f}  "
              f"std={np.std(f2_list):.0f}  "
              f"min-max=[{min(f2_list):.0f}, {max(f2_list):.0f}]")
        print()
        f1_range = max(f1_list) - min(f1_list)
        f2_range = max(f2_list) - min(f2_list)
        print(f"안정성: F1 range = {f1_range:.0f}Hz, F2 range = {f2_range:.0f}Hz")
        if f1_range < 100 and f2_range < 200:
            print("  → 매우 안정")
        elif f1_range < 200 and f2_range < 400:
            print("  → 양호")
        else:
            print("  → 불안정 (큰 문제)")


if __name__ == "__main__":
    main()
