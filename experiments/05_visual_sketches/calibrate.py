"""
calibrate.py — 내 목소리 모음 등록 (개인 캘리브레이션)

7모음을 한 번씩 녹음 → 본인 F1/F2 기준값을 my_vowels.json 에 저장.
저장되면 voice_input 이 자동으로 이 값을 써서 인식이 본인 목소리에 맞춰진다.
(02_formant 의 교훈: 평균값 cal-free 는 54% 천장, 개인 캘리브레이션은 85%로 점프)

실행:  python calibrate.py     (각 모음마다 Enter → 1.6초 녹음)
"""

import sys
import json
import numpy as np
import sounddevice as sd
import parselmouth
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_input import SAMPLE_RATE, VOWELS, _PERSONAL_PATH

REC_SEC = 1.6


def extract_f1f2(audio):
    """녹음 중앙부(30~80%)에서 F1/F2 중앙값."""
    audio = audio - np.mean(audio)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.01:
        return None, rms
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(SAMPLE_RATE))
    fmt = snd.to_formant_burg(time_step=None, max_number_of_formants=5,
                              maximum_formant=5500, window_length=0.025, pre_emphasis_from=50)
    dur = len(audio) / SAMPLE_RATE
    f1s, f2s = [], []
    for frac in np.linspace(0.30, 0.80, 11):
        t = dur * frac
        v1 = fmt.get_value_at_time(1, t)
        v2 = fmt.get_value_at_time(2, t)
        if v1 and not np.isnan(v1) and 100 < v1 < 1500:
            f1s.append(v1)
        if v2 and not np.isnan(v2) and 200 < v2 < 4000:
            f2s.append(v2)
    if len(f1s) < 3 or len(f2s) < 3:
        return None, rms
    return (float(np.median(f1s)), float(np.median(f2s))), rms


def record_one(vowel):
    """한 모음을 녹음·추출. 실패하면 재시도."""
    while True:
        input(f"\n  '{vowel}' 를 길게 발음할 준비가 되면 Enter →")
        print(f"  ● 녹음 {REC_SEC}초... '{vowel}~~~' 일정하게!")
        audio = sd.rec(int(REC_SEC * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        res, rms = extract_f1f2(audio[:, 0])
        if res is None:
            print(f"  ✗ 추출 실패 (소리가 작거나 불안정, rms={rms:.3f}). 다시 해볼게요.")
            continue
        print(f"  ✓ {vowel}:  F1={res[0]:.0f}Hz  F2={res[1]:.0f}Hz")
        ans = input("     이대로 저장? (Enter=예 / r=다시) ").strip().lower()
        if ans != "r":
            return res


def main():
    print("=" * 50)
    print(" 내 목소리 모음 등록 (캘리브레이션)")
    print(" 7모음을 한 번씩 또박또박 길게 발음하세요.")
    print(" 조용한 곳에서, 마이크와 일정한 거리로.")
    print("=" * 50)

    result = {}
    for v in VOWELS:
        f1, f2 = record_one(v)
        result[v] = [round(f1, 1), round(f2, 1)]

    with open(_PERSONAL_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f" 저장 완료 → {_PERSONAL_PATH.name}")
    for v, (f1, f2) in result.items():
        print(f"   {v}:  F1={f1:.0f}  F2={f2:.0f}")
    print(" 이제 sketch 들이 본인 목소리로 인식합니다.")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n중단됨 (저장 안 됨)")
