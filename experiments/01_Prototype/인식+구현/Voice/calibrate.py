"""
calibrate.py - 캘리브레이션 녹음 스크립트
관람객(화자) 1명에게 7모음 x 3테이크 녹음을 받아 wav로 저장한다.
테이크: 1=편한 피치, 2=조금 높게, 3=조금 낮게
저장 위치: calib_wavs/<이름>/<모음>_<테이크>.wav
"""
import os
import sys
import numpy as np
import soundfile as sf
from feature_utils import (
    SAMPLE_RATE, VOWELS, record, trim_silence
)

RECORD_SEC = 1.5   # 넉넉히 받고 trim_silence로 자름
PITCH_GUIDE = [
    "편한 피치, 마이크와 평소 거리에서",
    "조금 높은 피치, 마이크에 한 뼘 더 가까이",
    "조금 낮은 피치, 마이크에서 한 뼘 더 멀리",
]

def main():
    name = input("화자 이름(영문 추천): ").strip()
    if not name:
        print("이름이 비어있어 종료합니다.")
        sys.exit(1)

    out_dir = os.path.join("calib_wavs", name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== 캘리브레이션 시작: {name} ===")
    print("각 모음을 3번씩 녹음합니다. 엔터를 누르면 녹음이 시작돼요.\n")

    for vowel in VOWELS:
        for take in range(1, 4):
            path = os.path.join(out_dir, f"{vowel}_{take}.wav")
            while True:
                input(f"[{vowel}] 테이크 {take}/3 - {PITCH_GUIDE[take-1]} 발성. 준비되면 엔터 >")
                sig = record(RECORD_SEC)
                trimmed = trim_silence(sig)
                if trimmed is None:
                    print("  소리가 감지되지 않았어요. 다시 시도합니다.")
                    continue
                dur = len(trimmed) / SAMPLE_RATE
                if dur < 0.3:
                    print(f"  발성이 너무 짧아요({dur:.2f}s). 다시 시도합니다.")
                    continue
                sf.write(path, trimmed, SAMPLE_RATE)
                print(f"  저장 완료: {path} ({dur:.2f}s)")
                break
    
    # ---- 문제 모음 추가 테이크: 거리 스위프 ----
    SWEEP_VOWELS = ["오", "우"]
    SWEEP_SEC = 4.0
    print("\n--- 추가 녹음: 오/우 거리 스위프 ---")
    print("발성을 유지하면서 마이크를 천천히 가까이 -> 멀리 움직이세요.\n")
    for vowel in SWEEP_VOWELS:
        path = os.path.join(out_dir, f"{vowel}_4.wav")
        while True:
            input(f"[{vowel}] 스위프 테이크 - {SWEEP_SEC:.0f}초간 유지+이동. 준비되면 엔터 >")
            sig = record(SWEEP_SEC)
            trimmed = trim_silence(sig)
            if trimmed is None or len(trimmed) / SAMPLE_RATE < 2.0:
                print("  발성이 짧거나 감지 안 됨. 다시 시도합니다.")
                continue
            sf.write(path, trimmed, SAMPLE_RATE)
            print(f"  저장 완료: {path}")
            break

    print(f"\n=== 완료! {len(VOWELS)*3}개 파일이 {out_dir} 에 저장됨 ===")

if __name__ == "__main__":
    main()