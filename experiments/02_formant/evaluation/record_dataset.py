"""
evaluation/record_dataset.py — 모음 인식 평가용 데이터셋 녹음 스크립트

한국어 단모음 7개를 각 5회씩(총 35개) 녹음하여 wav 파일로 저장한다.
저장 위치: evaluation/dataset/{모음}_{회차:02d}.wav

사용법:
    cd C:\\Users\\admin\\Desktop\\realtime_formant
    python -m evaluation.record_dataset                # 처음부터
    python -m evaluation.record_dataset --start 에     # '에'부터 시작

기존 포먼트 추출 코드는 일절 import 하지 않으며,
config.SAMPLE_RATE 만 참조한다.
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# 프로젝트 루트의 config.py 만 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE


# ══════════════════════════════════════════
# 설정
# ══════════════════════════════════════════
VOWELS          = ["아", "에", "이", "오", "우", "으", "어"]
TAKES_PER_VOWEL = 5
RECORD_SEC      = 1.5
PAUSE_AFTER_SEC = 0.5
RMS_WARN_THRESH = 0.01

DATASET_DIR = Path(__file__).resolve().parent / "dataset"


# ══════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════

def countdown(seconds: int = 3) -> None:
    for i in range(seconds, 0, -1):
        print(f"  준비... {i}")
        time.sleep(1.0)
    print("  지금 발음하세요!")


def record_once(duration: float = RECORD_SEC) -> np.ndarray:
    """동기 녹음 — float32 mono 1D 배열 반환."""
    n_samples = int(SAMPLE_RATE * duration)
    audio = sd.rec(
        n_samples,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio[:, 0]


def save_wav(path: Path, audio: np.ndarray) -> None:
    """float32 mono WAV 저장 (scipy.io.wavfile)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), SAMPLE_RATE, audio.astype(np.float32))


def confirm_overwrite(path: Path) -> bool:
    """기존 파일이 있으면 덮어쓰기 여부를 사용자에게 묻는다."""
    if not path.exists():
        return True
    ans = input(f"  ⚠ {path.name} 이미 존재. 덮어쓸까요? (y/n): ").strip().lower()
    return ans == "y"


# ══════════════════════════════════════════
# 회차 단위 녹음
# ══════════════════════════════════════════

def record_take(vowel: str, take: int, total_takes: int) -> str:
    """
    한 회차 녹음.

    Returns
    -------
    "saved"   — 저장 성공
    "skipped" — 사용자가 덮어쓰기 거부
    "quit"    — 사용자가 'q' 입력으로 전체 중단
    """
    print(f"\n===== [{vowel}] {take}회차 / {total_takes}회차 =====")

    out_path = DATASET_DIR / f"{vowel}_{take:02d}.wav"
    if not confirm_overwrite(out_path):
        print("  → 건너뜀")
        return "skipped"

    while True:
        countdown(3)
        audio = record_once(RECORD_SEC)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"  녹음 완료. RMS = {rms:.4f}")

        if rms < RMS_WARN_THRESH:
            print("  ⚠ RMS 너무 낮음 (마이크 입력 약함)")

        ans = input("  엔터=저장 / r=다시 / q=종료 : ").strip().lower()
        if ans == "r":
            print("  → 재녹음")
            continue
        if ans == "q":
            return "quit"

        save_wav(out_path, audio)
        print(f"  ✓ 저장: {out_path}")
        time.sleep(PAUSE_AFTER_SEC)
        return "saved"


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="모음 데이터셋 녹음")
    parser.add_argument("--start", default=None,
                        help=f"시작할 모음 ({'/'.join(VOWELS)} 중 하나)")
    parser.add_argument("--start-take", type=int, default=1,
                        help="시작 회차 (1~%d). --start 와 함께 사용." % TAKES_PER_VOWEL)
    args = parser.parse_args()

    start_idx = 0
    start_take = max(1, min(args.start_take, TAKES_PER_VOWEL))
    if args.start:
        if args.start in VOWELS:
            start_idx = VOWELS.index(args.start)
        else:
            print(f"⚠ '{args.start}' 은 모음 목록에 없음. 처음부터 시작합니다.")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"한국어 단모음 데이터셋 녹음")
    print(f"  모음 {len(VOWELS)}개 × {TAKES_PER_VOWEL}회 = 총 {len(VOWELS)*TAKES_PER_VOWEL}개")
    print(f"  저장 위치: {DATASET_DIR}")
    print(f"  샘플레이트: {SAMPLE_RATE} Hz")
    print(f"  녹음 길이: {RECORD_SEC}초/회")
    if args.start:
        print(f"  ▶ 시작: [{VOWELS[start_idx]}] {start_take}회차부터")
    print("=" * 60)
    print()
    print("팁:")
    print("  - 조용한 환경에서 마이크와 ~30cm 거리 유지")
    print("  - 모음을 길게(1.5초 내내) 또렷하게 발음")
    print("  - 자음·숨소리 없이 모음만")
    print()
    input("준비되면 엔터를 누르세요...")

    saved = 0
    skipped = 0
    try:
        for v_i, vowel in enumerate(VOWELS[start_idx:], start=start_idx):
            take_from = start_take if v_i == start_idx else 1
            for take in range(take_from, TAKES_PER_VOWEL + 1):
                status = record_take(vowel, take, TAKES_PER_VOWEL)
                if status == "saved":
                    saved += 1
                elif status == "skipped":
                    skipped += 1
                else:  # quit
                    print("\n사용자가 중단했습니다.")
                    print(f"  저장 {saved}개 / 건너뜀 {skipped}개")
                    return
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt — 중단합니다.")
        print(f"  저장 {saved}개 / 건너뜀 {skipped}개")
        return

    print("\n" + "=" * 60)
    print(f"전체 완료. 저장 {saved}개 / 건너뜀 {skipped}개")
    print(f"  → {DATASET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
