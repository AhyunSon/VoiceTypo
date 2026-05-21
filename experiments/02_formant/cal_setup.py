"""cal_setup.py — 사용자별 모음 reference 캘리브레이션

7 모음 × 2 takes 녹음 → 본인 F1/F2/F3 평균 저장 → user_refs.pkl
ui_window 시작 시 자동 로드.

핵심: vowel-aware sanity check.
사용자가 [이] 라고 말했으니 측정값이 Yoon ref ± 3σ 범위에 들어와야 함.
범위 밖 (Praat 가 F0 harmonic 잡거나 F2 를 F1 으로 오인 한 경우) → 자동 재녹음.
→ user_refs.pkl 데이터 품질 보장.

실행:
  python cal_setup.py            # 새 cal
  python cal_setup.py --reset    # 기존 삭제 후 재캘리
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import joblib
import sounddevice as sd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SAMPLE_RATE, FORMANT_CEILINGS
from formant_engine import FormantEngine


VOWELS          = ["아", "에", "이", "오", "우", "으", "어"]
RECORD_SEC      = 2.0
TAKES_PER_VOWEL = 2
SANITY_RETRY    = 3        # vowel-aware sanity 실패 시 재녹음 횟수
RMS_MIN         = 0.005

STD_FLOOR = {"F1": 50.0, "F2": 100.0, "F3": 150.0}

CAL_PATH = Path(__file__).resolve().parent / "user_refs.pkl"

# Yoon 2015 reference centers (vowel-aware sanity check 용)
# (F1_μ, F1_σ, F2_μ, F2_σ)
SANITY_REFS = {
    "female": {
        "아": (978, 100, 1397, 175),
        "에": (548, 100, 2125, 185),
        "이": (352,  78, 2787, 250),
        "오": (487,  88,  840, 148),
        "우": (367,  78,  660, 121),
        "으": (435,  90, 1404, 217),
        "어": (671, 109, 1212, 178),
    },
    "male": {
        "아": (831,  88, 1145, 143),
        "에": (466,  88, 1743, 152),
        "이": (299,  68, 2285, 205),
        "오": (414,  78,  689, 121),
        "우": (312,  68,  541, 100),
        "으": (370,  79, 1151, 178),
        "어": (570,  95,  994, 146),
    },
}
SANITY_TOL_SIGMA = 3.0   # μ ± 3σ 안이면 OK


def _countdown(n: int = 3) -> None:
    for i in range(n, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(0.5)


def _record(duration: float = RECORD_SEC) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    a = sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return a[:, 0]


def _is_sane(vowel: str, gender: str, f1: float, f2: float) -> bool:
    """측정값이 Yoon ref μ ± 3σ 안에 있는지."""
    f1_mu, f1_sd, f2_mu, f2_sd = SANITY_REFS[gender][vowel]
    return (abs(f1 - f1_mu) <= SANITY_TOL_SIGMA * f1_sd and
            abs(f2 - f2_mu) <= SANITY_TOL_SIGMA * f2_sd)


def _record_take(engine, vowel: str, gender: str,
                 take_num: int, total: int) -> tuple | None:
    """1 take 녹음. sanity check 통과까지 자동 재시도."""
    for attempt in range(SANITY_RETRY):
        print(f"  take {take_num}/{total}: 준비... ", end="", flush=True)
        _countdown(3)
        print("녹음... ", end="", flush=True)
        audio = _record(RECORD_SEC)

        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < RMS_MIN:
            print("음량 낮음 — 다시")
            continue

        res = engine.extract(audio, gender=gender, ceilings=FORMANT_CEILINGS)
        f1, f2, f3 = res.get("f1"), res.get("f2"), res.get("f3")
        if f1 is None or f2 is None or f3 is None:
            print("포먼트 실패 — 다시")
            continue

        if not _is_sane(vowel, gender, f1, f2):
            print(f"비정상 (F1={f1:.0f}, F2={f2:.0f}) — Praat 오류, 다시")
            continue

        print(f"F1={f1:.0f}  F2={f2:.0f}  F3={f3:.0f}")
        return (f1, f2, f3)

    return None


def calibrate(gender: str) -> dict:
    """7 모음 × 2 takes → {vowel: (F1_μ, F1_σ, F2_μ, F2_σ, F3_μ, F3_σ)}."""
    engine = FormantEngine()
    refs = {}

    for v in VOWELS:
        print(f"\n[{v}] {TAKES_PER_VOWEL}번 발음")
        takes = []
        for t in range(TAKES_PER_VOWEL):
            result = _record_take(engine, v, gender, t + 1, TAKES_PER_VOWEL)
            if result:
                takes.append(result)

        if len(takes) < 2:
            print(f"  ⚠ {v}: 유효 take 부족 → 학계 _REFS 사용")
            continue

        arr = np.array(takes)
        f1, f2, f3 = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
        sd1 = max(arr[:, 0].std(), STD_FLOOR["F1"])
        sd2 = max(arr[:, 1].std(), STD_FLOOR["F2"])
        sd3 = max(arr[:, 2].std(), STD_FLOOR["F3"])
        refs[v] = (float(f1), float(sd1),
                   float(f2), float(sd2),
                   float(f3), float(sd3))
        print(f"  → F1={f1:.0f}±{sd1:.0f}  F2={f2:.0f}±{sd2:.0f}  F3={f3:.0f}±{sd3:.0f}")

    return refs


def run_cal(gender: str = "female") -> bool:
    """main.py 호출용. True = 성공."""
    print("=" * 50)
    print(f"캘리브레이션 — 7 모음 × {TAKES_PER_VOWEL}번 ({gender})")
    print("=" * 50)
    try:
        refs = calibrate(gender)
    except KeyboardInterrupt:
        print("\n취소.")
        return False

    if len(refs) < 5:
        print(f"\n⚠ {len(refs)}/7 모음만 cal — 저장 안 함")
        return False

    joblib.dump(refs, str(CAL_PATH))
    print(f"\n💾 저장: {CAL_PATH.name} ({len(refs)}/7 모음)")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true",
                   help="기존 user_refs 삭제 후 재캘리")
    args = p.parse_args()

    if args.reset and CAL_PATH.exists():
        CAL_PATH.unlink()
        print(f"삭제: {CAL_PATH}")

    if CAL_PATH.exists() and not args.reset:
        print(f"⚠ 기존 cal 있음. 덮어쓸까요? (y/n)")
        if input("> ").strip().lower() != "y":
            return

    print("성별? male / female / 엔터로 female:")
    ans = input("> ").strip().lower()
    gender = "male" if ans == "male" else "female"

    if run_cal(gender):
        print("python main.py 실행 시 자동 적용.")


if __name__ == "__main__":
    main()
