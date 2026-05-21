"""
evaluation/live_test_v2.py — 라이브 테스트 v2 (Lobanov + LDA + full feature)

v1 (cal+GMM) 대비 변경:
  - GMM → LDA (Phase B 검증된 분류기)
  - 3D feature → full 9D (3 시점 × F1/F2/F3, Hillenbrand 95.4% 의 best)
  - Bark Mahalanobis 거리 → Lobanov z-score + LDA
  - BIC/k 선택 X → 단순 LDA

흐름:
  1. 캘리브레이션: 7 모음 × 2초 (각 모음 5 시간점 측정 = 모음당 5 sample)
  2. 학습 데이터: 7 × 5 = 35 sample (full 9D)
  3. Lobanov 정규화 (화자 자기 mean/std)
  4. LDA fit (35 sample, 7-class, 9 feature)
  5. 테스트: 14 라운드, 각 모음 2번 무작위
     라이브 청크 → full feature → 정규화 → LDA → 결과

비교:
  v1 (cal+GMM): 본인 5-fold CV 80%, 라이브 미흡
  v2 (Lobanov+LDA+full 9D): 본 측정값
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import parselmouth
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE


VOWELS     = ["아", "에", "이", "오", "우", "으", "어"]
RECORD_SEC = 2.0
# Hillenbrand 의 full feature = 3 시점 × F1/F2/F3 = 9D
# 5 시간점 보다 3 시점이 학계 표준 (20%, 50%, 80%)
SAMPLE_POS = [0.20, 0.50, 0.80]
RMS_MIN    = 0.005

HERE       = Path(__file__).resolve().parent
RESULTS    = HERE / "results"
CAL_CACHE  = RESULTS / "live_v2_cal.npz"


# ══════════════════════════════════════════
# 녹음 / 추출
# ══════════════════════════════════════════

def countdown(n: int = 3) -> None:
    for i in range(n, 0, -1):
        print(f"   {i}...", end="\r", flush=True)
        time.sleep(0.7)


def record(duration: float = RECORD_SEC) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    a = sd.rec(n, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return a[:, 0]


def extract_full(audio: np.ndarray) -> np.ndarray:
    """full feature 9D: F1/F2/F3 at 3 시점 (20%, 50%, 80%).

    Returns: (9,) ndarray. 측정 실패 시 np.nan 포함.
    """
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64),
                            sampling_frequency=float(SAMPLE_RATE))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    dur = audio.shape[0] / SAMPLE_RATE
    feat = []
    for p in SAMPLE_POS:
        t = dur * p
        for n in [1, 2, 3]:
            v = fmt.get_value_at_time(n, t)
            feat.append(float(v) if (v is not None
                                     and not np.isnan(v)) else np.nan)
    return np.array(feat)


# ══════════════════════════════════════════
# 캘리브레이션
# ══════════════════════════════════════════

def show_vowel_box(vowel: str, label: str = ""):
    print()
    print("  ┌─────────────────────┐")
    print(f"  │         [{vowel}]         │ {label}")
    print("  └─────────────────────┘")


def calibrate(retry_max: int = 2, takes_per_vowel: int = 2) -> tuple:
    """7 모음 × takes_per_vowel 회 발음 → (X, y, audio_chunks).

    LDA 가 N(sample) > K(클래스=7) 필요 → 모음당 최소 2 발화.

    각 발화:
      준비 카운트 → 2초 녹음 → 1 9D feature (3 시점 × F1/F2/F3)
      모음당 takes_per_vowel sample 누적.
    """
    print("=" * 60)
    print("캘리브레이션 (Lobanov + LDA + full feature 9D)")
    print("=" * 60)
    print(f"각 모음을 {takes_per_vowel}번씩 발음합니다.")
    print(f"총 7 × {takes_per_vowel} = {7 * takes_per_vowel} 샘플 수집.")
    print("각 발화 약 2초, 자연스럽게.")
    print()

    X, y, audios = [], [], []
    for v in VOWELS:
        for take in range(1, takes_per_vowel + 1):
            for attempt in range(retry_max + 1):
                label = f"(take {take}/{takes_per_vowel})"
                if attempt > 0:
                    label += f" 시도 {attempt+1}"
                show_vowel_box(v, label)
                print(f"  준비... ", end="")
                countdown(3)
                print("  ▶ 녹음 중 (2초)...")
                audio = record(RECORD_SEC)
                print("  ◼ 종료")

                rms = float(np.sqrt(np.mean(audio**2)))
                if rms < RMS_MIN:
                    print(f"  ✗ 음량 낮음 (RMS={rms:.4f}). 다시.")
                    continue

                feat = extract_full(audio)
                if np.any(np.isnan(feat)):
                    print(f"  ✗ 포먼트 추출 실패. 다시.")
                    continue

                X.append(feat)
                y.append(v)
                audios.append(audio)
                print(f"  ✓ F1=[{feat[0]:.0f},{feat[3]:.0f},{feat[6]:.0f}] "
                      f"F2=[{feat[1]:.0f},{feat[4]:.0f},{feat[7]:.0f}]")
                break
            else:
                raise RuntimeError(f"캘리브레이션 실패: {v} take {take}")

    print()
    print(f"✓ 캘리브레이션 완료 — {len(X)} sample × {X[0].shape[0]}D")
    return np.array(X), np.array(y), audios


def save_cal(X, y, path: Path):
    np.savez(path, X=X, y=y)
    print(f"  💾 저장: {path}")


def load_cal(path: Path) -> tuple:
    data = np.load(path)
    return data["X"], data["y"]


# ══════════════════════════════════════════
# 학습 (Lobanov + LDA)
# ══════════════════════════════════════════

def train(X: np.ndarray, y: np.ndarray) -> tuple:
    """Lobanov 정규화 + LDA fit.

    Returns: (lda, (mean, std)) — 분류 시 정규화에 재사용.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std = np.maximum(std, 1e-6)
    X_norm = (X - mean) / std

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_norm, y)
    return lda, (mean, std)


# ══════════════════════════════════════════
# 분류 (라이브)
# ══════════════════════════════════════════

def classify_live(audio: np.ndarray, lda, stats: tuple) -> tuple:
    """라이브 청크 → full feature → 정규화 → LDA 분류.

    Returns: (vowel, confidence)
    """
    feat = extract_full(audio)
    if np.any(np.isnan(feat)):
        return "?", 0.0
    mean, std = stats
    feat_norm = ((feat - mean) / std).reshape(1, -1)
    pred = lda.predict(feat_norm)[0]
    proba = lda.predict_proba(feat_norm)[0]
    conf = float(proba.max())
    return pred, conf


# ══════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════

def run_test_round(lda, stats, target: str, idx: int, total: int) -> dict:
    show_vowel_box(target, f"[{idx}/{total}]")
    print(f"  준비... ", end="")
    countdown(3)
    print("  ▶ 녹음 중 (2초)...")
    audio = record(RECORD_SEC)
    print("  ◼ 종료")

    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < RMS_MIN:
        print(f"  ⚠ 음량 낮음 (RMS={rms:.4f}) — 무효")
        return dict(target=target, pred="?", conf=0.0, valid=False)

    pred, conf = classify_live(audio, lda, stats)
    valid = pred != "?"
    mark = "✓" if pred == target else "✗"
    print(f"  → 인식: {pred}  (목표: {target})  conf={conf:.2f}  {mark}")
    return dict(target=target, pred=pred, conf=conf, valid=valid)


def run_tests(lda, stats, n_rounds: int = 14, seed=None) -> list:
    print()
    print("=" * 60)
    print(f"테스트 — {n_rounds} 라운드")
    print("=" * 60)

    per_v = max(1, n_rounds // len(VOWELS))
    order = []
    for v in VOWELS:
        order.extend([v] * per_v)
    extras = n_rounds - len(order)
    if extras > 0:
        order.extend(random.sample(VOWELS, extras))
    rng = random.Random(seed)
    rng.shuffle(order)

    results = []
    for i, target in enumerate(order, 1):
        r = run_test_round(lda, stats, target, i, len(order))
        results.append(r)
    return results


# ══════════════════════════════════════════
# 보고
# ══════════════════════════════════════════

def report(results: list, save_path: Path = None):
    print()
    print("=" * 60)
    print("결과")
    print("=" * 60)

    valid = [r for r in results if r["valid"]]
    correct = sum(1 for r in valid if r["pred"] == r["target"])
    total = len(valid)
    invalid = len(results) - total

    if total > 0:
        print(f"  유효: {total}/{len(results)}")
        print(f"  정답: {correct}/{total} = {correct/total*100:.1f}%")
    if invalid:
        print(f"  무효: {invalid}")
    print()

    by_v = defaultdict(lambda: {"correct": 0, "total": 0,
                                "errors": Counter()})
    for r in valid:
        by_v[r["target"]]["total"] += 1
        if r["pred"] == r["target"]:
            by_v[r["target"]]["correct"] += 1
        else:
            by_v[r["target"]]["errors"][r["pred"]] += 1

    print("모음별:")
    for v in VOWELS:
        d = by_v[v]
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']}"
        err = (", ".join(f"{p}×{n}" for p, n in d["errors"].most_common())
               or "—")
        print(f"  {v}: {acc:>5s}  {err}")
    print()

    if total > 0:
        if correct / total >= 0.90:
            verdict = "✓ 90%+ 도달!"
        elif correct / total >= 0.85:
            verdict = "○ 85~90%, 매우 좋음"
        elif correct / total >= 0.75:
            verdict = "△ 75~85%, 향상 있음"
        else:
            verdict = "✗ 75% 미만"
        print(f"판정: {verdict}\n")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
        print(f"  💾 저장: {save_path}")


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", action="store_true",
                   help="저장된 cal 재사용")
    p.add_argument("--rounds", type=int, default=14)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    print()
    print("┌──────────────────────────────────────────────────┐")
    print("│  라이브 테스트 v2 (Lobanov + LDA + full 9D)        │")
    print("│  Phase B 핵심 파이프라인 적용                      │")
    print("└──────────────────────────────────────────────────┘")
    print()

    if args.replay and CAL_CACHE.exists():
        print(f"  📂 저장된 cal 로드: {CAL_CACHE}")
        X, y = load_cal(CAL_CACHE)
    else:
        X, y, _ = calibrate()
        save_cal(X, y, CAL_CACHE)

    print()
    print("─" * 60)
    print("학습 (Lobanov + LDA)")
    print("─" * 60)
    lda, stats = train(X, y)
    mean, std = stats
    print(f"  화자 stats: mean={mean[:3].astype(int)}... "
          f"std={std[:3].astype(int)}...")
    print(f"  LDA classes: {list(lda.classes_)}")
    print(f"  학습 sample: {len(X)}, feature dim: {X.shape[1]}")
    print()

    print("  3초 후 테스트 시작...")
    time.sleep(3)

    results = run_tests(lda, stats, n_rounds=args.rounds, seed=args.seed)
    report(results, save_path=RESULTS / "live_test_v2_result.json")


if __name__ == "__main__":
    main()
