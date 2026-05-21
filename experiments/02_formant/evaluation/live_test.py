"""
evaluation/live_test.py — 라이브 마이크 인식 테스트

v5 CV 결과 (오프라인 80%) 검증:
  본인이 일관 발음 시도하면 90% 도달 가능?
  (오프라인 80% 천장은 본인 takes 간 발음 변동성 — 우 F2 768~1656 등)

흐름:
  1. 캘리브레이션: 7 모음 × 각 2초 (총 ~17~30초, 안내 카운트다운 포함)
  2. GMM 학습 (Layer 1 + 4: k≤2 BIC strict + 4D bandwidth)
  3. 테스트: 14 라운드 (각 모음 2번, 무작위 순서)
  4. 결과: 정확도 + 모음별 + confusion

옵션:
  --replay  저장된 cal 사용 (cal 단계 건너뜀)
  --rounds N  테스트 라운드 수 (기본 14)

실행:
  cd /c/Users/admin/Desktop/realtime_formant
  python -m evaluation.live_test
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
from sklearn.mixture import GaussianMixture

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
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]
RMS_MIN    = 0.005

HERE       = Path(__file__).resolve().parent
RESULTS    = HERE / "results"
CAL_CACHE  = RESULTS / "live_cal.npz"


# ══════════════════════════════════════════
# 녹음 / 추출 (v5 와 동일)
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


def _formant(audio):
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64),
                            sampling_frequency=float(SAMPLE_RATE))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    return fmt, audio.shape[0] / SAMPLE_RATE


def extract_full(audio):
    """5 시간점 × 6 (F1/F2/F3/B1/B2/B3) 추출."""
    fmt, dur = _formant(audio)
    rows = []
    for p in SAMPLE_POS:
        t = dur * p
        row = []
        for n in [1, 2, 3]:
            v = fmt.get_value_at_time(n, t)
            row.append(None if (v is None or np.isnan(v)) else float(v))
        for n in [1, 2, 3]:
            v = fmt.get_bandwidth_at_time(n, t)
            row.append(None if (v is None or np.isnan(v)) else float(v))
        rows.append(tuple(row))
    return rows


def _bark(f):
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def to_feature(arr):
    """(N, 6) → (N, 4): Bark F1/F2/F3 + log(B1/F1)."""
    return np.column_stack([
        _bark(arr[:, 0]), _bark(arr[:, 1]), _bark(arr[:, 2]),
        np.log(np.clip(arr[:, 3] / arr[:, 0], 0.05, 5.0)),
    ])


# ══════════════════════════════════════════
# GMM (v5 와 동일)
# ══════════════════════════════════════════

def fit_gmms(samples_dict):
    gmms, chosen_k = {}, {}
    for v, pts in samples_dict.items():
        arr = np.asarray(pts, dtype=float)
        feat = to_feature(arr)
        best_k, best_bic, best_g = 1, float("inf"), None
        for k in [1, 2]:
            if len(feat) < k * 5:
                continue
            try:
                g = GaussianMixture(
                    n_components=k, covariance_type="diag",
                    reg_covar=1e-3, random_state=0, max_iter=200, n_init=3,
                )
                g.fit(feat)
                bic = g.bic(feat)
                if best_g is None or bic < best_bic - 4:
                    best_bic, best_k, best_g = bic, k, g
            except Exception:
                continue
        gmms[v] = best_g
        chosen_k[v] = best_k
    return gmms, chosen_k


def classify(features, gmms):
    pt = features.reshape(1, -1)
    best_v, best_lp, second = "?", float("-inf"), float("-inf")
    for v, g in gmms.items():
        try:
            lp = float(g.score(pt))
        except Exception:
            continue
        if lp > best_lp:
            second  = best_lp
            best_lp = lp
            best_v  = v
        elif lp > second:
            second = lp
    if best_v == "?" or second == float("-inf"):
        return best_v, 0.0
    return best_v, float(max(0.0, min(1.0, (best_lp - second) / 5.0)))


def vote(audio, gmms):
    samples = extract_full(audio)
    votes = defaultdict(float)
    nv = 0
    sample_log = []
    for full in samples:
        f1, f2, f3, b1, b2, b3 = full
        if any(x is None for x in [f1, f2, f3, b1, b2, b3]):
            continue
        arr = np.array([[f1, f2, f3, b1, b2, b3]])
        feat = to_feature(arr)[0]
        p, c = classify(feat, gmms)
        sample_log.append((f1, f2, f3, p, c))
        if p == "?" or c <= 0:
            continue
        votes[p] += c
        nv += 1
    if not votes:
        return "?", 0.0, 0, sample_log
    best = max(votes, key=votes.get)
    return best, votes[best], nv, sample_log


# ══════════════════════════════════════════
# 캘리브레이션
# ══════════════════════════════════════════

def show_vowel_box(vowel: str, label: str = ""):
    print()
    print("  ┌─────────────────────┐")
    print(f"  │         [{vowel}]         │ {label}")
    print("  └─────────────────────┘")


def calibrate(retry_max: int = 2) -> tuple:
    """7 모음 발음 → samples dict.

    각 모음:
      준비 카운트 → 2 초 녹음 → RMS / 샘플 수 검증 → 통과 시 다음
      실패 시 최대 retry_max 회 재시도.
    """
    print("=" * 60)
    print("캘리브레이션")
    print("=" * 60)
    print("각 모음을 자연스럽게 약 2초간 발음해주세요.")
    print("발음 일정한 음성으로 — 너무 짧거나 떨림 적게.")
    print()

    samples = defaultdict(list)
    for v in VOWELS:
        for attempt in range(retry_max + 1):
            label = f"(시도 {attempt+1}/{retry_max+1})" if attempt > 0 else ""
            show_vowel_box(v, label)
            print(f"  준비... ", end="")
            countdown(3)
            print("  ▶ 녹음 중 (2초)...")
            audio = record(RECORD_SEC)
            print("  ◼ 종료")

            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < RMS_MIN:
                print(f"  ✗ 음량 낮음 (RMS={rms:.4f}). 다시 시도.")
                continue

            rows = extract_full(audio)
            valid = [r for r in rows
                     if all(x is not None for x in r)]
            if len(valid) < 3:
                print(f"  ✗ 포먼트 추출 부족 ({len(valid)}/5). 다시 시도.")
                continue

            for r in valid:
                samples[v].append(r)
            f1s = [r[0] for r in valid]
            f2s = [r[1] for r in valid]
            print(f"  ✓ {len(valid)} 샘플 수집  "
                  f"F1≈{np.median(f1s):.0f} F2≈{np.median(f2s):.0f}")
            break
        else:
            raise RuntimeError(f"캘리브레이션 실패: {v}")

    print()
    print("✓ 캘리브레이션 완료")
    return dict(samples)


def save_cal(samples: dict, path: Path):
    arrays = {v: np.asarray(s, dtype=float) for v, s in samples.items()}
    np.savez(path, **arrays)
    print(f"  💾 저장: {path}")


def load_cal(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"저장된 cal 없음: {path}")
    data = np.load(path)
    return {v: data[v].tolist() for v in data.files}


# ══════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════

def run_test_round(gmms, target: str, idx: int, total: int) -> dict:
    show_vowel_box(target, f"[{idx}/{total}]")
    print(f"  준비... ", end="")
    countdown(3)
    print("  ▶ 녹음 중 (2초)...")
    audio = record(RECORD_SEC)
    print("  ◼ 종료")

    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < RMS_MIN:
        print(f"  ⚠ 음량 낮음 (RMS={rms:.4f}) — 무효 처리")
        return dict(target=target, pred="?", conf=0.0, voters=0,
                    valid=False, rms=rms)

    pred, conf, nv, log = vote(audio, gmms)
    valid = pred != "?"
    mark = "✓" if pred == target else "✗"
    print(f"  → 인식: {pred}  (목표: {target})  conf={conf:.2f}  "
          f"voters={nv}  {mark}")
    return dict(target=target, pred=pred, conf=conf, voters=nv,
                valid=valid, rms=rms)


def run_tests(gmms, n_rounds: int = 14, seed: int = None) -> list:
    print()
    print("=" * 60)
    print(f"테스트 단계 — {n_rounds} 라운드 (각 모음 ~{n_rounds//7}회)")
    print("=" * 60)
    print()

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
        r = run_test_round(gmms, target, i, len(order))
        results.append(r)
    return results


# ══════════════════════════════════════════
# 보고
# ══════════════════════════════════════════

def report(results: list, chosen_k: dict, save_path: Path = None):
    print()
    print("=" * 60)
    print("결과")
    print("=" * 60)

    valid = [r for r in results if r["valid"]]
    correct = sum(1 for r in valid if r["pred"] == r["target"])
    total = len(valid)
    invalid = len(results) - total

    print(f"  유효 라운드:    {total}/{len(results)}")
    print(f"  정답:          {correct}/{total} = "
          f"{correct/total*100:.1f}%" if total else "측정 불가")
    if invalid:
        print(f"  무효(저음량 등): {invalid}")
    print()

    by_v = defaultdict(lambda: {"correct": 0, "total": 0, "errors": Counter()})
    for r in valid:
        by_v[r["target"]]["total"] += 1
        if r["pred"] == r["target"]:
            by_v[r["target"]]["correct"] += 1
        else:
            by_v[r["target"]]["errors"][r["pred"]] += 1

    print("모음별:")
    print(f"  {'모음':<4s} {'정확도':>10s}  오답")
    print(f"  {'-'*4} {'-'*10}  {'-'*30}")
    for v in VOWELS:
        d = by_v[v]
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']}"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        print(f"  {v:<4s} {acc:>10s}  {err}")
    print()

    print(f"GMM k 선택: " + "  ".join(f"{v}={k}" for v, k in chosen_k.items()))
    print()

    # 판정
    if total == 0:
        verdict = "측정 불가 — 라운드 모두 무효"
    elif correct / total >= 0.90:
        verdict = "✓ 90% 도달! 일관 발음 시 천장 입증."
    elif correct / total >= 0.85:
        verdict = "○ 85%+ 도달. 90% 근접."
    elif correct / total >= 0.80:
        verdict = "△ 오프라인 CV 와 비슷한 수준."
    else:
        verdict = "✗ 오프라인 CV 보다 낮음 — 라이브 환경 차이 있음."
    print(f"판정: {verdict}")
    print()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump({
                "results": results,
                "chosen_k": chosen_k,
                "summary": {
                    "valid": total, "correct": correct,
                    "accuracy": (correct / total * 100.0) if total else 0.0,
                    "verdict": verdict,
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"  💾 저장: {save_path}")


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", action="store_true",
                   help="저장된 cal 재사용 (cal 단계 건너뜀)")
    p.add_argument("--rounds", type=int, default=14,
                   help="테스트 라운드 수 (기본 14)")
    p.add_argument("--seed", type=int, default=None,
                   help="라운드 순서 시드 (재현용)")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    print()
    print("┌─────────────────────────────────────────────┐")
    print("│  realtime_formant — 라이브 인식 테스트     │")
    print("│  v5 CV 80% 의 라이브 검증                   │")
    print("└─────────────────────────────────────────────┘")
    print()

    # 캘리브레이션
    if args.replay and CAL_CACHE.exists():
        print(f"  📂 저장된 cal 로드: {CAL_CACHE}")
        samples = load_cal(CAL_CACHE)
    else:
        samples = calibrate()
        save_cal(samples, CAL_CACHE)

    # GMM
    gmms, chosen_k = fit_gmms(samples)
    print()
    print("─" * 60)
    print("GMM 학습 완료")
    print("─" * 60)
    print(f"  k 선택: " + "  ".join(f"{v}={k}" for v, k in chosen_k.items()))
    print()

    input("  엔터를 누르면 테스트 시작...")

    # 테스트
    results = run_tests(gmms, n_rounds=args.rounds, seed=args.seed)

    # 보고
    report(results, chosen_k,
           save_path=RESULTS / "live_test_result.json")


if __name__ == "__main__":
    main()
