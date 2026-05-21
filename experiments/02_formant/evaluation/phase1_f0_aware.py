"""
evaluation/phase1_f0_aware.py — Phase 1: F0-aware F1 검증

baseline_simple 의 추출 로직에 보편 물리 제약 추가:
  Praat 으로 5개 formant 추출 → F0 의 1.3배 미만 reject →
  남은 것 중 주파수 순서대로 F1/F2/F3 할당.

이론 근거: F1 은 정의상 F0 보다 높음 (F0 는 기본 주파수,
포먼트는 그 위의 envelope peak). 화자별 튜닝 아닌 보편 제약.

본인 어 케이스 진단 (baseline_simple):
  F0 = 236 Hz
  측정 어 F1 = 258~356 Hz   ← F0 의 harmonic 으로 오인
  학계 어 F1 = 629 Hz
  → Praat LPC 가 F0 근처 spurious peak 을 F1 으로 잡음.

비교: baseline_simple (54.3%) vs Phase 1.

기존 코드 수정 없음. 새 평가 스크립트만 추가.

실행:
    cd /c/Users/admin/Desktop/realtime_formant
    python -m evaluation.phase1_f0_aware
"""

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from scipy.io import wavfile
import parselmouth
import pyworld as pw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE
from vowel_classifier import classify_vowel


# ══════════════════════════════════════════
# 상수
# ══════════════════════════════════════════
VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"

F1_F0_RATIO = 1.3   # F1 은 F0 의 최소 1.3배 이상
N_FORMANTS  = 5     # Praat 에서 추출할 formant 후보 수


# ══════════════════════════════════════════
# 데이터
# ══════════════════════════════════════════

def load_wav(path: Path) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    return data


def collect_files() -> list:
    items = []
    for f in sorted(DATASET.glob("*.wav")):
        stem = f.stem
        if "_" not in stem:
            continue
        v, _, t = stem.partition("_")
        if v in VOWELS and t.isdigit():
            items.append((v, int(t), f))
    items.sort(key=lambda x: (VOWELS.index(x[0]), x[1]))
    return items


# ══════════════════════════════════════════
# F0 / Formants
# ══════════════════════════════════════════

def compute_f0(audio: np.ndarray) -> float | None:
    """pyworld DIO + StoneMask 로 F0 평균값."""
    x = (audio - np.mean(audio)).astype(np.float64)
    f0_arr, t_arr = pw.dio(
        x, float(SAMPLE_RATE),
        f0_floor=50.0, f0_ceil=500.0,
        frame_period=10.0,
    )
    f0_arr = pw.stonemask(x, f0_arr, t_arr, float(SAMPLE_RATE))
    voiced = f0_arr[f0_arr > 0]
    if len(voiced) == 0:
        return None
    return float(np.mean(voiced))


def extract_baseline(audio: np.ndarray) -> tuple:
    """기존 baseline_simple 추출 — 비교용."""
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(
        audio.astype(np.float64),
        sampling_frequency=float(SAMPLE_RATE),
    )
    fmt = snd.to_formant_burg(
        time_step=None,
        max_number_of_formants=N_FORMANTS,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50,
    )
    t = audio.shape[0] / SAMPLE_RATE / 2

    def _get(n):
        v = fmt.get_value_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)

    return _get(1), _get(2), _get(3)


def extract_phase1(audio: np.ndarray, f0: float | None) -> tuple:
    """Phase 1: F0-aware filter.
    1) Praat 으로 N_FORMANTS 개 candidate 추출
    2) F0 × F1_F0_RATIO 미만 reject
    3) 남은 candidate 를 주파수 순으로 F1/F2/F3 할당
    """
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(
        audio.astype(np.float64),
        sampling_frequency=float(SAMPLE_RATE),
    )
    fmt = snd.to_formant_burg(
        time_step=None,
        max_number_of_formants=N_FORMANTS,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50,
    )
    t = audio.shape[0] / SAMPLE_RATE / 2

    candidates = []
    for n in range(1, N_FORMANTS + 1):
        v = fmt.get_value_at_time(n, t)
        if v is not None and not np.isnan(v) and v > 0:
            candidates.append(float(v))

    if not candidates:
        return None, None, None

    # F0 모르면 그냥 사용
    if f0 is None or f0 <= 0:
        candidates.sort()
        f1 = candidates[0] if len(candidates) >= 1 else None
        f2 = candidates[1] if len(candidates) >= 2 else None
        f3 = candidates[2] if len(candidates) >= 3 else None
        return f1, f2, f3

    # F0 의 1.3 배 미만 reject (F1 < F0 위반 방지)
    threshold = F1_F0_RATIO * f0
    valid = sorted([c for c in candidates if c >= threshold])

    f1 = valid[0] if len(valid) >= 1 else None
    f2 = valid[1] if len(valid) >= 2 else None
    f3 = valid[2] if len(valid) >= 3 else None
    return f1, f2, f3


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = collect_files()

    print("=" * 75)
    print("Phase 1: F0-aware F1 validation")
    print("=" * 75)
    print(f"  파일: {len(files)}개")
    print(f"  제약: F1 ≥ {F1_F0_RATIO} × F0  (LPC 후보 {N_FORMANTS} 개에서 reject)")
    print()

    results = []
    for i, (true_v, take, path) in enumerate(files, 1):
        audio = load_wav(path)
        f0 = compute_f0(audio)

        # baseline 추출
        f1_b, f2_b, f3_b = extract_baseline(audio)
        pred_b, _ = classify_vowel(f1_b, f2_b, "female", f3=f3_b, scale=1.0)

        # phase 1 추출
        f1_p, f2_p, f3_p = extract_phase1(audio, f0)
        pred_p, _ = classify_vowel(f1_p, f2_p, "female", f3=f3_p, scale=1.0)

        # F1 변화 마크
        change_mark = ""
        if f1_b is not None and f1_p is not None and abs(f1_b - f1_p) > 5:
            change_mark = " 🔄"

        mark_b = "✓" if pred_b == true_v else "✗"
        mark_p = "✓" if pred_p == true_v else "✗"

        print(f"  [{i:2d}/{len(files)}] {path.name:<20s}  "
              f"F0={f0 or 0:5.0f}  "
              f"base: F1={f1_b or 0:4.0f} → {pred_b}{mark_b}  "
              f"phase1: F1={f1_p or 0:4.0f} → {pred_p}{mark_p}{change_mark}")

        results.append(dict(
            true=true_v, path=path.name, f0=f0,
            base_f1=f1_b, base_f2=f2_b, base_f3=f3_b, base_pred=pred_b,
            p1_f1=f1_p, p1_f2=f2_p, p1_f3=f3_p, p1_pred=pred_p,
        ))

    # 요약
    base_correct = sum(1 for r in results if r["base_pred"] == r["true"])
    p1_correct   = sum(1 for r in results if r["p1_pred"]   == r["true"])
    total = len(results)

    print()
    print("=" * 75)
    print("결과 비교")
    print("=" * 75)
    print(f"  baseline (F0 검증 없음): {base_correct}/{total} = {base_correct/total*100:.1f}%")
    print(f"  Phase 1 (F0-aware):     {p1_correct}/{total} = {p1_correct/total*100:.1f}%")
    delta = (p1_correct - base_correct) / total * 100
    print(f"  변화:                   {delta:+.1f} %p")
    print()

    print(f"  {'모음':<4} | {'baseline':<13} | {'phase 1':<13} | 변화")
    print(f"  -----+---------------+---------------+------")
    per_vowel = {}
    for v in VOWELS:
        b_c = sum(1 for r in results if r["true"] == v and r["base_pred"] == v)
        b_t = sum(1 for r in results if r["true"] == v)
        p_c = sum(1 for r in results if r["true"] == v and r["p1_pred"]   == v)
        delta_v = p_c - b_c
        per_vowel[v] = (b_c, p_c, b_t)
        b_str = f"{b_c}/{b_t} ({b_c/b_t*100:3.0f}%)"
        p_str = f"{p_c}/{b_t} ({p_c/b_t*100:3.0f}%)"
        sign = "+" if delta_v >= 0 else ""
        print(f"  {v:<4} | {b_str:<13} | {p_str:<13} | {sign}{delta_v}")

    n_changed = sum(1 for r in results
                    if r["base_f1"] is not None and r["p1_f1"] is not None
                    and abs(r["base_f1"] - r["p1_f1"]) > 5)
    print()
    print(f"F1 값 변경된 파일: {n_changed}/{total}")

    write_md_report(results, base_correct, p1_correct, total, per_vowel)
    print(f"\n산출물: {RESULTS / 'phase1_f0_aware.md'}")


def write_md_report(results, base_correct, p1_correct, total, per_vowel):
    L = ["# Phase 1: F0-aware F1 검증 결과", ""]
    L.append("## 가설")
    L.append("")
    L.append("baseline_simple 의 어 인식률 20% 의 진짜 원인:")
    L.append("- 본인 F0 ≈ 236 Hz")
    L.append("- 어 측정 F1 = 258~356 Hz (F0 와 거의 같음)")
    L.append("- 학계 어 F1 = 629 Hz")
    L.append("- → **Praat LPC 가 F0 근처 spurious peak 을 F1 으로 오인**")
    L.append("")
    L.append("## 수정")
    L.append("")
    L.append(f"- Praat 으로 {N_FORMANTS} 개 formant 후보 추출")
    L.append(f"- F0 × {F1_F0_RATIO} 미만인 candidate reject (보편 물리 제약: F1 > F0)")
    L.append("- 남은 것 중 주파수 순으로 F1/F2/F3 할당")
    L.append("")
    L.append("**화자별 튜닝 아님** — 모든 화자에 동일하게 적용되는 물리 제약.")
    L.append("")
    L.append("## 전체 결과")
    L.append("")
    L.append("| 시스템 | 정확도 |")
    L.append("|---|---:|")
    L.append(f"| baseline_simple | {base_correct}/{total} = {base_correct/total*100:.1f}% |")
    L.append(f"| **Phase 1 (F0-aware)** | **{p1_correct}/{total} = {p1_correct/total*100:.1f}%** |")
    delta = (p1_correct - base_correct) / total * 100
    L.append(f"| 변화 | **{delta:+.1f} %p** |")
    L.append("")

    L.append("## 모음별 비교")
    L.append("")
    L.append("| 모음 | baseline | Phase 1 | 변화 |")
    L.append("|---|---|---|---:|")
    for v in VOWELS:
        b_c, p_c, b_t = per_vowel[v]
        delta_v = p_c - b_c
        sign = "+" if delta_v >= 0 else ""
        L.append(f"| {v} | {b_c}/{b_t} ({b_c/b_t*100:.0f}%) "
                 f"| {p_c}/{b_t} ({p_c/b_t*100:.0f}%) | {sign}{delta_v} |")
    L.append("")

    # F1 변경된 파일 상세
    changed = [r for r in results
               if r["base_f1"] is not None and r["p1_f1"] is not None
               and abs(r["base_f1"] - r["p1_f1"]) > 5]
    L.append(f"## F1 값 변경된 파일 ({len(changed)} / {total})")
    L.append("")
    L.append("| 파일 | 정답 | F0 | base F1 | phase1 F1 | base 결과 | phase1 결과 |")
    L.append("|---|---|---:|---:|---:|---|---|")
    for r in changed:
        bm = (f"**{r['base_pred']}** ✓" if r["base_pred"] == r["true"]
              else f"{r['base_pred']} ✗")
        pm = (f"**{r['p1_pred']}** ✓" if r["p1_pred"] == r["true"]
              else f"{r['p1_pred']} ✗")
        L.append(
            f"| {r['path']} | {r['true']} | {r['f0']:.0f} "
            f"| {r['base_f1']:.0f} | {r['p1_f1']:.0f} "
            f"| {bm} | {pm} |"
        )
    L.append("")

    # 전체 파일 상세
    L.append("## 전체 파일 결과")
    L.append("")
    L.append("| 파일 | 정답 | F0 | F1 (base/p1) | F2 (base/p1) | base | p1 |")
    L.append("|---|---|---:|---:|---:|---|---|")
    for r in results:
        bm = (f"**{r['base_pred']}** ✓" if r["base_pred"] == r["true"]
              else f"{r['base_pred']} ✗")
        pm = (f"**{r['p1_pred']}** ✓" if r["p1_pred"] == r["true"]
              else f"{r['p1_pred']} ✗")
        f1s = (f"{r['base_f1']:.0f}/{r['p1_f1']:.0f}"
               if r["base_f1"] and r["p1_f1"] else "—")
        f2s = (f"{r['base_f2']:.0f}/{r['p1_f2']:.0f}"
               if r["base_f2"] and r["p1_f2"] else "—")
        L.append(f"| {r['path']} | {r['true']} | {r['f0']:.0f} "
                 f"| {f1s} | {f2s} | {bm} | {pm} |")
    L.append("")

    (RESULTS / "phase1_f0_aware.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
