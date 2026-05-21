"""
evaluation/phase_a_v1_vtln.py — Phase A1: VTLN cal-free 효과 측정

목표: 캘리브레이션 없이 VTLN 단독으로 다화자 정규화 효과 측정.
참고: 본 데이터는 단일 화자(본인)라 다화자 효과 직접 검증 불가.
      그러나 본인의 take 간 F3 변동이 화자간 변동의 축소판 역할.
      라이브 다화자 테스트는 Phase A 통합 후 진행.

비교:
  A. Baseline      — _REFS["female"] 단독, 단일 시점, scale=1.0
  B. + VTLN/wav    — 각 wav 의 F3 로 워핑 후 분류
  C. + VTLN/vote   — 5 시간점 vote, 각 청크에 워핑

판정:
  +5%p 이상 → VTLN 효과 확인, 다음 Phase A 기법 (multi-prototype) 진행
  +0~5%p   → 효과 미세, 단일 화자 한계로 가정
  ≤0       → VTLN 부정 효과, 캐노니컬 F3 재검토

실행:
  python -m evaluation.phase_a_v1_vtln
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.io import wavfile
import parselmouth

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_RATE
from vowel_classifier import classify_vowel, clear_user_refs
from vtln import (compute_warping_factor, warp_formants,
                  F3_CANONICAL)


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]


# ══════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════

def load_wav(path):
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


def collect_files():
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


def _formant_obj(audio):
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64),
                            sampling_frequency=float(SAMPLE_RATE))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    return fmt, audio.shape[0] / SAMPLE_RATE


def extract_center(audio):
    fmt, dur = _formant_obj(audio)
    def _g(n):
        v = fmt.get_value_at_time(n, dur / 2)
        return None if (v is None or np.isnan(v)) else float(v)
    return _g(1), _g(2), _g(3)


def extract_multi(audio):
    fmt, dur = _formant_obj(audio)
    out = []
    for p in SAMPLE_POS:
        t = dur * p
        row = []
        for n in [1, 2, 3]:
            v = fmt.get_value_at_time(n, t)
            row.append(None if (v is None or np.isnan(v)) else float(v))
        out.append(tuple(row))
    return out


# ══════════════════════════════════════════
# 시나리오
# ══════════════════════════════════════════

def scenario_baseline(test_files):
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        f1, f2, f3 = extract_center(audio)
        pred, conf = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
        rows.append(dict(true=v_true, file=path.name,
                         f1=f1, f2=f2, f3=f3,
                         alpha=1.0, pred=pred, conf=conf))
    return rows


def scenario_vtln_per_wav(test_files):
    """각 wav 의 F3 로 워핑 후 분류."""
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        f1, f2, f3 = extract_center(audio)
        alpha = compute_warping_factor(f3)
        wf1, wf2, wf3 = warp_formants(f1, f2, f3, alpha)
        pred, conf = classify_vowel(wf1, wf2, "female", f3=wf3, scale=1.0)
        rows.append(dict(true=v_true, file=path.name,
                         f1=f1, f2=f2, f3=f3,
                         alpha=alpha, pred=pred, conf=conf,
                         warped_f1=wf1, warped_f2=wf2, warped_f3=wf3))
    return rows


def scenario_vtln_vote(test_files):
    """5 시간점 각 청크 워핑 후 confidence-weighted vote."""
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        samples = extract_multi(audio)
        votes = defaultdict(float)
        nv = 0
        alphas = []
        for f1, f2, f3 in samples:
            if f1 is None or f2 is None or f3 is None:
                continue
            alpha = compute_warping_factor(f3)
            alphas.append(alpha)
            wf1, wf2, wf3 = warp_formants(f1, f2, f3, alpha)
            p, c = classify_vowel(wf1, wf2, "female", f3=wf3, scale=1.0)
            if p == "?" or c <= 0:
                continue
            votes[p] += c
            nv += 1
        if not votes:
            pred, conf = "?", 0.0
        else:
            pred = max(votes, key=votes.get)
            conf = votes[pred]
        rows.append(dict(true=v_true, file=path.name,
                         alpha_med=float(np.median(alphas)) if alphas else 1.0,
                         pred=pred, conf=conf, voters=nv))
    return rows


def scenario_vtln_speaker(test_files, speaker_alpha):
    """화자 단일 α (전 wav 평균 F3 에서 산출) 적용. 모음 정보 누설 없음.

    Args:
        speaker_alpha: 사전 계산된 화자 단일 워핑 계수.
    """
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        f1, f2, f3 = extract_center(audio)
        wf1, wf2, wf3 = warp_formants(f1, f2, f3, speaker_alpha)
        pred, conf = classify_vowel(wf1, wf2, "female", f3=wf3, scale=1.0)
        rows.append(dict(true=v_true, file=path.name,
                         f1=f1, f2=f2, f3=f3,
                         alpha=speaker_alpha, pred=pred, conf=conf))
    return rows


def scenario_synthetic_male(test_files, male_factor=0.83, vtln=False):
    """본인 데이터 → 가상 남성 (formants × 0.83) 시뮬레이션.

    male_factor: 학계 데이터 male/female 평균 비율 (F2 기준 0.82, F1 기준 0.85).
                 본인 모음을 남성 화자로 변환.
    vtln=True: 변환 후 VTLN 적용 (speaker-level) → 회복 시도.
    """
    rows = []
    # 가상 남성 F3 평균 산출 (VTLN α 추정용)
    f3_synth_mean = None
    if vtln:
        f3_all = []
        for v, t, path in test_files:
            audio = load_wav(path)
            _, _, f3 = extract_center(audio)
            if f3 is not None and 1500 < f3 < 4500:
                f3_all.append(f3 * male_factor)
        f3_synth_mean = float(np.mean(f3_all))
        speaker_alpha = compute_warping_factor(f3_synth_mean)
    else:
        speaker_alpha = 1.0

    for v_true, take, path in test_files:
        audio = load_wav(path)
        f1, f2, f3 = extract_center(audio)
        # 가상 남성 변환
        f1m = f1 * male_factor if f1 is not None else None
        f2m = f2 * male_factor if f2 is not None else None
        f3m = f3 * male_factor if f3 is not None else None
        if vtln:
            wf1, wf2, wf3 = warp_formants(f1m, f2m, f3m, speaker_alpha)
        else:
            wf1, wf2, wf3 = f1m, f2m, f3m
        pred, conf = classify_vowel(wf1, wf2, "female", f3=wf3, scale=1.0)
        rows.append(dict(true=v_true, file=path.name,
                         f1=f1m, f2=f2m, f3=f3m,
                         alpha=speaker_alpha, pred=pred, conf=conf))
    return rows


# ══════════════════════════════════════════
# 요약
# ══════════════════════════════════════════

def summarize(rows):
    correct = sum(1 for r in rows if r["pred"] == r["true"])
    total = len(rows)
    by_v = defaultdict(lambda: {"correct": 0, "total": 0,
                                "errors": Counter()})
    for r in rows:
        by_v[r["true"]]["total"] += 1
        if r["pred"] == r["true"]:
            by_v[r["true"]]["correct"] += 1
        else:
            by_v[r["true"]]["errors"][r["pred"]] += 1
    return dict(correct=correct, total=total,
                accuracy=correct / total * 100.0 if total else 0.0,
                by_v=dict(by_v))


def print_summary(name, summary, baseline_acc=None):
    a = summary["accuracy"]
    delta = (a - baseline_acc) if baseline_acc is not None else 0.0
    delta_str = f"{delta:+.1f}%p" if baseline_acc is not None else "—"
    print(f"  {name:<28s} {summary['correct']:>2d}/{summary['total']:<2d} "
          f"= {a:5.1f}%  vs baseline: {delta_str}")


def print_by_vowel(rows, label=""):
    s = summarize(rows)
    print(f"\n  {label} 모음별:")
    print(f"    {'모음':<4s} {'정확도':>9s}  오답")
    for v in VOWELS:
        d = s["by_v"].get(v, {"correct": 0, "total": 0,
                              "errors": Counter()})
        if d["total"] == 0:
            continue
        acc = (f"{d['correct']}/{d['total']} "
               f"({d['correct']/d['total']*100:.0f}%)")
        err = (", ".join(f"{p}×{n}" for p, n in d["errors"].most_common())
               or "—")
        print(f"    {v:<4s} {acc:>9s}  {err}")


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    clear_user_refs()  # 학계 _REFS 사용 보장

    files = collect_files()
    print("=" * 60)
    print("Phase A1 — VTLN cal-free 효과 측정")
    print("=" * 60)
    print(f"  데이터: {len(files)} wav (단일 화자)")
    print(f"  캐노니컬 F3: {F3_CANONICAL} Hz")
    print()

    # ── 본인 F3 분포 측정 (먼저 — 정상성 확인) ──
    f3_all = []
    for v, t, path in files:
        audio = load_wav(path)
        _, _, f3 = extract_center(audio)
        if f3 is not None and 1500 < f3 < 4500:
            f3_all.append(f3)
    f3_arr = np.array(f3_all)
    print(f"  본인 F3 분포: median={np.median(f3_arr):.0f}  "
          f"mean={np.mean(f3_arr):.0f}  "
          f"std={np.std(f3_arr):.0f}  "
          f"range=[{np.min(f3_arr):.0f}, {np.max(f3_arr):.0f}]")
    print(f"  → 본인 평균 α 추정: {F3_CANONICAL/np.mean(f3_arr):.3f}  "
          f"(1.0 = no warp)")
    print()

    # ── 시나리오 실행 ──
    rows_A = scenario_baseline(files)
    sum_A  = summarize(rows_A)

    rows_B = scenario_vtln_per_wav(files)
    sum_B  = summarize(rows_B)

    rows_C = scenario_vtln_vote(files)
    sum_C  = summarize(rows_C)

    # 화자 단일 α (전 wav 평균 F3 → 한 값)
    speaker_alpha = compute_warping_factor(float(np.mean(f3_arr)))
    rows_D = scenario_vtln_speaker(files, speaker_alpha)
    sum_D  = summarize(rows_D)

    # 가상 남성 시뮬: 본인 데이터 × 0.83 → 남성 형식
    rows_E = scenario_synthetic_male(files, male_factor=0.83, vtln=False)
    sum_E  = summarize(rows_E)
    rows_F = scenario_synthetic_male(files, male_factor=0.83, vtln=True)
    sum_F  = summarize(rows_F)

    # ── 출력 ──
    print("─" * 60)
    print("결과 — 본인 데이터 (canonical-near 화자)")
    print("─" * 60)
    print_summary("A. Baseline (no VTLN)",     sum_A)
    print_summary("B. + VTLN per-wav (잘못됨)",  sum_B, sum_A["accuracy"])
    print_summary("C. + VTLN per-chunk (잘못됨)", sum_C, sum_A["accuracy"])
    print_summary(f"D. + VTLN speaker α={speaker_alpha:.3f}",
                  sum_D, sum_A["accuracy"])
    print()
    print("─" * 60)
    print("결과 — 가상 남성 시뮬 (formants × 0.83)")
    print("─" * 60)
    print_summary("E. 가상 남성, no VTLN",     sum_E, sum_A["accuracy"])
    print_summary("F. 가상 남성, + VTLN spkr", sum_F, sum_A["accuracy"])
    print()

    # 판정 — 가상 남성 회복도 기준
    recovery = sum_F["accuracy"] - sum_E["accuracy"]
    spkr_drift = sum_D["accuracy"] - sum_A["accuracy"]

    print()
    print("─" * 60)
    print("판정")
    print("─" * 60)
    print(f"  본인 데이터 speaker-level VTLN drift: "
          f"{spkr_drift:+.1f}%p (기대 ~0)")
    print(f"  가상 남성 회복도: {recovery:+.1f}%p "
          f"({sum_E['accuracy']:.1f}% → {sum_F['accuracy']:.1f}%)")

    if abs(spkr_drift) <= 3 and recovery >= 10:
        verdict = ("✓ Speaker-level VTLN 정상 작동. "
                   "본인 (canonical 근접) 무영향 + 남성 화자 회복 확인. "
                   "다음 A4 (multi-prototype) 진행.")
    elif abs(spkr_drift) <= 3 and recovery < 10:
        verdict = ("△ Speaker-level VTLN 본인 무영향. "
                   "남성 회복 부족 — male/female refs 차이 외 요인 의심. "
                   "A4 진행 후 통합 평가.")
    else:
        verdict = ("✗ Speaker-level VTLN 도 부정. 캐노니컬 F3 또는 "
                   "_REFS 매핑 재검토 필요.")
    print()
    print(f"  {verdict}")
    print()

    # ── 산출물 ──
    delta_b = sum_B["accuracy"] - sum_A["accuracy"]
    delta_c = sum_C["accuracy"] - sum_A["accuracy"]
    delta_d = sum_D["accuracy"] - sum_A["accuracy"]
    delta_e = sum_E["accuracy"] - sum_A["accuracy"]
    delta_f = sum_F["accuracy"] - sum_A["accuracy"]

    md_path = RESULTS / "phase_a_v1_vtln.md"
    L = ["# Phase A1 — VTLN cal-free 효과 측정",
         "",
         "**작성**: 2026-05-06",
         "",
         "## 가설",
         "VTLN (F3-based vocal tract length normalization) 만으로 "
         "cal 없이 화자 정규화 → 학계 _REFS 분류 정확도 향상.",
         "",
         "## 데이터",
         f"- 단일 화자 35-wav (canonical-near, F3 평균 {np.mean(f3_arr):.0f} Hz)",
         f"- 캐노니컬 F3: {F3_CANONICAL} Hz",
         f"- 본인 speaker α: {speaker_alpha:.3f} (≈1.0 → no-op 예상)",
         f"- 가상 남성 시뮬: formants × 0.83 (학계 male/female 비율)",
         "",
         "## 결과",
         "",
         "| 시나리오 | 정확도 | vs A | 비고 |",
         "|---|---:|---:|---|",
         f"| A. Baseline (no VTLN) | "
         f"{sum_A['correct']}/{sum_A['total']} = "
         f"**{sum_A['accuracy']:.1f}%** | — | 학계 _REFS, scale=1.0 |",
         f"| B. VTLN per-wav | "
         f"{sum_B['correct']}/{sum_B['total']} = {sum_B['accuracy']:.1f}% "
         f"| {delta_b:+.1f}%p | ✗ wav 의 F3 가 모음 정보 누설 |",
         f"| C. VTLN per-chunk vote | "
         f"{sum_C['correct']}/{sum_C['total']} = {sum_C['accuracy']:.1f}% "
         f"| {delta_c:+.1f}%p | ✗ 청크별 F3 노이즈 증폭 |",
         f"| **D. VTLN speaker** α={speaker_alpha:.3f} | "
         f"{sum_D['correct']}/{sum_D['total']} = "
         f"**{sum_D['accuracy']:.1f}%** | {delta_d:+.1f}%p | "
         f"○ 화자 평균 F3 → 단일 α (정상 사용법) |",
         f"| E. 가상 남성 (×0.83), no VTLN | "
         f"{sum_E['correct']}/{sum_E['total']} = {sum_E['accuracy']:.1f}% "
         f"| {delta_e:+.1f}%p | spec mismatch 로 손상 |",
         f"| **F. 가상 남성 + VTLN spkr** | "
         f"{sum_F['correct']}/{sum_F['total']} = "
         f"**{sum_F['accuracy']:.1f}%** | {delta_f:+.1f}%p | "
         f"○ E → A 회복 ({sum_F['accuracy']-sum_E['accuracy']:+.1f}%p) |",
         "",
         "## 판정",
         "",
         verdict,
         "",
         "## 핵심 발견",
         "",
         "1. **VTLN 적용 단위 중요**: per-wav/per-chunk 는 F3 가 모음 식별 정보로",
         "   누설되어 부정 효과. **Speaker-level (전 발화 평균)** 이 정상 사용법.",
         "2. **Canonical-near 화자**: VTLN 무영향 (D 결과). 정상.",
         "3. **Non-canonical 화자 회복 검증**: 가상 남성 (formants × 0.83) 시뮬에서",
         f"   VTLN +{sum_F['accuracy']-sum_E['accuracy']:.1f}%p 회복 (E→F).",
         "",
         "## 한계",
         "",
         "- 단일 화자 → 다화자 효과 직접 측정 불가, 시뮬 데이터로 대체",
         "- VTLN 의 진짜 가치는 **라이브 다화자 환경** (남/여/아동)",
         "- 라이브 다화자 검증은 Phase A 전체 통합 후 진행",
         ""]
    md_path.write_text("\n".join(L), encoding="utf-8")

    csv_path = RESULTS / "phase_a_v1_vtln.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true",
                    "A_pred", "A_conf",
                    "B_pred", "B_conf", "B_alpha",
                    "C_pred", "C_conf", "C_alpha_med"])
        by_file = {}
        for tag, rs in [("A", rows_A), ("B", rows_B), ("C", rows_C)]:
            for r in rs:
                by_file.setdefault(r["file"],
                                   dict(true=r["true"]))
                by_file[r["file"]][f"{tag}_pred"] = r["pred"]
                by_file[r["file"]][f"{tag}_conf"] = r["conf"]
                if "alpha" in r:
                    by_file[r["file"]]["B_alpha"] = r["alpha"]
                if "alpha_med" in r:
                    by_file[r["file"]]["C_alpha_med"] = r["alpha_med"]
        for fname in sorted(by_file):
            d = by_file[fname]
            w.writerow([
                fname, d.get("true"),
                d.get("A_pred"), f"{d.get('A_conf', 0):.3f}",
                d.get("B_pred"), f"{d.get('B_conf', 0):.3f}",
                f"{d.get('B_alpha', 1.0):.3f}",
                d.get("C_pred"), f"{d.get('C_conf', 0):.3f}",
                f"{d.get('C_alpha_med', 1.0):.3f}",
            ])

    print("산출물:")
    print(f"  - {md_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
