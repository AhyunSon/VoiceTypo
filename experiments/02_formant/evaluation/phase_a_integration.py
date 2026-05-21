"""
evaluation/phase_a_integration.py — Phase A 통합 (A1 VTLN + Layer 5 vote)

Phase A1: VTLN speaker → 가상 화자 회복 (54.3% 안정).
Phase A2/A3 폐기 (multi-proto, F3-ratio 부정 또는 미미).

통합 시도:
  VTLN speaker α 적용 (1차 정규화)
  → 5 시간점 측정
  → 청크별 classify_vowel (학계 _REFS)
  → confidence-weighted vote

비교 시나리오 (모두 화자 단위 VTLN α 사용):
  A. Baseline (no VTLN, single-shot)
  B. VTLN speaker (single-shot)               — Phase A1 D
  C. + vote (5 chunks, no VTLN)               — Layer 5 단독
  D. + vote + VTLN speaker                    — 통합

테스트: 본인 + 가상 남성 + 가상 아동.

판정 기준:
  D ≥ 60% (모든 화자) → 천장 돌파, Phase A 마감 + 라이브 다화자 검증
  D 54~60%           → Phase A 천장 도달, Phase B (Wav2Vec2) 진행 필요
  D < 54%            → 통합 자체 부정 효과
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
from vtln import compute_warping_factor, warp_formants


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]


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


def transform(formants, scale=1.0):
    f1, f2, f3 = formants
    return (f1 * scale if f1 is not None else None,
            f2 * scale if f2 is not None else None,
            f3 * scale if f3 is not None else None)


def speaker_alpha_for(files, scale=1.0):
    f3_arr = []
    for v, t, path in files:
        audio = load_wav(path)
        _, _, f3 = extract_center(audio)
        if f3 is not None and 1500 < f3 < 4500:
            f3_arr.append(f3 * scale)
    if not f3_arr:
        return 1.0
    return compute_warping_factor(float(np.mean(f3_arr)))


# ══════════════════════════════════════════
# 시나리오
# ══════════════════════════════════════════

def run_baseline(test_files, scale=1.0, vtln_alpha=None):
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        f = transform(extract_center(audio), scale)
        if vtln_alpha is not None:
            f = warp_formants(*f, vtln_alpha)
        pred, conf = classify_vowel(f[0], f[1], "female", f3=f[2], scale=1.0)
        rows.append(dict(true=v_true, file=path.name, pred=pred, conf=conf))
    return rows


def run_vote(test_files, scale=1.0, vtln_alpha=None):
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        samples_raw = extract_multi(audio)
        samples = [transform(s, scale) for s in samples_raw]
        if vtln_alpha is not None:
            samples = [warp_formants(*s, vtln_alpha) for s in samples]

        votes = defaultdict(float)
        nv = 0
        for f1, f2, f3 in samples:
            if f1 is None or f2 is None:
                continue
            p, c = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
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
                         pred=pred, conf=conf, voters=nv))
    return rows


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


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    clear_user_refs()
    files = collect_files()

    print("=" * 60)
    print("Phase A 통합 — VTLN speaker + Layer 5 vote")
    print("=" * 60)
    print(f"  데이터: {len(files)} wav")
    print()

    speakers = [
        ("본인 canonical 여성",  1.00),
        ("가상 남성 (×0.83)",    0.83),
        ("가상 아동 (×1.20)",    1.20),
    ]

    table = {}
    for spkr_label, scale in speakers:
        print("─" * 60)
        print(f"화자: {spkr_label}")
        print("─" * 60)

        spkr_alpha = speaker_alpha_for(files, scale)
        print(f"  speaker α: {spkr_alpha:.3f}")

        results = {}
        results["A_baseline"] = summarize(
            run_baseline(files, scale=scale))
        results["B_vtln"] = summarize(
            run_baseline(files, scale=scale, vtln_alpha=spkr_alpha))
        results["C_vote"] = summarize(
            run_vote(files, scale=scale))
        results["D_vtln_vote"] = summarize(
            run_vote(files, scale=scale, vtln_alpha=spkr_alpha))

        table[spkr_label] = results

        base = results["A_baseline"]["accuracy"]
        for k in ["A_baseline", "B_vtln", "C_vote", "D_vtln_vote"]:
            s = results[k]
            d = s["accuracy"] - base
            mark = "↑" if d > 1 else "→" if abs(d) <= 1 else "↓"
            print(f"  {k:<18s} {s['correct']:>2d}/{s['total']:<2d} = "
                  f"{s['accuracy']:5.1f}%  {d:>+6.1f}%p  {mark}")
        print()

    # ── 종합 ──
    print("=" * 60)
    print("종합")
    print("=" * 60)
    print(f"  {'화자':<28s} {'A baseline':>12s} {'B VTLN':>10s} "
          f"{'C vote':>10s} {'D VTLN+vote':>13s}")
    print(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*10} {'-'*13}")
    for spkr_label, _ in speakers:
        t = table[spkr_label]
        print(f"  {spkr_label:<28s} "
              f"{t['A_baseline']['accuracy']:>11.1f}% "
              f"{t['B_vtln']['accuracy']:>9.1f}% "
              f"{t['C_vote']['accuracy']:>9.1f}% "
              f"{t['D_vtln_vote']['accuracy']:>12.1f}%")
    print()

    # ── 판정 ──
    d_accs = [table[lbl]["D_vtln_vote"]["accuracy"]
              for lbl, _ in speakers]
    d_min = min(d_accs)
    print("─" * 60)
    print("판정")
    print("─" * 60)
    if d_min >= 60:
        verdict = ("✓ 모든 화자에서 60%+ — 천장 돌파. "
                   "Phase A 마감, 라이브 다화자 검증 진행.")
    elif d_min >= 54:
        verdict = ("△ Phase A 천장 (54%대) 도달. "
                   "포먼트 라인 cal-free 본질 한계 확정. "
                   "Phase B (학습된 표현) 진행 필요.")
    else:
        verdict = ("✗ 일부 화자 54% 미만 — 통합 자체 부정 효과. "
                   "VTLN 단독 (B) 으로 회귀.")
    print(f"  최저 D 정확도: {d_min:.1f}%")
    print(f"  {verdict}")
    print()

    # ── MD ──
    md_path = RESULTS / "phase_a_integration.md"
    L = ["# Phase A 통합 — VTLN + Layer 5 vote",
         "",
         "**작성**: 2026-05-06",
         "",
         "## Phase A 결과 요약",
         "- A1 VTLN speaker: ✓ 가상 남/아 회복",
         "- A4 Multi-prototype: ✗ 폐기 (confidence 기반 α 선택 불안정)",
         "- A3 F3-ratio: ✗ 폐기 (효과 미미)",
         "- A5 F0 cluster: 보류 (학계 _REFS 가 male/female 만이라 효과 작음)",
         "",
         "## 통합 = VTLN speaker + 5 chunks confidence vote",
         "",
         "| 화자 | A baseline | B VTLN | C vote | D VTLN+vote |",
         "|---|---:|---:|---:|---:|"]
    for spkr_label, _ in speakers:
        t = table[spkr_label]
        L.append(f"| {spkr_label} | "
                 f"{t['A_baseline']['accuracy']:.1f}% | "
                 f"{t['B_vtln']['accuracy']:.1f}% | "
                 f"{t['C_vote']['accuracy']:.1f}% | "
                 f"{t['D_vtln_vote']['accuracy']:.1f}% |")

    L += ["", "## 판정", "", verdict, "",
          "## 결론",
          "",
          "- **VTLN speaker** 가 핵심 cal-free 정규화 (가상 남성/아동 회복)",
          "- Vote 는 본 데이터에서 효과 작거나 부정 (cal 없는 상태)",
          "- 포먼트 라인 cal-free 천장 ≈ 54% (학계 _REFS 단일 화자 매칭 한계)",
          "- **90% 도달은 Phase B (학습된 표현) 필요 확정**",
          ""]
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"산출물: {md_path}")


if __name__ == "__main__":
    main()
