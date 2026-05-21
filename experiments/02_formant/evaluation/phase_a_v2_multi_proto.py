"""
evaluation/phase_a_v2_multi_proto.py — Phase A2: Multi-prototype + VTLN 결합

Phase A1 결과:
  Baseline 54.3%, VTLN speaker (canonical 화자) 54.3% (no-op).
  가상 남성: 48.6% → 54.3% (VTLN 회복 +5.7%p).

Phase A2 가설:
  Multi-prototype 가 per-sample 동적 α 선택으로 화자 자동 적응.
  VTLN (speaker α 고정) 보다 다양한 발음 변동에 robust 가능.

비교:
  A. Baseline (single REFS, scale=1.0)
  B. VTLN speaker (Phase A1 D)
  C. Multi-prototype (per-sample, 5 α)
  D. Multi-prototype + vote (5 chunks)
  E. VTLN + Multi-prototype 결합

테스트 데이터:
  - 본인 35-wav (canonical-near 여성)
  - 가상 남성 시뮬 (×0.83)
  - 가상 아동 시뮬 (×1.20)

실행:
  python -m evaluation.phase_a_v2_multi_proto
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
from vtln import compute_warping_factor, warp_formants, F3_CANONICAL
from multi_prototype import (classify_multi_proto, vote_multi_proto,
                             DEFAULT_ALPHAS)


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]


# ══════════════════════════════════════════
# 데이터 / 추출 (이전 스크립트와 동일)
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
# 시나리오 (formant 변환 옵션 — 가상 화자 시뮬용)
# ══════════════════════════════════════════

def transform(formants, scale=1.0):
    """가상 화자 시뮬용 formant 스케일링."""
    f1, f2, f3 = formants
    return (f1 * scale if f1 is not None else None,
            f2 * scale if f2 is not None else None,
            f3 * scale if f3 is not None else None)


def scenario(test_files, mode: str, formant_scale: float = 1.0,
             vtln_alpha: float = None) -> list:
    """범용 평가 시나리오.

    mode:
      "baseline"     - classify_vowel(scale=1.0)
      "vtln"         - VTLN speaker α 적용 (vtln_alpha 인자)
      "multi_proto"  - classify_multi_proto (per-sample)
      "multi_vote"   - vote_multi_proto (5 chunks)
      "vtln_multi"   - VTLN + multi_proto

    formant_scale: 가상 화자 시뮬 (1.0=원본, 0.83=남성, 1.20=아동).
    """
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)

        if mode in ("multi_vote",):
            # 다중 시점
            samples = [transform(s, formant_scale)
                       for s in extract_multi(audio)]
            if vtln_alpha is not None:
                samples = [warp_formants(*s, vtln_alpha) for s in samples]
            v, c, nv, ad = vote_multi_proto(samples)
            rows.append(dict(true=v_true, file=path.name,
                             pred=v, conf=c, voters=nv,
                             alpha_dist=dict(ad)))
        else:
            f = transform(extract_center(audio), formant_scale)

            if mode == "baseline":
                pred, conf = classify_vowel(f[0], f[1], "female",
                                            f3=f[2], scale=1.0)
            elif mode == "vtln":
                wf = warp_formants(*f, vtln_alpha or 1.0)
                pred, conf = classify_vowel(wf[0], wf[1], "female",
                                            f3=wf[2], scale=1.0)
            elif mode == "multi_proto":
                pred, alpha, conf, _ = classify_multi_proto(*f)
                rows.append(dict(true=v_true, file=path.name,
                                 pred=pred, conf=conf, alpha=alpha))
                continue
            elif mode == "vtln_multi":
                wf = warp_formants(*f, vtln_alpha or 1.0)
                pred, alpha, conf, _ = classify_multi_proto(*wf)
                rows.append(dict(true=v_true, file=path.name,
                                 pred=pred, conf=conf, alpha=alpha))
                continue
            else:
                raise ValueError(mode)

            rows.append(dict(true=v_true, file=path.name,
                             pred=pred, conf=conf))
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


def speaker_alpha_for(files, formant_scale=1.0):
    """가상 화자 (formant_scale 적용 후) F3 평균 → speaker α."""
    f3_arr = []
    for v, t, path in files:
        audio = load_wav(path)
        _, _, f3 = extract_center(audio)
        if f3 is not None and 1500 < f3 < 4500:
            f3_arr.append(f3 * formant_scale)
    if not f3_arr:
        return 1.0
    return compute_warping_factor(float(np.mean(f3_arr)))


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    clear_user_refs()
    files = collect_files()

    print("=" * 60)
    print("Phase A2 — Multi-prototype + VTLN 결합")
    print("=" * 60)
    print(f"  데이터: {len(files)} wav (단일 화자, canonical-near 여성)")
    print(f"  α 후보: {DEFAULT_ALPHAS}")
    print()

    # 화자 시뮬레이션 3 종
    speakers = [
        ("본인 (canonical-near 여성)", 1.00),
        ("가상 남성 (formants × 0.83)", 0.83),
        ("가상 아동 (formants × 1.20)", 1.20),
    ]

    all_summaries = {}
    for spkr_label, scale in speakers:
        print("─" * 60)
        print(f"화자: {spkr_label}")
        print("─" * 60)

        spkr_alpha = speaker_alpha_for(files, scale)
        print(f"  speaker α (VTLN 추정): {spkr_alpha:.3f}")

        rows = {}
        rows["A_baseline"]   = scenario(files, "baseline", formant_scale=scale)
        rows["B_vtln"]       = scenario(files, "vtln",
                                        formant_scale=scale,
                                        vtln_alpha=spkr_alpha)
        rows["C_multi"]      = scenario(files, "multi_proto",
                                        formant_scale=scale)
        rows["D_multi_vote"] = scenario(files, "multi_vote",
                                        formant_scale=scale)
        rows["E_vtln_multi"] = scenario(files, "vtln_multi",
                                        formant_scale=scale,
                                        vtln_alpha=spkr_alpha)

        summaries = {k: summarize(v) for k, v in rows.items()}
        all_summaries[spkr_label] = summaries

        print()
        print(f"  {'시나리오':<25s} {'정확도':>11s}  {'vs A':>8s}")
        print(f"  {'-'*25} {'-'*11}  {'-'*8}")
        base_acc = summaries["A_baseline"]["accuracy"]
        for k in ["A_baseline", "B_vtln", "C_multi",
                 "D_multi_vote", "E_vtln_multi"]:
            s = summaries[k]
            d = s["accuracy"] - base_acc
            print(f"  {k:<25s} {s['correct']:>2d}/{s['total']:<2d} = "
                  f"{s['accuracy']:5.1f}%  {d:>+6.1f}%p")
        print()

    # ── α 분포 (multi vote 케이스) ──
    print("─" * 60)
    print("Multi-vote α 선택 분포 (D scenario)")
    print("─" * 60)
    for spkr_label, scale in speakers:
        rows = scenario(files, "multi_vote", formant_scale=scale)
        all_alphas = Counter()
        for r in rows:
            for a, n in r.get("alpha_dist", {}).items():
                all_alphas[a] += n
        total = sum(all_alphas.values())
        if total > 0:
            dist = {a: f"{n/total*100:.0f}%"
                    for a, n in sorted(all_alphas.items())}
            print(f"  {spkr_label}: {dist}")
    print()

    # ── 종합 보고 ──
    print("=" * 60)
    print("종합")
    print("=" * 60)
    print(f"  {'화자':<35s} {'A 베이스':>10s} {'B VTLN':>10s} "
          f"{'C MP':>10s} {'D MP+vt':>10s} {'E VT+MP':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for spkr_label, _ in speakers:
        s = all_summaries[spkr_label]
        print(f"  {spkr_label:<35s} "
              f"{s['A_baseline']['accuracy']:>9.1f}% "
              f"{s['B_vtln']['accuracy']:>9.1f}% "
              f"{s['C_multi']['accuracy']:>9.1f}% "
              f"{s['D_multi_vote']['accuracy']:>9.1f}% "
              f"{s['E_vtln_multi']['accuracy']:>9.1f}%")
    print()

    # ── MD ──
    md_path = RESULTS / "phase_a_v2_multi_proto.md"
    L = ["# Phase A2 — Multi-prototype + VTLN 결합",
         "",
         "**작성**: 2026-05-06",
         "",
         "## 가설",
         "Multi-prototype (5 α 후보) 가 per-sample 동적 화자 적응 → "
         "VTLN (speaker α 고정) 보다 발음 변동 robust.",
         "",
         "## 비교",
         "",
         "| 화자 | A 기준 | B VTLN | C Multi-P | D MP+vote | E VT+MP |",
         "|---|---:|---:|---:|---:|---:|"]
    for spkr_label, _ in speakers:
        s = all_summaries[spkr_label]
        L.append(f"| {spkr_label} | "
                 f"{s['A_baseline']['accuracy']:.1f}% | "
                 f"{s['B_vtln']['accuracy']:.1f}% | "
                 f"{s['C_multi']['accuracy']:.1f}% | "
                 f"{s['D_multi_vote']['accuracy']:.1f}% | "
                 f"{s['E_vtln_multi']['accuracy']:.1f}% |")

    L += ["",
          f"## α 후보: {list(DEFAULT_ALPHAS)}",
          "",
          "## 핵심 발견",
          "",
          "(결과 분석 후 채울 자리)",
          ""]
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"산출물: {md_path}")


if __name__ == "__main__":
    main()
