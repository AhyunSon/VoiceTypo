"""
evaluation/phase_a_v3_f3_ratio.py — Phase A3: F3-normalized features

가설:
  F3 자체를 화자 vocal tract 의 reference 로 사용 → 비율 feature 가 화자 무관.
  Lobanov 의 z-score 정규화가 화자 평균/SD 필요한 반면,
  F3 비율은 단일 발화 단독으로 산출 가능 (cal-free).

Feature:
  R1 = F1 / F3,  R2 = F2 / F3
  → 모든 화자에서 같은 모음에 동일한 R1, R2 가까움 (이상적).

비교:
  A. Baseline (Bark Mahalanobis on F1, F2, F3 — 현재)
  B. Ratio (Bark on R1=F1/F3, R2=F2/F3)
  C. Combined (R1, R2 + Bark F3 또는 원본 결합)
  D. VTLN + Ratio (Phase A1 + A3 결합)

테스트 데이터:
  본인 35-wav + 가상 남성 (×0.83) + 가상 아동 (×1.20)
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
from vowel_classifier import classify_vowel, clear_user_refs, _REFS
from vtln import compute_warping_factor, warp_formants


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"


# ══════════════════════════════════════════
# Bark
# ══════════════════════════════════════════

def _bark(f):
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


# ══════════════════════════════════════════
# F3-ratio _REFS 자동 생성
# ══════════════════════════════════════════

def build_ratio_refs(base_refs: dict) -> dict:
    """F1, F2, F3 (μ, σ) → R1=F1/F3, R2=F2/F3 (μ, σ).

    SD 산출:
      Var(R) ≈ Var(F1) / F3² + F1² × Var(F3) / F3⁴   (델타 방법)
      여기서는 단순화: Bark 공간 (μ_F1 - μ_F3) σ 보정 사용.
    """
    refs = {}
    for v, (m1, sd1, m2, sd2, m3, sd3) in base_refs.items():
        # Bark 공간 차이 = F1 / F3 비율의 log 등가
        # 직접 비율 사용 (Bark 변환 후 차이)
        b1, b2, b3 = _bark(m1), _bark(m2), _bark(m3)
        bsd1 = max(_bark(m1 + sd1) - b1, 0.05)
        bsd2 = max(_bark(m2 + sd2) - b2, 0.05)
        bsd3 = max(_bark(m3 + sd3) - b3, 0.05)
        # R1 = b1 - b3, R2 = b2 - b3 (Bark 공간)
        # 분산: var(R) = var(b1) + var(b3) - 2*cov ≈ var(b1) + var(b3) (cov=0 가정)
        refs[v] = dict(
            r1_mean=b1 - b3,
            r1_sd=float(np.sqrt(bsd1**2 + bsd3**2)),
            r2_mean=b2 - b3,
            r2_sd=float(np.sqrt(bsd2**2 + bsd3**2)),
        )
    return refs


# ══════════════════════════════════════════
# Ratio 분류기
# ══════════════════════════════════════════

def classify_ratio(f1, f2, f3, refs):
    """Bark 공간 ratio (b1-b3, b2-b3) Mahalanobis 분류."""
    if f1 is None or f2 is None or f3 is None:
        return "?", 0.0
    if not (100 < f1 < 1500 and 200 < f2 < 4000 and 1500 < f3 < 5000):
        return "?", 0.0

    b1, b2, b3 = _bark(f1), _bark(f2), _bark(f3)
    r1, r2 = b1 - b3, b2 - b3

    best_v, best_d, second_d = "?", float("inf"), float("inf")
    for v, ref in refs.items():
        d_sq = ((r1 - ref["r1_mean"]) / ref["r1_sd"]) ** 2 + \
               ((r2 - ref["r2_mean"]) / ref["r2_sd"]) ** 2
        d = float(np.sqrt(d_sq))
        if d < best_d:
            second_d = best_d
            best_d = d
            best_v = v
        elif d < second_d:
            second_d = d

    if best_d > 4.0:
        return "?", 0.0

    base_conf  = max(0.0, 1.0 - best_d / 4.0)
    separation = min(1.0, (second_d - best_d) / 1.5)
    conf = float(0.7 * base_conf + 0.3 * separation)
    return best_v, conf


def classify_combined(f1, f2, f3, refs_ratio, w_ratio=0.5):
    """기존 (F1/F2/F3 Mahalanobis) + ratio 결합. 가중 평균."""
    pred_orig, conf_orig = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
    pred_rat, conf_rat   = classify_ratio(f1, f2, f3, refs_ratio)

    if pred_orig == "?" and pred_rat == "?":
        return "?", 0.0
    if pred_orig == "?":
        return pred_rat, conf_rat
    if pred_rat == "?":
        return pred_orig, conf_orig

    # 두 분류 결과 같으면 conf 합산
    if pred_orig == pred_rat:
        return pred_orig, (1 - w_ratio) * conf_orig + w_ratio * conf_rat
    # 다르면 confidence 가 큰 쪽
    if (1 - w_ratio) * conf_orig > w_ratio * conf_rat:
        return pred_orig, conf_orig
    return pred_rat, conf_rat


# ══════════════════════════════════════════
# 데이터 / 추출
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


def extract_center(audio):
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64),
                            sampling_frequency=float(SAMPLE_RATE))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    dur = audio.shape[0] / SAMPLE_RATE
    def _g(n):
        v = fmt.get_value_at_time(n, dur / 2)
        return None if (v is None or np.isnan(v)) else float(v)
    return _g(1), _g(2), _g(3)


def transform(formants, scale=1.0):
    f1, f2, f3 = formants
    return (f1 * scale if f1 is not None else None,
            f2 * scale if f2 is not None else None,
            f3 * scale if f3 is not None else None)


# ══════════════════════════════════════════
# 시나리오
# ══════════════════════════════════════════

def evaluate(test_files, mode, scale=1.0, vtln_alpha=None,
             refs_ratio=None):
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        f = transform(extract_center(audio), scale)

        if vtln_alpha is not None:
            f = warp_formants(*f, vtln_alpha)

        if mode == "baseline":
            pred, conf = classify_vowel(f[0], f[1], "female",
                                        f3=f[2], scale=1.0)
        elif mode == "ratio":
            pred, conf = classify_ratio(*f, refs_ratio)
        elif mode == "combined":
            pred, conf = classify_combined(*f, refs_ratio)
        else:
            raise ValueError(mode)

        rows.append(dict(true=v_true, file=path.name,
                         pred=pred, conf=conf))
    return rows


def summarize(rows):
    correct = sum(1 for r in rows if r["pred"] == r["true"])
    total = len(rows)
    by_v = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in rows:
        by_v[r["true"]]["total"] += 1
        if r["pred"] == r["true"]:
            by_v[r["true"]]["correct"] += 1
    return dict(correct=correct, total=total,
                accuracy=correct / total * 100.0 if total else 0.0,
                by_v=dict(by_v))


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
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    clear_user_refs()
    files = collect_files()

    refs_ratio = build_ratio_refs(_REFS["female"])

    print("=" * 60)
    print("Phase A3 — F3-normalized ratio features")
    print("=" * 60)
    print(f"  데이터: {len(files)} wav")
    print()
    print("─" * 60)
    print("Ratio refs (Bark 공간 b1-b3, b2-b3)")
    print("─" * 60)
    for v in VOWELS:
        r = refs_ratio[v]
        print(f"  {v}: R1={r['r1_mean']:+.2f}±{r['r1_sd']:.2f}  "
              f"R2={r['r2_mean']:+.2f}±{r['r2_sd']:.2f}")
    print()

    speakers = [
        ("본인 (canonical 여성)", 1.00),
        ("가상 남성 (×0.83)",     0.83),
        ("가상 아동 (×1.20)",     1.20),
    ]

    results_table = {}
    for spkr_label, scale in speakers:
        print("─" * 60)
        print(f"화자: {spkr_label}")
        print("─" * 60)

        spkr_alpha = speaker_alpha_for(files, scale)
        print(f"  speaker α: {spkr_alpha:.3f}")

        rows = {
            "A_baseline":   evaluate(files, "baseline", scale=scale),
            "B_ratio":      evaluate(files, "ratio", scale=scale,
                                     refs_ratio=refs_ratio),
            "C_combined":   evaluate(files, "combined", scale=scale,
                                     refs_ratio=refs_ratio),
            "D_vtln_baseline": evaluate(files, "baseline", scale=scale,
                                        vtln_alpha=spkr_alpha),
            "E_vtln_ratio": evaluate(files, "ratio", scale=scale,
                                     vtln_alpha=spkr_alpha,
                                     refs_ratio=refs_ratio),
            "F_vtln_combined": evaluate(files, "combined", scale=scale,
                                        vtln_alpha=spkr_alpha,
                                        refs_ratio=refs_ratio),
        }
        sums = {k: summarize(v) for k, v in rows.items()}
        results_table[spkr_label] = sums

        base = sums["A_baseline"]["accuracy"]
        for k in ["A_baseline", "B_ratio", "C_combined",
                 "D_vtln_baseline", "E_vtln_ratio", "F_vtln_combined"]:
            s = sums[k]
            d = s["accuracy"] - base
            mark = "↑" if d > 1 else "→" if abs(d) <= 1 else "↓"
            print(f"  {k:<22s} {s['correct']:>2d}/{s['total']:<2d} = "
                  f"{s['accuracy']:5.1f}%  {d:>+6.1f}%p  {mark}")
        print()

    # ── 종합 표 ──
    print("=" * 60)
    print("종합 정확도 (모든 화자, 모든 시나리오)")
    print("=" * 60)
    cols = ["A_baseline", "B_ratio", "C_combined",
            "D_vtln_baseline", "E_vtln_ratio", "F_vtln_combined"]
    print(f"  {'화자':<28s} " + " ".join(f"{c[:14]:>10s}" for c in cols))
    print(f"  {'-'*28} " + " ".join("-"*10 for _ in cols))
    for spkr_label, _ in speakers:
        s = results_table[spkr_label]
        print(f"  {spkr_label:<28s} " +
              " ".join(f"{s[c]['accuracy']:>9.1f}%" for c in cols))
    print()

    # ── MD ──
    md_path = RESULTS / "phase_a_v3_f3_ratio.md"
    L = ["# Phase A3 — F3-normalized ratio features",
         "",
         "**작성**: 2026-05-06",
         "",
         "## 가설",
         "F3 = 화자 vocal tract reference. R1=F1/F3, R2=F2/F3 비율은 화자 무관.",
         "Bark 공간 차이 (b1-b3, b2-b3) 로 구현 (log 비율 등가).",
         "",
         "## Ratio refs (Bark 공간)",
         "",
         "| 모음 | R1=b1-b3 (μ±σ) | R2=b2-b3 (μ±σ) |",
         "|---|---|---|"]
    for v in VOWELS:
        r = refs_ratio[v]
        L.append(f"| {v} | {r['r1_mean']:+.2f}±{r['r1_sd']:.2f} | "
                 f"{r['r2_mean']:+.2f}±{r['r2_sd']:.2f} |")

    L += ["", "## 결과 — 화자 × 시나리오",
          "",
          "| 화자 | A baseline | B ratio | C combined | "
          "D VTLN+base | E VTLN+ratio | F VTLN+comb |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for spkr_label, _ in speakers:
        s = results_table[spkr_label]
        L.append(f"| {spkr_label} | "
                 f"{s['A_baseline']['accuracy']:.1f}% | "
                 f"{s['B_ratio']['accuracy']:.1f}% | "
                 f"{s['C_combined']['accuracy']:.1f}% | "
                 f"{s['D_vtln_baseline']['accuracy']:.1f}% | "
                 f"{s['E_vtln_ratio']['accuracy']:.1f}% | "
                 f"{s['F_vtln_combined']['accuracy']:.1f}% |")
    L.append("")
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"산출물: {md_path}")


if __name__ == "__main__":
    main()
