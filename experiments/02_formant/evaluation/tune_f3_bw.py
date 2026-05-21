"""
evaluation/tune_f3_bw.py — F3 가중치 + Bandwidth 추가 조정

목표: cal 없이 학계 _REFS 휴리스틱 분류 정확도 향상.
대상: 모든 모음 (특히 우/오, 어/으, 에/이 같은 군집)

조정 변수:
  W_F3 ∈ {0.20, 0.30, 0.40, 0.50, 0.60, 0.70}
    - 현재 vowel_classifier._W_F3 = 0.30
    - 우/오 (rounding 효과로 F3 차이) 분리에 영향
  W_BW ∈ {0.0, 0.05, 0.10, 0.15, 0.20}
    - bandwidth (대역폭) 정보 추가
    - B1/F1 ratio (formant 폭 / 위치 비율)

평가 데이터:
  - 본인 35-wav (canonical-near 여성)
  - 합성 가상 남성 (×0.83)
  - 합성 가상 아동 (×1.20)
  → 다화자 일반화 확인

비교 baseline (현재):
  본인 54.3%, 가상 남성 48.6%, 가상 아동 22.9%
  → 평균 41.9%
"""

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
from vowel_classifier import _REFS


VOWELS = ["아", "에", "이", "오", "우", "으", "어"]
HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"


# ══════════════════════════════════════════
# Bark
# ══════════════════════════════════════════

def _bark(f):
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


# ══════════════════════════════════════════
# Custom 분류기 (W_F3 + W_BW)
# ══════════════════════════════════════════

def classify_tuned(f1, f2, f3, b1, b2,
                   w_f3: float = 0.30,
                   w_bw: float = 0.0,
                   gender: str = "female") -> tuple:
    """W_F3 가중치 + bandwidth 추가 분류.

    distance =  (1 - w_f3 - w_bw) × dist_F1F2
              + w_f3 × dist_F3
              + w_bw × dist_BW
    """
    if f1 is None or f2 is None or f1 < 100 or f2 < 200:
        return "?", 0.0

    b1_pt, b2_pt = _bark(f1), _bark(f2)
    use_f3 = (f3 is not None and 1500 < f3 < 4500)
    b3_pt = _bark(f3) if use_f3 else None

    use_bw = (w_bw > 0
              and b1 is not None and b2 is not None
              and 30 < b1 < 1000 and 30 < b2 < 1500)
    if use_bw:
        bw1_pt = np.log(max(b1 / f1, 0.01))    # log(B1/F1)
        bw2_pt = np.log(max(b2 / f2, 0.01))

    refs = _REFS.get(gender, _REFS["female"])
    best_v, best_d, second_d = "?", float("inf"), float("inf")

    for v, (m1, sd1, m2, sd2, m3, sd3) in refs.items():
        bm1, bm2, bm3 = _bark(m1), _bark(m2), _bark(m3)
        bsd1 = max(_bark(m1 + sd1) - bm1, 0.05)
        bsd2 = max(_bark(m2 + sd2) - bm2, 0.05)
        bsd3 = max(_bark(m3 + sd3) - bm3, 0.05)

        d_f1f2 = ((b1_pt - bm1) / bsd1) ** 2 + ((b2_pt - bm2) / bsd2) ** 2
        d_total = d_f1f2

        d_f3 = 0.0
        if use_f3:
            d_f3 = ((b3_pt - bm3) / bsd3) ** 2
            d_total = (1 - w_f3) * d_f1f2 + w_f3 * d_f3
            if use_bw:
                d_total = (1 - w_bw) * d_total + w_bw * 0.0  # placeholder

        # Bandwidth term (모음 무관 reference 사용 — 학계 평균 BW ratio 가정)
        # BW/F 의 typical 값: B1/F1 ≈ 0.10, B2/F2 ≈ 0.05
        # 모음별 분리 정보 거의 없음 → BW 추가 효과 낮을 수 있음
        if use_bw:
            # 단순 reference: 모음 간 BW ratio 차이는 작음.
            # log(B/F) typical: -2.3 (=0.10), -3.0 (=0.05).
            # 학계 BW data 없으므로 generic reference 적용.
            target_bw1, target_bw2 = -2.3, -3.0
            d_bw = ((bw1_pt - target_bw1) / 0.5) ** 2 + \
                   ((bw2_pt - target_bw2) / 0.5) ** 2
            d_total = (1 - w_bw) * d_total + w_bw * d_bw

        d = float(np.sqrt(max(d_total, 0)))

        if d < best_d:
            second_d = best_d
            best_d = d
            best_v = v
        elif d < second_d:
            second_d = d

    if best_d > 4.0:
        return "?", 0.0

    base_conf = max(0.0, 1.0 - best_d / 4.0)
    sep = min(1.0, (second_d - best_d) / 1.5)
    conf = float(0.7 * base_conf + 0.3 * sep)

    # 으/어 tiebreaker (vowel_classifier 와 동일)
    if best_v in {"으", "어"} and second_d - best_d < 0.8:
        f1_mid = 553.0 if gender == "female" else 470.0
        if f1 > f1_mid + 40:
            best_v = "어"
        elif f1 < f1_mid - 40:
            best_v = "으"

    return best_v, conf


# ══════════════════════════════════════════
# 데이터 로드
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


def extract_full(audio):
    """F1/F2/F3 + B1/B2 (중앙 시점)."""
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64),
                            sampling_frequency=float(SAMPLE_RATE))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    t = audio.shape[0] / SAMPLE_RATE / 2
    def _g(n):
        v = fmt.get_value_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)
    def _gb(n):
        v = fmt.get_bandwidth_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)
    return _g(1), _g(2), _g(3), _gb(1), _gb(2)


# ══════════════════════════════════════════
# 평가
# ══════════════════════════════════════════

def transform(formants_bw, scale=1.0):
    """formant 만 스케일, bw 는 그대로."""
    f1, f2, f3, b1, b2 = formants_bw
    return (f1 * scale if f1 else None,
            f2 * scale if f2 else None,
            f3 * scale if f3 else None,
            b1, b2)   # bw 는 가상 화자 시뮬도 그대로


def evaluate(files, scale, w_f3, w_bw):
    correct = 0
    total = 0
    by_v = defaultdict(lambda: {"correct": 0, "total": 0,
                                "errors": Counter()})
    for v_true, take, path in files:
        audio = load_wav(path)
        feat = transform(extract_full(audio), scale)
        f1, f2, f3, b1, b2 = feat
        pred, _ = classify_tuned(f1, f2, f3, b1, b2,
                                 w_f3=w_f3, w_bw=w_bw)
        by_v[v_true]["total"] += 1
        if pred == v_true:
            by_v[v_true]["correct"] += 1
            correct += 1
        else:
            by_v[v_true]["errors"][pred] += 1
        total += 1
    return correct / total, dict(by_v)


# ══════════════════════════════════════════
# 메인 — 그리드 탐색
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = collect_files()

    print("=" * 60)
    print("F3 weight + Bandwidth 그리드 탐색")
    print("=" * 60)
    print(f"  데이터: {len(files)} wav × 3 시나리오 (본인/남성시뮬/아동시뮬)")
    print()

    w_f3_grid = [0.20, 0.30, 0.40, 0.50, 0.60]
    w_bw_grid = [0.0, 0.05, 0.10, 0.15]

    speakers = [
        ("본인",   1.00),
        ("남성",   0.83),
        ("아동",   1.20),
    ]

    # 1. W_F3 sweep (W_BW=0)
    print("─" * 60)
    print("1) W_F3 sweep (W_BW=0)")
    print("─" * 60)
    print(f"  {'W_F3':>6s} | {'본인':>7s} {'남성':>7s} {'아동':>7s} {'평균':>7s}")
    f3_results = {}
    for w_f3 in w_f3_grid:
        accs = []
        for spkr_label, scale in speakers:
            acc, _ = evaluate(files, scale, w_f3, 0.0)
            accs.append(acc)
        avg = np.mean(accs)
        f3_results[w_f3] = (accs, avg)
        print(f"  {w_f3:>6.2f} | {accs[0]*100:>6.1f}% {accs[1]*100:>6.1f}% "
              f"{accs[2]*100:>6.1f}% {avg*100:>6.1f}%")

    best_f3 = max(f3_results, key=lambda k: f3_results[k][1])
    print(f"\n  → 최적 W_F3 = {best_f3} (평균 {f3_results[best_f3][1]*100:.1f}%)")
    print()

    # 2. W_BW sweep (W_F3 = best_f3)
    print("─" * 60)
    print(f"2) W_BW sweep (W_F3={best_f3})")
    print("─" * 60)
    print(f"  {'W_BW':>6s} | {'본인':>7s} {'남성':>7s} {'아동':>7s} {'평균':>7s}")
    bw_results = {}
    for w_bw in w_bw_grid:
        accs = []
        for spkr_label, scale in speakers:
            acc, _ = evaluate(files, scale, best_f3, w_bw)
            accs.append(acc)
        avg = np.mean(accs)
        bw_results[w_bw] = (accs, avg)
        print(f"  {w_bw:>6.2f} | {accs[0]*100:>6.1f}% {accs[1]*100:>6.1f}% "
              f"{accs[2]*100:>6.1f}% {avg*100:>6.1f}%")

    best_bw = max(bw_results, key=lambda k: bw_results[k][1])
    print(f"\n  → 최적 W_BW = {best_bw} "
          f"(평균 {bw_results[best_bw][1]*100:.1f}%)")
    print()

    # 3. 최적 조합 모음별 분석
    print("─" * 60)
    print(f"3) 최적 조합 (W_F3={best_f3}, W_BW={best_bw}) 모음별 분석")
    print("─" * 60)
    for spkr_label, scale in speakers:
        acc, by_v = evaluate(files, scale, best_f3, best_bw)
        print(f"\n  화자: {spkr_label} ({acc*100:.1f}%)")
        for v in VOWELS:
            d = by_v.get(v, {"correct": 0, "total": 0,
                             "errors": Counter()})
            if d["total"] == 0:
                continue
            mark = f"{d['correct']}/{d['total']}"
            err = ", ".join(f"{p}×{n}"
                            for p, n in d["errors"].most_common()) or "—"
            print(f"    {v}: {mark:>5s}  {err}")

    # 4. baseline (W_F3=0.30, W_BW=0) 대비
    print()
    print("─" * 60)
    print("4) Baseline (W_F3=0.30, W_BW=0) 대비")
    print("─" * 60)
    baseline_avg = f3_results[0.30][1] if 0.30 in f3_results else None
    best_avg = bw_results[best_bw][1]
    if baseline_avg is not None:
        delta = (best_avg - baseline_avg) * 100
        print(f"  baseline 평균: {baseline_avg*100:.1f}%")
        print(f"  최적 평균:    {best_avg*100:.1f}%")
        print(f"  변화:         {delta:+.1f}%p")
    print()

    # ── MD ──
    md_path = RESULTS / "tune_f3_bw.md"
    L = ["# F3 weight + Bandwidth 조정",
         "",
         "**작성**: 2026-05-06",
         "",
         "## W_F3 sweep (W_BW=0)",
         "",
         "| W_F3 | 본인 | 남성 시뮬 | 아동 시뮬 | 평균 |",
         "|---|---:|---:|---:|---:|"]
    for w_f3 in w_f3_grid:
        accs, avg = f3_results[w_f3]
        L.append(f"| {w_f3:.2f} | {accs[0]*100:.1f}% | "
                 f"{accs[1]*100:.1f}% | {accs[2]*100:.1f}% | "
                 f"**{avg*100:.1f}%** |")

    L += ["", f"**최적 W_F3 = {best_f3}**", "",
          f"## W_BW sweep (W_F3={best_f3})",
          "",
          "| W_BW | 본인 | 남성 시뮬 | 아동 시뮬 | 평균 |",
          "|---|---:|---:|---:|---:|"]
    for w_bw in w_bw_grid:
        accs, avg = bw_results[w_bw]
        L.append(f"| {w_bw:.2f} | {accs[0]*100:.1f}% | "
                 f"{accs[1]*100:.1f}% | {accs[2]*100:.1f}% | "
                 f"**{avg*100:.1f}%** |")

    L += ["", f"**최적 W_BW = {best_bw}**", "",
          f"## 최적 조합 (W_F3={best_f3}, W_BW={best_bw})",
          ""]
    if baseline_avg is not None:
        L.append(f"- Baseline (W_F3=0.30): {baseline_avg*100:.1f}%")
    L.append(f"- 최적: {best_avg*100:.1f}%")
    L.append("")

    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"산출물: {md_path}")


if __name__ == "__main__":
    main()
