"""
evaluation/baseline_simple.py — 단순 baseline (Praat 단독 + Mahalanobis)

원칙: 복잡한 것 추가 금지.
  - 추출: Praat Burg only, ceiling 5500 (단일)
  - 분류: vowel_classifier.classify_vowel (Bark Mahalanobis), scale=1.0
  - 우회: ensemble / Kalman / wav2vec2 / F0 정규화 / HNR 게이팅

기존 코드 수정 없음. classify_vowel 만 import 해서 그대로 사용.

실행:
  cd /c/Users/admin/Desktop/realtime_formant
  python -m evaluation.baseline_simple
"""

import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.io import wavfile
import parselmouth

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

plt.rcParams["font.family"] = ["Malgun Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_RATE
from vowel_classifier import classify_vowel


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"


# ══════════════════════════════════════════
# 데이터 로드
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
# 포먼트 추출 (Praat 단독, ceiling 5500, 중앙 시점)
# ══════════════════════════════════════════

def extract_formants(audio: np.ndarray, sr: int = SAMPLE_RATE):
    audio = audio - np.mean(audio)               # DC 제거
    snd = parselmouth.Sound(
        audio.astype(np.float64),
        sampling_frequency=float(sr),
    )
    fmt = snd.to_formant_burg(
        time_step=None,                # auto = 0.025/4 = 6.25ms
        max_number_of_formants=5,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50,
    )
    t = audio.shape[0] / sr / 2

    def _get(n):
        v = fmt.get_value_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)

    return _get(1), _get(2), _get(3)


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    print("=" * 60)
    print("Simple Baseline (Praat ceiling 5500 + classify_vowel)")
    print("=" * 60)
    print(f"  파일: {len(files)}개")
    print()

    results = []
    for i, (true_v, take, path) in enumerate(files, 1):
        audio = load_wav(path)
        try:
            f1, f2, f3 = extract_formants(audio)
            pred, conf = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
        except Exception as e:
            print(f"  [{i:2d}/{len(files)}] {path.name:<20s} ERROR: {e}")
            results.append(dict(true=true_v, path=path.name,
                                f1=None, f2=None, f3=None,
                                pred="?", conf=0.0))
            continue

        mark = "✓" if pred == true_v else "✗"
        f1s = f"{f1:.0f}" if f1 is not None else "—"
        f2s = f"{f2:.0f}" if f2 is not None else "—"
        f3s = f"{f3:.0f}" if f3 is not None else "—"
        print(f"  [{i:2d}/{len(files)}] {path.name:<20s} "
              f"F1={f1s:>5s} F2={f2s:>5s} F3={f3s:>5s}  "
              f"pred={pred}({conf:.2f}) {mark}")
        results.append(dict(true=true_v, path=path.name,
                            f1=f1, f2=f2, f3=f3,
                            pred=pred, conf=conf))

    # ── 통계 ──
    correct = sum(1 for r in results if r["pred"] == r["true"])
    total   = len(results)

    print()
    print("=" * 60)
    print(f"전체 정확도: {correct}/{total} = {correct/total*100:.1f}%")
    print("=" * 60)

    # 모음별 + 오답 분포
    by_v = defaultdict(lambda: {"correct": 0, "total": 0, "errors": Counter()})
    for r in results:
        by_v[r["true"]]["total"] += 1
        if r["pred"] == r["true"]:
            by_v[r["true"]]["correct"] += 1
        else:
            by_v[r["true"]]["errors"][r["pred"]] += 1

    print(f"\n  {'모음':<4} | {'정확도':<11} | 오답 분포")
    print(f"  -----+-------------+-----------------------")
    for v in VOWELS:
        d = by_v[v]
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:3.0f}%)"
        errors_str = ", ".join(f"{p}×{n}"
                                for p, n in d["errors"].most_common()) or "—"
        print(f"  {v:<4} | {acc:<11} | {errors_str}")

    # 혼동 행렬
    cols  = VOWELS + ["?"]
    col_i = {v: i for i, v in enumerate(cols)}
    M = np.zeros((len(VOWELS), len(cols)), dtype=int)
    for r in results:
        i = VOWELS.index(r["true"])
        j = col_i.get(r["pred"], len(VOWELS))
        M[i, j] += 1

    # ── 출력 파일 ──
    L = ["# Simple Baseline 결과", ""]
    L.append("## 구성")
    L.append("")
    L.append("- **추출**: Praat Burg only, ceiling 5500 (단일)")
    L.append("- **분류**: `classify_vowel` (Bark Mahalanobis), `scale=1.0`")
    L.append("- **추가 처리 없음**: ensemble / Kalman / wav2vec2 / F0 정규화 / HNR 게이팅 모두 우회")
    L.append("- 단일 청크 (전체 1.5s) → 중앙 시점 측정")
    L.append("")
    L.append("## 전체 정확도")
    L.append("")
    L.append(f"**{correct}/{total} = {correct/total*100:.1f}%**")
    L.append("")

    L.append("## 모음별 정확도")
    L.append("")
    L.append("| 모음 | 정확도 | 오답 분포 |")
    L.append("|---|---|---|")
    for v in VOWELS:
        d = by_v[v]
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        errors_str = ", ".join(f"{p}×{n}"
                                for p, n in d["errors"].most_common()) or "—"
        L.append(f"| {v} | {acc} | {errors_str} |")
    L.append("")

    L.append("## 기존 시스템 대비")
    L.append("")
    L.append("| 시스템 | 정확도 |")
    L.append("|---|---:|")
    L.append("| 기존 (앙상블 + wav2vec2 + Kalman + 정규화) | 22.9% |")
    L.append(f"| **Simple baseline (Praat 단독 + Mahalanobis)** | **{correct/total*100:.1f}%** |")
    L.append(f"| 차이 | **{(correct/total*100 - 22.9):+.1f} %p** |")
    L.append("")

    L.append("## 혼동 행렬")
    L.append("")
    L.append("rows = 정답, cols = 예측. `?` = 분류 거부")
    L.append("")
    header = "| 정답＼예측 | " + " | ".join(cols) + " |"
    sep    = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    L.append(header)
    L.append(sep)
    for i, v in enumerate(VOWELS):
        row = [str(M[i, j]) if M[i, j] else "·"
               for j in range(len(cols))]
        L.append(f"| **{v}** | " + " | ".join(row) + " |")
    L.append("")

    L.append("## 파일별 상세")
    L.append("")
    L.append("| 파일 | 정답 | F1 | F2 | F3 | 예측 | conf |")
    L.append("|---|---|---:|---:|---:|---|---:|")
    for r in results:
        mark = (f"**{r['pred']}** ✓" if r["pred"] == r["true"]
                else f"{r['pred']} ✗")
        f1 = f"{r['f1']:.0f}" if r["f1"] is not None else "—"
        f2 = f"{r['f2']:.0f}" if r["f2"] is not None else "—"
        f3 = f"{r['f3']:.0f}" if r["f3"] is not None else "—"
        L.append(f"| {r['path']} | {r['true']} | {f1} | {f2} | {f3} "
                 f"| {mark} | {r['conf']:.2f} |")
    L.append("")

    (RESULTS / "baseline_simple.md").write_text(
        "\n".join(L), encoding="utf-8",
    )

    # 혼동 행렬 png
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(VOWELS)))
    ax.set_xticklabels(cols, fontsize=12)
    ax.set_yticklabels(VOWELS, fontsize=12)
    ax.set_xlabel("예측 (Predicted)", fontsize=11)
    ax.set_ylabel("정답 (True)", fontsize=11)
    ax.set_title(f"Simple Baseline ({correct}/{total} = {correct/total*100:.1f}%)",
                 fontsize=13)
    vmax = M.max() if M.max() > 0 else 1
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = int(M[i, j])
            if v == 0:
                continue
            color = "white" if v > vmax * 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    color=color, fontsize=12)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(RESULTS / "baseline_simple_confusion.png", dpi=120)
    plt.close(fig)

    print()
    print("산출물:")
    print(f"  - {RESULTS / 'baseline_simple.md'}")
    print(f"  - {RESULTS / 'baseline_simple_confusion.png'}")


if __name__ == "__main__":
    main()
