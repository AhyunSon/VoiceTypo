"""
evaluation/calibration_v2.py — Layer 1 (다중 발화 cal) + Layer 5 (다중 청크 vote)

목표: 90% 진입 가능성 측정
v1 결과 (단일 발화 cal + 단일 시점): -3.6%p (실패)
v2 가설:
  Layer 1 — 다중 발화 cal 로 *_01 비전형성 + 어 bimodal 흡수
  Layer 5 — 다중 청크 vote 로 단일 시점 측정 노이즈 평준화

4-시나리오 비교 (모두 동일 평가 set):
  A. baseline_single   학계 _REFS, 단일 시점 (현재 production)
  B. baseline_vote     학계 _REFS, 5시간점 confidence-weighted vote
  C. cal_single        user_refs (cal=1,2,3 takes), 단일 시점
  D. cal_vote          user_refs + 5시간점 vote (Layer 1 + Layer 5)

데이터 split:
  cal set:  takes 1, 2, 3 per vowel  (21 wav)
  test set: takes 4, 5    per vowel  (14 wav)

판단 기준:
  D ≥ 85% → 90% 도달 가능 영역 진입, Layer 2/3/4 추가 진행
  D 75~85%  → 효과 확인, 추가 레이어 필요
  D 65~75%  → 단일 화자에서 정체, 다화자 데이터 필요
  D ≤ 65%   → 접근 자체 재검토

실행:
  cd /c/Users/admin/Desktop/realtime_formant
  python -m evaluation.calibration_v2
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
from vowel_classifier import classify_vowel, set_user_refs, clear_user_refs
from calibrator import VowelCalibrator


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"

CAL_TAKES  = (1, 2, 3)            # Layer 1: 다중 발화
TEST_TAKES = (4, 5)
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]  # Layer 5: 5 시간점


# ══════════════════════════════════════════
# 데이터 / 추출
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


def _formant_obj(audio: np.ndarray, sr: int = SAMPLE_RATE):
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
    fmt = snd.to_formant_burg(
        time_step=None,
        max_number_of_formants=5,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50,
    )
    duration = audio.shape[0] / sr
    return fmt, duration


def _get_at(fmt, t: float):
    def _g(n):
        v = fmt.get_value_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)
    return _g(1), _g(2), _g(3)


def extract_center(audio: np.ndarray) -> tuple:
    fmt, dur = _formant_obj(audio)
    return _get_at(fmt, dur / 2)


def extract_multi(audio: np.ndarray, positions: list = None) -> list:
    fmt, dur = _formant_obj(audio)
    return [_get_at(fmt, dur * p) for p in (positions or SAMPLE_POS)]


# ══════════════════════════════════════════
# Layer 1: 다중 발화 캘리브레이션
# ══════════════════════════════════════════

def build_calibration(cal_files: list) -> VowelCalibrator:
    """cal_files (각 모음 takes 1,2,3 = 21 wav) 로 calibrator 구축.

    각 모음 3 wav × 5 시간점 = 15 샘플 누적.
    advance_vowel(validate=False) — 라벨링된 데이터 신뢰.
    """
    cal = VowelCalibrator()
    cal.start()

    print("─" * 60)
    print(f"Layer 1: 다중 발화 캘리브레이션 (takes={CAL_TAKES} → 21 wav)")
    print("─" * 60)

    # 모음별 그룹화
    by_vowel = defaultdict(list)
    for v, t, p in cal_files:
        by_vowel[v].append((t, p))

    for v in VOWELS:
        if cal.current_vowel != v:
            raise RuntimeError(f"순서 오류: 기대 {cal.current_vowel}, got {v}")
        accepted = 0
        f1_all, f2_all = [], []
        for t, path in by_vowel[v]:
            audio = load_wav(path)
            for f1, f2, f3 in extract_multi(audio):
                before = cal.current_sample_count()
                cal.feed_chunk(f1, f2, f3)
                if cal.current_sample_count() > before:
                    accepted += 1
                    if f1 is not None:
                        f1_all.append(f1)
                    if f2 is not None:
                        f2_all.append(f2)
        f1m = float(np.median(f1_all)) if f1_all else 0.0
        f2m = float(np.median(f2_all)) if f2_all else 0.0
        ok, msg = cal.advance_vowel(validate=False)
        mark = "✓" if ok else "✗"
        takes_str = ",".join(str(t) for t, _ in by_vowel[v])
        print(f"  {v}: takes=[{takes_str}] samples={accepted:<3d} "
              f"F1≈{f1m:.0f} F2≈{f2m:.0f}  {mark}")
        if not ok:
            raise RuntimeError(f"calibration fail at {v}: {msg}")

    print()
    print(f"✓ 완료 (is_ready={cal.is_ready})")
    print()
    print("user_refs:")
    for v, ref in cal.user_refs.items():
        print(f"  {v}: F1={ref[0]:.0f}±{ref[1]:.0f}  "
              f"F2={ref[2]:.0f}±{ref[3]:.0f}  "
              f"F3={ref[4]:.0f}±{ref[5]:.0f}")
    print()
    return cal


# ══════════════════════════════════════════
# Layer 5: 다중 청크 confidence-weighted vote
# ══════════════════════════════════════════

def classify_with_vote(audio: np.ndarray) -> tuple:
    """5 시간점 측정 → confidence 합산 vote.

    Returns:
        (vowel: str, confidence_sum: float, n_voters: int)
    """
    samples = extract_multi(audio)
    votes = defaultdict(float)
    n_voters = 0
    for f1, f2, f3 in samples:
        if f1 is None or f2 is None:
            continue
        pred, conf = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
        if pred == "?" or conf <= 0:
            continue
        votes[pred] += conf
        n_voters += 1

    if not votes:
        return "?", 0.0, 0

    best = max(votes, key=votes.get)
    return best, votes[best], n_voters


def classify_single(audio: np.ndarray) -> tuple:
    """단일 시점 (중앙) 측정 → classify_vowel."""
    f1, f2, f3 = extract_center(audio)
    pred, conf = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
    return pred, conf


# ══════════════════════════════════════════
# 평가
# ══════════════════════════════════════════

def evaluate(test_files: list, mode: str, label: str) -> list:
    """
    mode = "single" or "vote"
    """
    print("─" * 60)
    print(f"평가: {label} (mode={mode}, {len(test_files)} wav)")
    print("─" * 60)

    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        try:
            if mode == "vote":
                pred, conf, nv = classify_with_vote(audio)
                meta = f"voters={nv}"
            else:
                pred, conf = classify_single(audio)
                meta = ""
        except Exception as e:
            print(f"  {path.name} ERROR: {e}")
            rows.append(dict(true=v_true, file=path.name, pred="?", conf=0.0))
            continue

        mark = "✓" if pred == v_true else "✗"
        print(f"  {path.name:<14s} pred={pred}({conf:.2f}) {meta:>10s} {mark}")
        rows.append(dict(true=v_true, file=path.name, pred=pred, conf=conf))
    return rows


def summarize(rows: list) -> dict:
    correct = sum(1 for r in rows if r["pred"] == r["true"])
    total = len(rows)
    by_v = defaultdict(lambda: {"correct": 0, "total": 0, "errors": Counter()})
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
    files = collect_files()
    cal_files  = [f for f in files if f[1] in CAL_TAKES]
    test_files = [f for f in files if f[1] in TEST_TAKES]

    print("=" * 60)
    print("Calibration v2 — Layer 1 (다중 발화) + Layer 5 (다중 vote)")
    print("=" * 60)
    print(f"  cal:  {len(cal_files)} wav (takes {CAL_TAKES})")
    print(f"  test: {len(test_files)} wav (takes {TEST_TAKES})")
    print()

    # ── Scenario A: baseline + single shot ──
    clear_user_refs()
    rows_A = evaluate(test_files, "single", "A. baseline_single")
    sum_A  = summarize(rows_A)
    print(f"  → A. baseline_single: {sum_A['correct']}/{sum_A['total']} "
          f"= {sum_A['accuracy']:.1f}%\n")

    # ── Scenario B: baseline + vote ──
    rows_B = evaluate(test_files, "vote", "B. baseline_vote (Layer 5 only)")
    sum_B  = summarize(rows_B)
    print(f"  → B. baseline_vote: {sum_B['correct']}/{sum_B['total']} "
          f"= {sum_B['accuracy']:.1f}%\n")

    # ── Layer 1 캘리브레이션 ──
    cal = build_calibration(cal_files)
    set_user_refs(cal.user_refs)

    # ── Scenario C: cal + single ──
    rows_C = evaluate(test_files, "single", "C. cal_single (Layer 1 only)")
    sum_C  = summarize(rows_C)
    print(f"  → C. cal_single: {sum_C['correct']}/{sum_C['total']} "
          f"= {sum_C['accuracy']:.1f}%\n")

    # ── Scenario D: cal + vote ──
    rows_D = evaluate(test_files, "vote", "D. cal_vote (Layer 1 + Layer 5)")
    sum_D  = summarize(rows_D)
    print(f"  → D. cal_vote: {sum_D['correct']}/{sum_D['total']} "
          f"= {sum_D['accuracy']:.1f}%\n")

    clear_user_refs()

    # ── 종합 비교 ──
    print("=" * 60)
    print("종합 비교")
    print("=" * 60)
    rows = [
        ("A. baseline_single", sum_A),
        ("B. baseline_vote",   sum_B),
        ("C. cal_single",      sum_C),
        ("D. cal_vote",        sum_D),
    ]
    print(f"  {'시나리오':<25s} {'정확도':>10s}  {'vs A':>8s}")
    print(f"  {'-'*25} {'-'*10}  {'-'*8}")
    for name, s in rows:
        delta = s["accuracy"] - sum_A["accuracy"]
        print(f"  {name:<25s} {s['accuracy']:>9.1f}%  {delta:>+7.1f}%p")
    print()

    # ── 모음별 D scenario ──
    print("D scenario 모음별:")
    print(f"  {'모음':<4s} {'정확도':>11s}  오답")
    print(f"  {'-'*4} {'-'*11}  {'-'*30}")
    for v in VOWELS:
        d = sum_D["by_v"].get(v, {"correct": 0, "total": 0, "errors": Counter()})
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        print(f"  {v:<4s} {acc:>11s}  {err}")
    print()

    # ── 판정 ──
    final = sum_D["accuracy"]
    if final >= 85:
        verdict = "✓ 90% 도달 가능 영역 진입. Layer 2/3/4 추가 진행 권장."
    elif final >= 75:
        verdict = "△ 효과 확인. 추가 레이어 필요. Layer 2 (Lobanov) 우선."
    elif final >= 65:
        verdict = "△ 단일 화자 정체. 다화자 데이터 / 학계 코퍼스 필요."
    else:
        verdict = "✗ 접근 자체 재검토 필요."
    print(f"판정: {verdict}")
    print()

    # ── CSV ──
    csv_path = RESULTS / "calibration_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true",
                    "A_pred", "A_conf",
                    "B_pred", "B_conf",
                    "C_pred", "C_conf",
                    "D_pred", "D_conf"])
        by_file = {}
        for tag, rs in [("A", rows_A), ("B", rows_B),
                        ("C", rows_C), ("D", rows_D)]:
            for r in rs:
                by_file.setdefault(r["file"], dict(true=r["true"]))
                by_file[r["file"]][f"{tag}_pred"] = r["pred"]
                by_file[r["file"]][f"{tag}_conf"] = r["conf"]
        for fname in sorted(by_file):
            d = by_file[fname]
            w.writerow([
                fname, d.get("true"),
                d.get("A_pred"), f"{d.get('A_conf', 0):.3f}",
                d.get("B_pred"), f"{d.get('B_conf', 0):.3f}",
                d.get("C_pred"), f"{d.get('C_conf', 0):.3f}",
                d.get("D_pred"), f"{d.get('D_conf', 0):.3f}",
            ])

    # ── MD ──
    md_path = RESULTS / "calibration_v2.md"
    L = ["# 캘리브레이션 v2 — Layer 1 + Layer 5",
         "",
         "**작성**: 2026-05-06",
         "",
         "## 가설",
         "- Layer 1: 다중 발화 cal 로 atypical 첫 녹음 + bimodal 흡수",
         "- Layer 5: 다중 청크 vote 로 단일 시점 측정 노이즈 평준화",
         "",
         "## 데이터",
         f"- cal:  takes {CAL_TAKES} → {len(cal_files)} wav (각 모음 3개)",
         f"- test: takes {TEST_TAKES} → {len(test_files)} wav (각 모음 2개)",
         "",
         "## 결과",
         "",
         "| 시나리오 | 정확도 | vs A |",
         "|---|---:|---:|"]
    for name, s in rows:
        delta = s["accuracy"] - sum_A["accuracy"]
        L.append(f"| {name} | "
                 f"{s['correct']}/{s['total']} = **{s['accuracy']:.1f}%** | "
                 f"{delta:+.1f}%p |")
    L += ["",
          "## D scenario 모음별",
          "",
          "| 모음 | 정확도 | 오답 |",
          "|---|---|---|"]
    for v in VOWELS:
        d = sum_D["by_v"].get(v, {"correct": 0, "total": 0,
                                  "errors": Counter()})
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        L.append(f"| {v} | {acc} | {err} |")

    L += ["",
          "## 판정",
          "",
          verdict,
          "",
          "## user_refs (Layer 1 출력)",
          "",
          "| 모음 | F1 (μ±σ) | F2 (μ±σ) | F3 (μ±σ) |",
          "|---|---|---|---|"]
    for v, ref in cal.user_refs.items():
        L.append(f"| {v} | {ref[0]:.0f}±{ref[1]:.0f} | "
                 f"{ref[2]:.0f}±{ref[3]:.0f} | {ref[4]:.0f}±{ref[5]:.0f} |")
    L.append("")

    md_path.write_text("\n".join(L), encoding="utf-8")

    print("산출물:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
