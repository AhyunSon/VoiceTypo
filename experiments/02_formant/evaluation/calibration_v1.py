"""
evaluation/calibration_v1.py — 화자 self-reference 캘리브레이션 효과 검증

검증 방법:
  - 캘리브레이션 set: *_01.wav × 7 (각 모음 첫 녹음, 실제 사용 시나리오 시뮬)
  - 평가 set:        *_02 ~ *_05.wav × 28 (캘리브레이션과 분리)

  각 wav 에서 5개 시간점(0.20, 0.35, 0.50, 0.65, 0.80) 측정 → calibrator 에 누적.

비교:
  baseline_28 (학계 _REFS, 28-wav)  vs  calibrated_28 (user_refs, 28-wav)

판단 기준:
  +10%p 이상 → Step 3 (UI) 진행
  +5~10%p   → 추가 분석 후 결정
  +0~5%p    → 다른 방향 검토
  ≤0       → Step 1 결과 재검토

실행:
  cd /c/Users/admin/Desktop/realtime_formant
  python -m evaluation.calibration_v1
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

# 캘리브레이션 wav 에서 측정할 시간점 (wav 길이 비율)
CAL_SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]


# ══════════════════════════════════════════
# 데이터 로드 & 포먼트 추출 (baseline_simple 과 동일)
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


def _extract_at(snd: parselmouth.Sound, fmt, t: float):
    def _get(n):
        v = fmt.get_value_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)
    return _get(1), _get(2), _get(3)


def extract_formants_center(audio: np.ndarray, sr: int = SAMPLE_RATE):
    """단일 시점(중앙) 측정. baseline_simple 과 동일."""
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
    fmt = snd.to_formant_burg(
        time_step=None,
        max_number_of_formants=5,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50,
    )
    t = audio.shape[0] / sr / 2
    return _extract_at(snd, fmt, t)


def extract_formants_multi(audio: np.ndarray,
                           positions: list,
                           sr: int = SAMPLE_RATE) -> list:
    """여러 시간점 측정. 캘리브레이션용."""
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
    out = []
    for p in positions:
        t = duration * p
        f1, f2, f3 = _extract_at(snd, fmt, t)
        out.append((f1, f2, f3))
    return out


# ══════════════════════════════════════════
# 캘리브레이션 단계
# ══════════════════════════════════════════

def build_calibration(cal_files: list) -> VowelCalibrator:
    """*_01.wav × 7 로부터 calibrator 구축.

    Args:
        cal_files: [(vowel, take, path), ...] (모두 take=1)
    Returns:
        finalized VowelCalibrator (is_ready=True)
    """
    cal = VowelCalibrator()
    cal.start()

    print("─" * 60)
    print("캘리브레이션 단계 (*_01.wav × 7)")
    print("─" * 60)

    for v_expected, take, path in cal_files:
        if cal.current_vowel != v_expected:
            raise RuntimeError(
                f"캘리브레이션 순서 오류: 기대 {cal.current_vowel}, 파일 {v_expected}")

        audio = load_wav(path)
        samples = extract_formants_multi(audio, CAL_SAMPLE_POS)
        accepted = 0
        for f1, f2, f3 in samples:
            before = cal.current_sample_count()
            cal.feed_chunk(f1, f2, f3)
            if cal.current_sample_count() > before:
                accepted += 1

        # 회귀 테스트: validate=False (라벨링된 wav 신뢰)
        ok, msg = cal.advance_vowel(validate=False)
        f1s = [s[0] for s in samples if s[0] is not None]
        f2s = [s[1] for s in samples if s[1] is not None]
        f1m = float(np.median(f1s)) if f1s else 0.0
        f2m = float(np.median(f2s)) if f2s else 0.0
        mark = "✓" if ok else "✗"
        print(f"  {v_expected} ({path.name}): "
              f"samples={accepted}/{len(samples)}  "
              f"F1≈{f1m:.0f} F2≈{f2m:.0f}  {mark} {msg}")
        if not ok:
            raise RuntimeError(f"캘리브레이션 실패 at {v_expected}: {msg}")

    print()
    print(f"✓ 캘리브레이션 완료 (is_ready={cal.is_ready})")
    print()
    print("user_refs:")
    for v, ref in cal.user_refs.items():
        print(f"  {v}: F1={ref[0]:.0f}±{ref[1]:.0f}  "
              f"F2={ref[2]:.0f}±{ref[3]:.0f}  "
              f"F3={ref[4]:.0f}±{ref[5]:.0f}")
    print()
    return cal


# ══════════════════════════════════════════
# 평가 단계 (baseline_simple 과 동일 단일 시점 측정)
# ══════════════════════════════════════════

def evaluate(test_files: list, label: str) -> list:
    """test_files 분류. set_user_refs 상태에 따라 baseline 또는 calibrated."""
    print("─" * 60)
    print(f"평가: {label}  ({len(test_files)}개)")
    print("─" * 60)

    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        try:
            f1, f2, f3 = extract_formants_center(audio)
            pred, conf = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
        except Exception as e:
            print(f"  {path.name} ERROR: {e}")
            rows.append(dict(true=v_true, take=take, file=path.name,
                             f1=None, f2=None, f3=None,
                             pred="?", conf=0.0))
            continue

        mark = "✓" if pred == v_true else "✗"
        f1s = f"{f1:.0f}" if f1 is not None else "—"
        f2s = f"{f2:.0f}" if f2 is not None else "—"
        f3s = f"{f3:.0f}" if f3 is not None else "—"
        print(f"  {path.name:<14s} F1={f1s:>5s} F2={f2s:>5s} F3={f3s:>5s}  "
              f"pred={pred}({conf:.2f}) {mark}")
        rows.append(dict(true=v_true, take=take, file=path.name,
                         f1=f1, f2=f2, f3=f3, pred=pred, conf=conf))
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
    return dict(correct=correct, total=total, accuracy=correct / total * 100.0,
                by_v=dict(by_v))


# ══════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════

def write_csv(out: Path,
              baseline_rows: list,
              calibrated_rows: list) -> None:
    rows_by_file = {}
    for r in baseline_rows:
        rows_by_file[r["file"]] = dict(file=r["file"], true=r["true"],
                                       f1=r["f1"], f2=r["f2"], f3=r["f3"],
                                       baseline_pred=r["pred"],
                                       baseline_conf=r["conf"])
    for r in calibrated_rows:
        d = rows_by_file.get(r["file"], {})
        d["calibrated_pred"] = r["pred"]
        d["calibrated_conf"] = r["conf"]
        d["delta"] = (1 if r["pred"] == r["true"] else 0) - \
                     (1 if d.get("baseline_pred") == r["true"] else 0)
        rows_by_file[r["file"]] = d

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file", "true", "f1", "f2", "f3",
            "baseline_pred", "baseline_conf",
            "calibrated_pred", "calibrated_conf",
            "delta",
        ])
        w.writeheader()
        for fname in sorted(rows_by_file):
            r = rows_by_file[fname]
            w.writerow({
                k: (f"{v:.3f}" if isinstance(v, float) else v)
                for k, v in r.items()
            })


def write_md(out: Path,
             baseline_summary: dict,
             calibrated_summary: dict,
             cal_user_refs: dict) -> None:
    L = ["# 화자 self-reference 캘리브레이션 v1 (경로 C)",
         "",
         "**작성**: 2026-04-29",
         "",
         "## 검증 구성",
         "",
         "- 캘리브레이션 set: 각 모음 *_01.wav (7개)",
         "- 평가 set:        각 모음 *_02~05.wav (28개)",
         "- 캘리브레이션 시간점: wav 길이의 [0.20, 0.35, 0.50, 0.65, 0.80] (5개)",
         "- 평가 측정: wav 중앙 단일 시점 (baseline_simple 과 동일)",
         "- 분류기: classify_vowel (Bark Mahalanobis), gender=female, scale=1.0",
         "",
         "## 결과 요약",
         ""]
    b_acc = baseline_summary["accuracy"]
    c_acc = calibrated_summary["accuracy"]
    delta = c_acc - b_acc
    L += [
        f"| 조건 | 정확도 |",
        f"|---|---:|",
        f"| baseline_28 (학계 _REFS, 28-wav) | {baseline_summary['correct']}/28 = **{b_acc:.1f}%** |",
        f"| calibrated_28 (user_refs, 28-wav) | {calibrated_summary['correct']}/28 = **{c_acc:.1f}%** |",
        f"| **변화** | **{delta:+.1f} %p** |",
        "",
        "참고: 35-wav 전체 baseline은 54.3%. 본 비교는 캘리브레이션 wav 7개를 제외한 28-wav 동일 평가셋.",
        "",
    ]

    L += ["## 모음별 정확도 비교", ""]
    L.append("| 모음 | baseline | calibrated | 변화 |")
    L.append("|---|---|---|---:|")
    for v in VOWELS:
        bd = baseline_summary["by_v"].get(v, {"correct": 0, "total": 0})
        cd = calibrated_summary["by_v"].get(v, {"correct": 0, "total": 0})
        b_pct = (bd["correct"] / bd["total"] * 100.0) if bd["total"] else 0.0
        c_pct = (cd["correct"] / cd["total"] * 100.0) if cd["total"] else 0.0
        L.append(f"| {v} | {bd['correct']}/{bd['total']} ({b_pct:.0f}%) "
                 f"| {cd['correct']}/{cd['total']} ({c_pct:.0f}%) "
                 f"| {c_pct - b_pct:+.0f} %p |")
    L.append("")

    L += ["## user_refs (calibrator 출력)", "",
          "| 모음 | F1 | F1 SD | F2 | F2 SD | F3 | F3 SD |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for v in VOWELS:
        ref = cal_user_refs[v]
        L.append(f"| {v} | {ref[0]:.0f} | {ref[1]:.0f} | "
                 f"{ref[2]:.0f} | {ref[3]:.0f} | "
                 f"{ref[4]:.0f} | {ref[5]:.0f} |")
    L.append("")

    L += ["## 판단 기준 적용", ""]
    if delta >= 10:
        verdict = "**+10%p 이상 → Step 3 (UI) 진행**"
    elif delta >= 5:
        verdict = "**+5~10%p → 추가 분석 후 결정**"
    elif delta >= 0:
        verdict = "**+0~5%p → 다른 방향 검토 (Nearey 변환 등)**"
    else:
        verdict = "**감소 → Step 1 결과 재검토**"
    L += [verdict, ""]

    L += ["## 한계", "",
          "- 1인 데이터로는 다화자 효과 검증 불가 (사용자 본인 데이터로만 측정)",
          "- 본인 환경 효과는 최대 추정치 (다화자 효과는 별도 검증 필요)",
          "- 캘리브레이션 wav 1개만 사용 → SD floor 의존도 큼",
          ""]

    out.write_text("\n".join(L), encoding="utf-8")


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    cal_files  = [(v, t, p) for (v, t, p) in files if t == 1]
    test_files = [(v, t, p) for (v, t, p) in files if t != 1]

    print("=" * 60)
    print("Calibration v1 — 화자 self-reference (경로 C)")
    print("=" * 60)
    print(f"  캘리브레이션 wav: {len(cal_files)}개")
    print(f"  평가 wav:        {len(test_files)}개")
    print()

    # 1) 캘리브레이션
    cal = build_calibration(cal_files)

    # 2) baseline 평가 (user_refs 미적용)
    clear_user_refs()
    baseline_rows = evaluate(test_files, label="baseline_28 (학계 _REFS)")
    baseline_summary = summarize(baseline_rows)
    print()
    print(f"baseline_28: {baseline_summary['correct']}/{baseline_summary['total']} "
          f"= {baseline_summary['accuracy']:.1f}%")
    print()

    # 3) calibrated 평가 (user_refs 적용)
    set_user_refs(cal.user_refs)
    calibrated_rows = evaluate(test_files, label="calibrated_28 (user_refs)")
    calibrated_summary = summarize(calibrated_rows)
    print()
    print(f"calibrated_28: {calibrated_summary['correct']}/{calibrated_summary['total']} "
          f"= {calibrated_summary['accuracy']:.1f}%")
    delta = calibrated_summary["accuracy"] - baseline_summary["accuracy"]
    print(f"변화: {delta:+.1f} %p")
    print()
    clear_user_refs()

    # 4) 산출물
    csv_path = RESULTS / "calibration_v1.csv"
    md_path  = RESULTS / "calibration_v1.md"
    write_csv(csv_path, baseline_rows, calibrated_rows)
    write_md(md_path, baseline_summary, calibrated_summary, cal.user_refs)
    print("산출물:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
