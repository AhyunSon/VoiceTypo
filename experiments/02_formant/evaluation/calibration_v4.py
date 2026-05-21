"""
evaluation/calibration_v4.py — Layer 3 (확장 feature) 시도

v3 결과: 78.6% (cal + vote + GMM, Layer 1+4+5)
잔존 오답:
  오_04, 오_05 → 우  (오/우 혼동, 본인 오 takes 4,5 의 F2 비전형)
  으_04 → 우         (GMM 으 컴포넌트 과적합 의심)

v3 GMM 의심: 모든 모음 k=2 선택 (BIC) — 15 샘플 + 측정 노이즈 → 과적합

Layer 3 가설:
  3D (F1/F2/F3) → 더 풍부한 feature 로 오/우 분리 개선
  feature 후보:
    a) F3 가중치 증가 (Mahalanobis): 0.30 → sweep
    b) Bandwidth 추가: B1, B2, B3 (Praat 반환값)
    c) F2-F1 거리: 입 후방화 정도

테스트 설계 (3 트랙 병렬):
  Track A: Mahalanobis + F3 weight sweep (가장 단순, 즉시)
  Track B: GMM (k=1 강제) — 과적합 vs Mahalanobis 베이스라인 측정
  Track C: GMM + bandwidth 추가 (4D)

비교 (모두 cal + vote 사용):
  D₀ Mahalanobis F3=0.30  (v2 D = 71.4%)
  G₁..₅ Mahalanobis F3 sweep
  H GMM k=1 강제
  I GMM + bandwidth (4D)

실행:
  cd /c/Users/admin/Desktop/realtime_formant
  python -m evaluation.calibration_v4
"""

import csv
import sys
import importlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
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
import vowel_classifier
from vowel_classifier import (classify_vowel, set_user_refs,
                              clear_user_refs)
from calibrator import VowelCalibrator


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"

CAL_TAKES  = (1, 2, 3)
TEST_TAKES = (4, 5)
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]


# ══════════════════════════════════════════
# 데이터/추출 + bandwidth
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


def _formant_obj(audio, sr=SAMPLE_RATE):
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    return fmt, audio.shape[0] / sr


def _get_full(fmt, t):
    """F1, F2, F3, B1, B2, B3 반환."""
    out = []
    for n in [1, 2, 3]:
        v = fmt.get_value_at_time(n, t)
        out.append(None if (v is None or np.isnan(v)) else float(v))
    for n in [1, 2, 3]:
        v = fmt.get_bandwidth_at_time(n, t)
        out.append(None if (v is None or np.isnan(v)) else float(v))
    return tuple(out)  # (F1, F2, F3, B1, B2, B3)


def extract_center_full(audio):
    fmt, dur = _formant_obj(audio)
    return _get_full(fmt, dur / 2)


def extract_multi_full(audio):
    fmt, dur = _formant_obj(audio)
    return [_get_full(fmt, dur * p) for p in SAMPLE_POS]


# ══════════════════════════════════════════
# Track A: Mahalanobis F3 weight sweep
# ══════════════════════════════════════════

def vote_with_mahalanobis(audio, gender="female"):
    """기존 classify_vowel + vote. _W_F3 변경 후 호출 시 효과 반영."""
    samples = extract_multi_full(audio)
    votes = defaultdict(float)
    nv = 0
    for f1, f2, f3, b1, b2, b3 in samples:
        if f1 is None or f2 is None:
            continue
        p, c = classify_vowel(f1, f2, gender, f3=f3, scale=1.0)
        if p == "?" or c <= 0:
            continue
        votes[p] += c
        nv += 1
    if not votes:
        return "?", 0.0, 0
    best = max(votes, key=votes.get)
    return best, votes[best], nv


def sweep_f3_weight(test_files, weights):
    """weights: list of W_F3 candidates. 각 weight 로 cal+vote 정확도 측정."""
    results = {}
    original = vowel_classifier._W_F3
    for w in weights:
        vowel_classifier._W_F3 = w
        correct = 0
        for v_true, take, path in test_files:
            audio = load_wav(path)
            pred, _, _ = vote_with_mahalanobis(audio)
            if pred == v_true:
                correct += 1
        results[w] = correct
    vowel_classifier._W_F3 = original
    return results


# ══════════════════════════════════════════
# Bark 변환
# ══════════════════════════════════════════

def _bark(f):
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


# ══════════════════════════════════════════
# Track B: GMM k=1 강제
# Track C: GMM + bandwidth (4D)
# ══════════════════════════════════════════

def collect_cal_samples_full(cal_files, include_bw=False):
    """모음별 raw 샘플. include_bw=True 면 (F1,F2,F3,B1,B2,B3)."""
    samples = defaultdict(list)
    for v, t, path in cal_files:
        audio = load_wav(path)
        for full in extract_multi_full(audio):
            f1, f2, f3, b1, b2, b3 = full
            if f1 is None or f2 is None or f3 is None:
                continue
            if include_bw and (b1 is None or b2 is None or b3 is None):
                continue
            if include_bw:
                samples[v].append((f1, f2, f3, b1, b2, b3))
            else:
                samples[v].append((f1, f2, f3))
    return dict(samples)


def to_feature_3d(arr):
    return np.column_stack([_bark(arr[:, 0]), _bark(arr[:, 1]),
                            _bark(arr[:, 2])])


def to_feature_4d_bw(arr):
    """4D: Bark F1/F2/F3 + log(B1/F1) (대역폭/포먼트 비율의 log)."""
    return np.column_stack([
        _bark(arr[:, 0]), _bark(arr[:, 1]), _bark(arr[:, 2]),
        np.log(np.clip(arr[:, 3] / arr[:, 0], 0.05, 5.0)),
    ])


def fit_gmm_fixed_k(samples, feat_fn, k=1):
    gmms = {}
    for v, pts in samples.items():
        arr = np.asarray(pts, dtype=float)
        feat = feat_fn(arr)
        n_comp = min(k, max(1, len(feat) // 4))
        g = GaussianMixture(
            n_components=n_comp, covariance_type="diag",
            reg_covar=1e-3, random_state=0, max_iter=200, n_init=3,
        )
        g.fit(feat)
        gmms[v] = g
    return gmms


def classify_gmm_full(features, gmms):
    """features: 1D array. gmms: dict[vowel→GMM]."""
    pt = features.reshape(1, -1)
    best_v, best_lp = "?", float("-inf")
    second = float("-inf")
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
    sep = best_lp - second
    return best_v, float(max(0.0, min(1.0, sep / 5.0)))


def vote_gmm(audio, gmms, feat_fn, include_bw=False):
    samples = extract_multi_full(audio)
    votes = defaultdict(float)
    nv = 0
    for full in samples:
        f1, f2, f3, b1, b2, b3 = full
        if f1 is None or f2 is None or f3 is None:
            continue
        if include_bw and (b1 is None or b2 is None or b3 is None):
            continue
        if include_bw:
            arr = np.array([[f1, f2, f3, b1, b2, b3]])
        else:
            arr = np.array([[f1, f2, f3]])
        feat = feat_fn(arr)[0]
        p, c = classify_gmm_full(feat, gmms)
        if p == "?" or c <= 0:
            continue
        votes[p] += c
        nv += 1
    if not votes:
        return "?", 0.0, 0
    best = max(votes, key=votes.get)
    return best, votes[best], nv


# ══════════════════════════════════════════
# 평가
# ══════════════════════════════════════════

def evaluate_track(test_files, vote_fn, label):
    print("─" * 60)
    print(f"평가: {label} ({len(test_files)} wav)")
    print("─" * 60)
    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        try:
            pred, conf, nv = vote_fn(audio)
            meta = f"voters={nv}"
        except Exception as e:
            print(f"  {path.name} ERROR: {e}")
            rows.append(dict(true=v_true, file=path.name, pred="?", conf=0.0))
            continue
        mark = "✓" if pred == v_true else "✗"
        print(f"  {path.name:<14s} pred={pred}({conf:.2f}) {meta:>10s} {mark}")
        rows.append(dict(true=v_true, file=path.name, pred=pred, conf=conf))
    return rows


def summarize(rows):
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
    print("Calibration v4 — Layer 3 (확장 feature) 시도")
    print("=" * 60)
    print(f"  cal:  {len(cal_files)} wav")
    print(f"  test: {len(test_files)} wav")
    print()

    # ── Mahalanobis 캘리브레이션 ──
    cal = VowelCalibrator()
    cal.start()
    by_vowel = defaultdict(list)
    for v, t, p in cal_files:
        by_vowel[v].append((t, p))
    for v in VOWELS:
        for t, path in by_vowel[v]:
            audio = load_wav(path)
            for full in extract_multi_full(audio):
                f1, f2, f3 = full[0], full[1], full[2]
                cal.feed_chunk(f1, f2, f3)
        cal.advance_vowel(validate=False)
    set_user_refs(cal.user_refs)

    all_results = {}

    # ── Track A: F3 weight sweep ──
    print("─" * 60)
    print("Track A: Mahalanobis F3 weight sweep (cal + vote)")
    print("─" * 60)
    weights = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    sweep = sweep_f3_weight(test_files, weights)
    print(f"  {'W_F3':>5s}  {'정확도':>11s}")
    print(f"  {'-'*5}  {'-'*11}")
    for w in weights:
        c = sweep[w]
        print(f"  {w:>5.2f}  {c:>2d}/14 ({c/14*100:5.1f}%)")
    best_w = max(sweep, key=sweep.get)
    best_corr = sweep[best_w]
    print(f"  → 최적 W_F3 = {best_w} ({best_corr}/14 = {best_corr/14*100:.1f}%)\n")

    # 최적 W_F3 로 상세 출력
    vowel_classifier._W_F3 = best_w
    rows_A = evaluate_track(test_files,
                            vote_with_mahalanobis,
                            f"A. Mahalanobis (W_F3={best_w})")
    sum_A = summarize(rows_A)
    print(f"  → A 모음별: ", end="")
    for v in VOWELS:
        d = sum_A["by_v"].get(v, {"correct": 0, "total": 0})
        print(f"{v}={d['correct']}/{d['total']} ", end="")
    print(f"\n  → A: {sum_A['accuracy']:.1f}%\n")
    vowel_classifier._W_F3 = 0.30  # restore
    all_results["A"] = sum_A

    # ── Track B: GMM k=1 forced ──
    print("─" * 60)
    print("Track B: GMM k=1 강제 (3D Bark, 과적합 방지)")
    print("─" * 60)
    samples_3d = collect_cal_samples_full(cal_files, include_bw=False)
    gmms_b = fit_gmm_fixed_k(samples_3d, to_feature_3d, k=1)
    rows_B = evaluate_track(
        test_files,
        lambda a: vote_gmm(a, gmms_b, to_feature_3d, include_bw=False),
        "B. GMM k=1 (3D)")
    sum_B = summarize(rows_B)
    print(f"  → B: {sum_B['accuracy']:.1f}%\n")
    all_results["B"] = sum_B

    # ── Track C: GMM + bandwidth (4D) ──
    print("─" * 60)
    print("Track C: GMM k=1 + log(B1/F1) (4D)")
    print("─" * 60)
    samples_4d = collect_cal_samples_full(cal_files, include_bw=True)
    counts_4d = {v: len(s) for v, s in samples_4d.items()}
    print(f"  4D 샘플 수: {counts_4d}")
    gmms_c = fit_gmm_fixed_k(samples_4d, to_feature_4d_bw, k=1)
    rows_C = evaluate_track(
        test_files,
        lambda a: vote_gmm(a, gmms_c, to_feature_4d_bw, include_bw=True),
        "C. GMM k=1 + B1/F1 (4D)")
    sum_C = summarize(rows_C)
    print(f"  → C: {sum_C['accuracy']:.1f}%\n")
    all_results["C"] = sum_C

    # ── Track D: GMM k=2 BIC + 4D ──
    print("─" * 60)
    print("Track D: GMM k=2 BIC 자동 + 4D bandwidth")
    print("─" * 60)
    gmms_d = {}
    for v, pts in samples_4d.items():
        arr = np.asarray(pts, dtype=float)
        feat = to_feature_4d_bw(arr)
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
                if bic < best_bic - 4:    # 엄격 기준 (delta_BIC > 4)
                    best_bic = bic
                    best_k   = k
                    best_g   = g
                elif best_g is None:
                    best_bic = bic
                    best_g   = g
                    best_k   = k
            except Exception:
                continue
        gmms_d[v] = best_g
        print(f"  {v}: k={best_k}")
    print()
    rows_D = evaluate_track(
        test_files,
        lambda a: vote_gmm(a, gmms_d, to_feature_4d_bw, include_bw=True),
        "D. GMM k≤2 BIC strict + 4D")
    sum_D = summarize(rows_D)
    print(f"  → D: {sum_D['accuracy']:.1f}%\n")
    all_results["D"] = sum_D

    clear_user_refs()

    # ── 종합 ──
    print("=" * 60)
    print(f"종합 비교 (vs v3 F: 78.6% / v2 D: 71.4%)")
    print("=" * 60)
    refs = {"A": "Mahalanobis F3-tuned", "B": "GMM k=1 (3D)",
            "C": "GMM k=1 + bandwidth (4D)",
            "D": "GMM k≤2 BIC strict + bandwidth"}
    print(f"  {'시나리오':<35s} {'정확도':>10s}  {'vs v3 F':>10s}")
    print(f"  {'-'*35} {'-'*10}  {'-'*10}")
    for tag, s in all_results.items():
        delta = s["accuracy"] - 78.6
        print(f"  {tag}. {refs[tag]:<32s} {s['accuracy']:>9.1f}%  "
              f"{delta:>+9.1f}%p")
    print()

    best_tag = max(all_results, key=lambda k: all_results[k]["accuracy"])
    best_sum = all_results[best_tag]
    print(f"최고: {best_tag}. {refs[best_tag]} = {best_sum['accuracy']:.1f}%")
    print()
    print(f"{best_tag} scenario 모음별:")
    print(f"  {'모음':<4s} {'정확도':>11s}  오답")
    print(f"  {'-'*4} {'-'*11}  {'-'*30}")
    for v in VOWELS:
        d = best_sum["by_v"].get(v, {"correct": 0, "total": 0,
                                     "errors": Counter()})
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        print(f"  {v:<4s} {acc:>11s}  {err}")
    print()

    # ── 산출물 ──
    md_path = RESULTS / "calibration_v4.md"
    L = ["# 캘리브레이션 v4 — Layer 3 (확장 feature)",
         "",
         "**작성**: 2026-05-06",
         "",
         "## v3 의심: GMM k=2 모든 모음 → 과적합 신호",
         "v4 가설: 더 풍부한 feature + GMM k=1 강제 / strict BIC 로 안정성 회복",
         "",
         "## F3 weight sweep (Track A)",
         "",
         "| W_F3 | 정확도 |",
         "|---|---|"]
    for w in weights:
        c = sweep[w]
        L.append(f"| {w:.2f} | {c}/14 ({c/14*100:.1f}%) |")
    L.append(f"\n→ 최적 W_F3 = {best_w}\n")

    L += ["## 종합 비교", "",
          "| 시나리오 | 정확도 | vs v3 F (78.6%) |",
          "|---|---:|---:|"]
    for tag, s in all_results.items():
        delta = s["accuracy"] - 78.6
        L.append(f"| {tag}. {refs[tag]} | "
                 f"{s['correct']}/{s['total']} = **{s['accuracy']:.1f}%** | "
                 f"{delta:+.1f}%p |")
    L.append("")
    L.append(f"**최고**: {best_tag}. {refs[best_tag]} = "
             f"{best_sum['accuracy']:.1f}%")
    L.append("")
    L += [f"## {best_tag} 모음별",
          "",
          "| 모음 | 정확도 | 오답 |",
          "|---|---|---|"]
    for v in VOWELS:
        d = best_sum["by_v"].get(v, {"correct": 0, "total": 0,
                                     "errors": Counter()})
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        L.append(f"| {v} | {acc} | {err} |")
    md_path.write_text("\n".join(L), encoding="utf-8")

    print("산출물:")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
