"""
evaluation/calibration_v5_cv.py — 5-fold leave-one-take-out CV

v4 D (GMM k≤2 strict BIC + 4D bandwidth) 가 14-wav 평가에서 85.7%.
14-wav 는 표본 작아 노이즈 큰 결과 — 5-fold CV 로 검증.

Fold k (k=1..5):
  cal:  takes [1..5] except k → 28 wav (각 모음 4 takes)
  test: take k                → 7 wav  (각 모음 1 take)

총 35 예측 (각 wav 가 정확히 한 번씩 test 에 등장).

이게 **단일 화자, 다양한 take split 평균** 의 진짜 성능.
"""

import csv
import sys
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


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]
ALL_TAKES = [1, 2, 3, 4, 5]


# ── 데이터 / 추출 ─────────────────────────────────────────

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
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(SAMPLE_RATE))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    return fmt, audio.shape[0] / SAMPLE_RATE


def extract_multi_full(audio):
    fmt, dur = _formant_obj(audio)
    out = []
    for p in SAMPLE_POS:
        t = dur * p
        row = []
        for n in [1, 2, 3]:
            v = fmt.get_value_at_time(n, t)
            row.append(None if (v is None or np.isnan(v)) else float(v))
        for n in [1, 2, 3]:
            v = fmt.get_bandwidth_at_time(n, t)
            row.append(None if (v is None or np.isnan(v)) else float(v))
        out.append(tuple(row))
    return out


def _bark(f):
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def to_feature_4d(arr):
    """arr: (N, 6)  [F1, F2, F3, B1, B2, B3]"""
    return np.column_stack([
        _bark(arr[:, 0]), _bark(arr[:, 1]), _bark(arr[:, 2]),
        np.log(np.clip(arr[:, 3] / arr[:, 0], 0.05, 5.0)),
    ])


# ── GMM 학습 (v4 Track D) ─────────────────────────────────

def fit_gmms(samples_dict):
    gmms = {}
    chosen_k = {}
    for v, pts in samples_dict.items():
        arr = np.asarray(pts, dtype=float)
        feat = to_feature_4d(arr)
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
                if best_g is None or bic < best_bic - 4:
                    best_bic = bic
                    best_k   = k
                    best_g   = g
            except Exception:
                continue
        gmms[v] = best_g
        chosen_k[v] = best_k
    return gmms, chosen_k


def classify(features, gmms):
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
    return best_v, float(max(0.0, min(1.0, (best_lp - second) / 5.0)))


def vote(audio, gmms):
    samples = extract_multi_full(audio)
    votes = defaultdict(float)
    nv = 0
    for full in samples:
        f1, f2, f3, b1, b2, b3 = full
        if any(x is None for x in [f1, f2, f3, b1, b2, b3]):
            continue
        arr = np.array([[f1, f2, f3, b1, b2, b3]])
        feat = to_feature_4d(arr)[0]
        p, c = classify(feat, gmms)
        if p == "?" or c <= 0:
            continue
        votes[p] += c
        nv += 1
    if not votes:
        return "?", 0.0, 0
    best = max(votes, key=votes.get)
    return best, votes[best], nv


# ── 5-fold CV ─────────────────────────────────────────────

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = collect_files()

    print("=" * 60)
    print("Calibration v5 — 5-fold leave-one-take-out CV")
    print("=" * 60)
    print(f"  총 wav: {len(files)}")
    print(f"  fold 별: cal=28 wav, test=7 wav")
    print()

    all_preds = []                    # (true, pred, fold)
    fold_acc = []
    fold_chosen_k = []

    for fold_take in ALL_TAKES:
        cal_files  = [f for f in files if f[1] != fold_take]
        test_files = [f for f in files if f[1] == fold_take]

        # 캘리브레이션 샘플 수집
        cal_samples = defaultdict(list)
        for v, t, path in cal_files:
            audio = load_wav(path)
            for full in extract_multi_full(audio):
                f1, f2, f3, b1, b2, b3 = full
                if any(x is None for x in [f1, f2, f3, b1, b2, b3]):
                    continue
                cal_samples[v].append((f1, f2, f3, b1, b2, b3))

        gmms, chosen_k = fit_gmms(cal_samples)
        fold_chosen_k.append(chosen_k)

        # 평가
        correct = 0
        print(f"─ Fold {fold_take} (test=takes {fold_take}, cal=20 samples/모음) ─")
        ks = " ".join(f"{v}={chosen_k[v]}" for v in VOWELS)
        print(f"  k: {ks}")
        for v_true, take, path in test_files:
            audio = load_wav(path)
            pred, conf, nv = vote(audio, gmms)
            mark = "✓" if pred == v_true else "✗"
            print(f"  {path.name:<14s} pred={pred}({conf:.2f}) "
                  f"voters={nv}  {mark}")
            all_preds.append((v_true, pred, fold_take))
            if pred == v_true:
                correct += 1
        acc = correct / len(test_files) * 100.0
        fold_acc.append(acc)
        print(f"  → fold {fold_take}: {correct}/7 = {acc:.1f}%\n")

    # ── 종합 ──
    print("=" * 60)
    print("CV 종합")
    print("=" * 60)
    total_correct = sum(1 for tr, pr, _ in all_preds if tr == pr)
    total_n       = len(all_preds)
    cv_acc        = total_correct / total_n * 100.0

    print(f"  Fold 별: {[f'{a:.1f}%' for a in fold_acc]}")
    print(f"  평균:    {np.mean(fold_acc):.1f}% ± {np.std(fold_acc):.1f}%p")
    print(f"  종합:    {total_correct}/{total_n} = {cv_acc:.1f}%")
    print()

    # 모음별
    by_v = defaultdict(lambda: {"correct": 0, "total": 0, "errors": Counter()})
    for tr, pr, _ in all_preds:
        by_v[tr]["total"] += 1
        if tr == pr:
            by_v[tr]["correct"] += 1
        else:
            by_v[tr]["errors"][pr] += 1

    print("모음별 (5 takes 합산):")
    print(f"  {'모음':<4s} {'정확도':>11s}  오답")
    print(f"  {'-'*4} {'-'*11}  {'-'*30}")
    for v in VOWELS:
        d = by_v[v]
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        print(f"  {v:<4s} {acc:>11s}  {err}")
    print()

    # k 선택 빈도
    print("k 선택 (5 fold 별):")
    for v in VOWELS:
        ks = [fc[v] for fc in fold_chosen_k]
        bimodal_ratio = sum(1 for k in ks if k == 2) / 5
        print(f"  {v}: {ks}  bimodal 빈도={bimodal_ratio*100:.0f}%")
    print()

    # 판정
    if cv_acc >= 85:
        verdict = "✓ 단일 화자 5-fold CV 85%+ 달성. 다음: 라이브 다화자 검증."
    elif cv_acc >= 75:
        verdict = ("△ 단일 화자 75%+ 안정. v4 14-wav 결과의 일부는 split 노이즈, "
                   "그러나 의미 있는 효과 확인.")
    else:
        verdict = "✗ CV 평균이 14-wav 단일 split 보다 크게 낮음. 과적합 의심."
    print(f"판정: {verdict}\n")

    # MD
    md_path = RESULTS / "calibration_v5_cv.md"
    L = ["# 캘리브레이션 v5 — 5-fold leave-one-take-out CV",
         "",
         "**작성**: 2026-05-06",
         "",
         "## 검증 동기",
         "v4 D (GMM k≤2 BIC + 4D bandwidth) 가 14-wav 단일 split 에서 85.7%.",
         "표본 작아 split 노이즈 가능 — 5-fold CV 로 진짜 성능 측정.",
         "",
         "## CV 결과",
         "",
         f"- Fold 별: {fold_acc}",
         f"- 평균:    **{np.mean(fold_acc):.1f}% ± {np.std(fold_acc):.1f}%p**",
         f"- 종합:    {total_correct}/{total_n} = **{cv_acc:.1f}%**",
         "",
         "## 모음별",
         "",
         "| 모음 | 정확도 | 오답 |",
         "|---|---|---|"]
    for v in VOWELS:
        d = by_v[v]
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        L.append(f"| {v} | {acc} | {err} |")

    L += ["",
          "## k 선택 안정성",
          "",
          "| 모음 | Fold 별 k | bimodal 빈도 |",
          "|---|---|---:|"]
    for v in VOWELS:
        ks = [fc[v] for fc in fold_chosen_k]
        bm = sum(1 for k in ks if k == 2) / 5 * 100
        L.append(f"| {v} | {ks} | {bm:.0f}% |")

    L += ["", "## 판정", "", verdict, ""]
    md_path.write_text("\n".join(L), encoding="utf-8")

    print(f"산출물: {md_path}")


if __name__ == "__main__":
    main()
