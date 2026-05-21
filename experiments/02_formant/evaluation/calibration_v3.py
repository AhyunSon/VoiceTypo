"""
evaluation/calibration_v3.py — Layer 4 (GMM) 추가

v2 D 결과: 71.4% (cal + vote, Mahalanobis)
잔존 오답:
  오_04, 오_05 → 우  (오/우 혼동 — 본인 우 발음 F2 비전형)
  우_04 → 오
  어_04 → 으         (어 bimodal closed mode)

Layer 4 가설:
  Mahalanobis (단일 가우시안) 의 한계 → GMM 으로 mode 자동 분리
  - 모음별 BIC 로 k=1 또는 k=2 자동 선택
  - 어/오/우 처럼 bimodal 발음 가능한 모음 → 2-component
  - 단일 모드 모음 → 1-component (과적합 회피)
  - 다이아고날 covariance + reg_covar=1e-3 (15 샘플/모음 안정성)

비교 시나리오:
  D. cal + vote + Mahalanobis (v2 결과 = 71.4%)
  E. cal + single + GMM
  F. cal + vote + GMM (Layer 1 + Layer 4 + Layer 5)

실행:
  cd /c/Users/admin/Desktop/realtime_formant
  python -m evaluation.calibration_v3
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
from vowel_classifier import classify_vowel, set_user_refs, clear_user_refs
from calibrator import VowelCalibrator


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"

CAL_TAKES  = (1, 2, 3)
TEST_TAKES = (4, 5)
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]


# ══════════════════════════════════════════
# 데이터 / 추출 (v2 와 동일)
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


def _formant_obj(audio, sr=SAMPLE_RATE):
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    return fmt, audio.shape[0] / sr


def _get_at(fmt, t):
    def _g(n):
        v = fmt.get_value_at_time(n, t)
        return None if (v is None or np.isnan(v)) else float(v)
    return _g(1), _g(2), _g(3)


def extract_center(audio):
    fmt, dur = _formant_obj(audio)
    return _get_at(fmt, dur / 2)


def extract_multi(audio):
    fmt, dur = _formant_obj(audio)
    return [_get_at(fmt, dur * p) for p in SAMPLE_POS]


# ══════════════════════════════════════════
# Bark 변환
# ══════════════════════════════════════════

def _bark(f):
    f = np.asarray(f, dtype=float)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


# ══════════════════════════════════════════
# Layer 4: GMM 학습 (BIC 자동 선택)
# ══════════════════════════════════════════

def collect_cal_samples(cal_files: list) -> dict:
    """모음별 raw (F1, F2, F3) 샘플 수집."""
    samples = defaultdict(list)
    for v, t, path in cal_files:
        audio = load_wav(path)
        for f1, f2, f3 in extract_multi(audio):
            if f1 is None or f2 is None or f3 is None:
                continue
            samples[v].append((f1, f2, f3))
    return dict(samples)


def fit_gmms(samples: dict, k_max: int = 2) -> dict:
    """모음별 GMM 학습. BIC 로 k=1 또는 k=2 자동 선택.

    Bark 공간에서 학습 (분류 시 동일 공간 사용).
    """
    gmms = {}
    chosen_k = {}
    for v, pts in samples.items():
        arr = np.asarray(pts, dtype=float)
        bark = np.column_stack([_bark(arr[:, 0]),
                                _bark(arr[:, 1]),
                                _bark(arr[:, 2])])

        best_k, best_bic, best_g = 1, float("inf"), None
        for k in range(1, k_max + 1):
            if len(bark) < k * 4:   # k 컴포넌트당 최소 4 샘플
                continue
            try:
                g = GaussianMixture(
                    n_components=k,
                    covariance_type="diag",
                    reg_covar=1e-3,
                    random_state=0,
                    max_iter=200,
                    n_init=3,
                )
                g.fit(bark)
                bic = g.bic(bark)
                if bic < best_bic:
                    best_bic = bic
                    best_k   = k
                    best_g   = g
            except Exception:
                continue

        if best_g is None:
            # fallback: 단일 평균/공분산
            mean = bark.mean(axis=0, keepdims=True)
            g = GaussianMixture(
                n_components=1, covariance_type="diag",
                reg_covar=1e-2, random_state=0, max_iter=10,
            )
            g.means_init = mean
            g.fit(bark)
            best_g = g
            best_k = 1

        gmms[v] = best_g
        chosen_k[v] = best_k
    return gmms, chosen_k


def classify_gmm(f1, f2, f3, gmms) -> tuple:
    """GMM log-likelihood 기반 분류.

    Returns (vowel, confidence)
    confidence = 정규화된 (best_logp − second_logp) ∈ [0, 1]
    """
    if f1 is None or f2 is None or f1 < 100 or f2 < 200:
        return "?", 0.0
    if f3 is None or not (1500 < f3 < 4500):
        # F3 무효 시 평균값 fallback
        f3 = 2700.0

    pt = np.array([[_bark(f1), _bark(f2), _bark(f3)]])
    best_v, best_logp = "?", float("-inf")
    second_logp = float("-inf")
    for v, g in gmms.items():
        try:
            lp = float(g.score(pt))
        except Exception:
            continue
        if lp > best_logp:
            second_logp = best_logp
            best_logp   = lp
            best_v      = v
        elif lp > second_logp:
            second_logp = lp

    if best_v == "?" or second_logp == float("-inf"):
        return best_v, 0.0

    sep = best_logp - second_logp
    conf = float(max(0.0, min(1.0, sep / 5.0)))
    return best_v, conf


def classify_gmm_vote(audio, gmms) -> tuple:
    """5 시간점 측정 → GMM confidence-weighted vote."""
    samples = extract_multi(audio)
    votes = defaultdict(float)
    n_voters = 0
    for f1, f2, f3 in samples:
        if f1 is None or f2 is None:
            continue
        pred, conf = classify_gmm(f1, f2, f3, gmms)
        if pred == "?" or conf <= 0:
            continue
        votes[pred] += conf
        n_voters += 1
    if not votes:
        return "?", 0.0, 0
    best = max(votes, key=votes.get)
    return best, votes[best], n_voters


# ══════════════════════════════════════════
# 평가
# ══════════════════════════════════════════

def evaluate(test_files, mode, label, gmms=None):
    """
    mode: "single" | "vote" — Mahalanobis 사용
    mode: "gmm_single" | "gmm_vote" — GMM 사용 (gmms 인자 필요)
    """
    print("─" * 60)
    print(f"평가: {label} (mode={mode}, {len(test_files)} wav)")
    print("─" * 60)

    rows = []
    for v_true, take, path in test_files:
        audio = load_wav(path)
        try:
            if mode == "vote":
                from collections import defaultdict as dd
                samples = extract_multi(audio)
                votes = dd(float)
                nv = 0
                for f1, f2, f3 in samples:
                    if f1 is None or f2 is None:
                        continue
                    p, c = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
                    if p == "?" or c <= 0:
                        continue
                    votes[p] += c
                    nv += 1
                if votes:
                    pred = max(votes, key=votes.get)
                    conf = votes[pred]
                else:
                    pred, conf = "?", 0.0
                meta = f"voters={nv}"
            elif mode == "gmm_single":
                f1, f2, f3 = extract_center(audio)
                pred, conf = classify_gmm(f1, f2, f3, gmms)
                meta = ""
            elif mode == "gmm_vote":
                pred, conf, nv = classify_gmm_vote(audio, gmms)
                meta = f"voters={nv}"
            else:  # "single"
                f1, f2, f3 = extract_center(audio)
                pred, conf = classify_vowel(f1, f2, "female", f3=f3, scale=1.0)
                meta = ""
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
    print("Calibration v3 — Layer 1 + Layer 5 + Layer 4 (GMM)")
    print("=" * 60)
    print(f"  cal:  {len(cal_files)} wav (takes {CAL_TAKES})")
    print(f"  test: {len(test_files)} wav (takes {TEST_TAKES})")
    print()

    # ── Layer 1: Mahalanobis 캘리브레이션 (v2 와 동일) ──
    cal = VowelCalibrator()
    cal.start()
    print("─" * 60)
    print("Layer 1: 다중 발화 캘리브레이션 (Mahalanobis 용 user_refs)")
    print("─" * 60)
    by_vowel = defaultdict(list)
    for v, t, p in cal_files:
        by_vowel[v].append((t, p))
    for v in VOWELS:
        for t, path in by_vowel[v]:
            audio = load_wav(path)
            for f1, f2, f3 in extract_multi(audio):
                cal.feed_chunk(f1, f2, f3)
        cal.advance_vowel(validate=False)
    print(f"  ✓ user_refs ready: is_ready={cal.is_ready}")
    print()

    # ── Layer 4: GMM 학습 ──
    print("─" * 60)
    print("Layer 4: GMM 학습 (BIC 로 k=1 또는 k=2 자동 선택)")
    print("─" * 60)
    samples = collect_cal_samples(cal_files)
    for v in VOWELS:
        print(f"  {v}: {len(samples[v])} 샘플")
    gmms, chosen_k = fit_gmms(samples, k_max=2)
    print()
    print("  선택된 k:")
    for v in VOWELS:
        means_hz = []
        g = gmms[v]
        # Bark → Hz 역변환은 복잡하므로 means_ Bark 그대로 표시
        for c in range(g.n_components):
            means_hz.append(f"({g.means_[c, 0]:.1f},{g.means_[c, 1]:.1f},"
                            f"{g.means_[c, 2]:.1f})Bark")
        bimodal = "★ BIMODAL" if chosen_k[v] == 2 else ""
        print(f"  {v}: k={chosen_k[v]}  centers={', '.join(means_hz)}  {bimodal}")
    print()

    # ── Scenario D: v2 best (cal + vote + Mahalanobis) ──
    set_user_refs(cal.user_refs)
    rows_D = evaluate(test_files, "vote", "D. cal+vote+Mahalanobis (v2)")
    sum_D  = summarize(rows_D)
    print(f"  → D: {sum_D['correct']}/{sum_D['total']} = {sum_D['accuracy']:.1f}%\n")

    # ── Scenario E: cal + single + GMM ──
    rows_E = evaluate(test_files, "gmm_single", "E. cal+single+GMM (Layer 4 only)",
                      gmms=gmms)
    sum_E  = summarize(rows_E)
    print(f"  → E: {sum_E['correct']}/{sum_E['total']} = {sum_E['accuracy']:.1f}%\n")

    # ── Scenario F: cal + vote + GMM ──
    rows_F = evaluate(test_files, "gmm_vote", "F. cal+vote+GMM (Layer 1+4+5)",
                      gmms=gmms)
    sum_F  = summarize(rows_F)
    print(f"  → F: {sum_F['correct']}/{sum_F['total']} = {sum_F['accuracy']:.1f}%\n")

    clear_user_refs()

    # ── 종합 ──
    print("=" * 60)
    print("종합 비교 (vs v2 baseline_single 57.1%)")
    print("=" * 60)
    rows = [
        ("D. cal+vote+Mahalanobis", sum_D),
        ("E. cal+single+GMM",       sum_E),
        ("F. cal+vote+GMM",         sum_F),
    ]
    print(f"  {'시나리오':<28s} {'정확도':>10s}  {'vs 57.1%':>10s}")
    print(f"  {'-'*28} {'-'*10}  {'-'*10}")
    for name, s in rows:
        delta = s["accuracy"] - 57.1
        print(f"  {name:<28s} {s['accuracy']:>9.1f}%  {delta:>+9.1f}%p")
    print()

    print("F scenario 모음별:")
    print(f"  {'모음':<4s} {'정확도':>11s}  오답")
    print(f"  {'-'*4} {'-'*11}  {'-'*30}")
    for v in VOWELS:
        d = sum_F["by_v"].get(v, {"correct": 0, "total": 0,
                                  "errors": Counter()})
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        print(f"  {v:<4s} {acc:>11s}  {err}")
    print()

    final = sum_F["accuracy"]
    if final >= 85:
        verdict = ("✓ 90% 도달 가능 영역. Layer 3 (확장 feature) 추가 진행, "
                   "또는 라이브 검증으로 cross-speaker 효과 측정.")
    elif final >= 75:
        verdict = "△ 효과 확인. Layer 3 (확장 feature) 또는 Layer 6 (gating) 추가 권장."
    elif final > 71.4:
        verdict = "△ 미세 개선. GMM 단독으론 부족, Layer 3 병행 필요."
    else:
        verdict = "✗ GMM 효과 없음. Layer 3 (F3 weight + bandwidths) 직접 시도."
    print(f"판정: {verdict}\n")

    # ── 산출물 ──
    csv_path = RESULTS / "calibration_v3.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true",
                    "D_pred", "D_conf",
                    "E_pred", "E_conf",
                    "F_pred", "F_conf"])
        by_file = {}
        for tag, rs in [("D", rows_D), ("E", rows_E), ("F", rows_F)]:
            for r in rs:
                by_file.setdefault(r["file"], dict(true=r["true"]))
                by_file[r["file"]][f"{tag}_pred"] = r["pred"]
                by_file[r["file"]][f"{tag}_conf"] = r["conf"]
        for fname in sorted(by_file):
            d = by_file[fname]
            w.writerow([fname, d.get("true"),
                        d.get("D_pred"), f"{d.get('D_conf', 0):.3f}",
                        d.get("E_pred"), f"{d.get('E_conf', 0):.3f}",
                        d.get("F_pred"), f"{d.get('F_conf', 0):.3f}"])

    md_path = RESULTS / "calibration_v3.md"
    L = ["# 캘리브레이션 v3 — Layer 4 (GMM) 추가",
         "",
         "**작성**: 2026-05-06",
         "",
         "## 가설",
         "- Mahalanobis(단일 가우시안) 한계 → GMM 으로 mode 자동 분리",
         "- BIC 로 모음별 k=1 또는 k=2 자동 선택",
         "- 어/오/우 등 bimodal 가능 → 2-component, 단일 모드 → 1-component",
         "",
         "## GMM 컴포넌트 자동 선택",
         "",
         "| 모음 | k | 비고 |",
         "|---|---|---|"]
    for v in VOWELS:
        bim = "BIMODAL" if chosen_k[v] == 2 else "single"
        L.append(f"| {v} | {chosen_k[v]} | {bim} |")

    L += ["",
          "## 결과",
          "",
          "| 시나리오 | 정확도 | vs A 57.1% |",
          "|---|---:|---:|"]
    for name, s in rows:
        delta = s["accuracy"] - 57.1
        L.append(f"| {name} | {s['correct']}/{s['total']} = "
                 f"**{s['accuracy']:.1f}%** | {delta:+.1f}%p |")

    L += ["",
          "## F scenario 모음별",
          "",
          "| 모음 | 정확도 | 오답 |",
          "|---|---|---|"]
    for v in VOWELS:
        d = sum_F["by_v"].get(v, {"correct": 0, "total": 0,
                                  "errors": Counter()})
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']} ({d['correct']/d['total']*100:.0f}%)"
        err = ", ".join(f"{p}×{n}" for p, n in d["errors"].most_common()) or "—"
        L.append(f"| {v} | {acc} | {err} |")

    L += ["", "## 판정", "", verdict, ""]
    md_path.write_text("\n".join(L), encoding="utf-8")

    print("산출물:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    main()
