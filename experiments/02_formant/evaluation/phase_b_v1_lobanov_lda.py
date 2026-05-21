"""
evaluation/phase_b_v1_lobanov_lda.py — Lobanov + LDA 실 데이터 검증

Phase A 결과 한계:
  학계 _REFS 휴리스틱 매칭 → 천장 54.3% (cal-free) / 80% (cal+GMM)

Phase B 가설 (Nature 2023 재현):
  Lobanov 정규화 + LDA 학습 분류기 → 천장 더 높음.
  학습 = 데이터에서 클래스 결정 경계 도출 (휴리스틱 거리 X).

실험:
  실험 1: 본인 35-wav within-speaker 5-fold CV
    - 4 takes 학습, 1 take 테스트, leave-one-take-out
    - 본인 stats 로 Lobanov 정규화
    - LDA 학습 + 평가

  실험 2: 합성 다화자 LOSO
    - 본인 (×1.0) / 가상 남성 (×0.83) / 가상 아동 (×1.20)
    - 한 화자 빼고 학습 → 그 화자 cal (모음당 3 샘플) → 분류

비교:
  Phase A 통합 (VTLN + 휴리스틱): 54.3%
  Phase B v1 (Lobanov + LDA): 측정값 ?
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
from lobanov_lda import LobanovLDA


VOWELS  = ["아", "에", "이", "오", "우", "으", "어"]
HERE    = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
RESULTS = HERE / "results"
SAMPLE_POS = [0.20, 0.35, 0.50, 0.65, 0.80]


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


def _formant_obj(audio):
    audio = audio - np.mean(audio)
    snd = parselmouth.Sound(audio.astype(np.float64),
                            sampling_frequency=float(SAMPLE_RATE))
    fmt = snd.to_formant_burg(
        time_step=None, max_number_of_formants=5,
        maximum_formant=5500, window_length=0.025, pre_emphasis_from=50,
    )
    return fmt, audio.shape[0] / SAMPLE_RATE


def extract_features(audio, mode="static"):
    """포먼트 feature 추출.

    mode:
      "static": F1, F2, F3 중앙 시점 (3D)
      "multi": F1, F2, F3 5 시간점 평균 (3D)
      "dynamic": 5 시간점 평균 + slope (6D)
    """
    fmt, dur = _formant_obj(audio)

    if mode == "static":
        t = dur / 2
        out = []
        for n in [1, 2, 3]:
            v = fmt.get_value_at_time(n, t)
            out.append(float(v) if (v is not None
                                    and not np.isnan(v)) else np.nan)
        return np.array(out)

    # multi-time
    samples_t = []
    for p in SAMPLE_POS:
        t = dur * p
        row = []
        for n in [1, 2, 3]:
            v = fmt.get_value_at_time(n, t)
            row.append(float(v) if (v is not None
                                    and not np.isnan(v)) else np.nan)
        samples_t.append(row)
    samples_t = np.array(samples_t)   # (5, 3)

    if mode == "multi":
        return np.nanmean(samples_t, axis=0)

    if mode == "dynamic":
        # 평균 + slope (linear regression vs SAMPLE_POS)
        means = np.nanmean(samples_t, axis=0)
        slopes = []
        for col in range(3):
            vals = samples_t[:, col]
            mask = ~np.isnan(vals)
            if mask.sum() < 2:
                slopes.append(0.0)
            else:
                p = np.polyfit(np.array(SAMPLE_POS)[mask], vals[mask], 1)
                slopes.append(p[0])
        return np.concatenate([means, slopes])

    raise ValueError(mode)


def transform_features(X, formant_factor=1.0):
    """가상 화자 시뮬: formant 차원만 곱셈 (slope 차원은 비례 X)."""
    X = X.copy()
    if X.shape[1] >= 3:
        X[:, :3] = X[:, :3] * formant_factor
    if X.shape[1] >= 6:
        # slope 도 비례 (시간 변화율도 formant 와 함께 변함)
        X[:, 3:6] = X[:, 3:6] * formant_factor
    return X


# ══════════════════════════════════════════
# 실험 1: 본인 within-speaker 5-fold CV
# ══════════════════════════════════════════

def experiment_within_speaker(files, mode="static"):
    """5-fold leave-one-take-out CV on 본인 35-wav.

    각 fold:
      train = 4 takes (28 wav) — 본인 stats 로 Lobanov 정규화
      test  = 1 take (7 wav)   — 같은 stats 로 정규화
      LDA fit on train, predict test.

    참고: 단일 화자라 stats 학습 = 정규화 = 1 화자만으로.
    """
    print("─" * 60)
    print(f"실험 1: 본인 within-speaker 5-fold CV (feature={mode})")
    print("─" * 60)

    # 모든 wav 의 feature 추출
    all_data = []   # [(vowel, take, feature_vec)]
    for v, t, path in files:
        audio = load_wav(path)
        feat = extract_features(audio, mode)
        if np.any(np.isnan(feat)):
            continue
        all_data.append((v, t, feat))

    print(f"  유효 wav: {len(all_data)}")

    fold_accs = []
    all_preds = []
    for fold_take in [1, 2, 3, 4, 5]:
        train_data = [(v, f) for v, t, f in all_data if t != fold_take]
        test_data  = [(v, f) for v, t, f in all_data if t == fold_take]

        if not test_data:
            continue

        X_tr = np.array([f for _, f in train_data])
        y_tr = np.array([v for v, _ in train_data])
        X_te = np.array([f for _, f in test_data])
        y_te = np.array([v for v, _ in test_data])

        # 단일 화자 → speaker_data dict 의 key 1 개
        model = LobanovLDA()
        model.fit({"본인": (X_tr, y_tr)})
        # test 도 같은 화자 stats 로 정규화 → predict_for_train_speaker
        preds, _ = model.predict_for_train_speaker("본인", X_te)
        acc = float(np.mean(preds == y_te))
        fold_accs.append(acc)
        for p, t in zip(preds, y_te):
            all_preds.append((t, p, fold_take))
        print(f"  fold {fold_take} (test=take {fold_take}): "
              f"{int(np.sum(preds == y_te))}/{len(y_te)} = {acc*100:.1f}%")

    overall = float(np.mean([1 if t == p else 0
                             for t, p, _ in all_preds]))
    print(f"\n  CV 평균: {np.mean(fold_accs)*100:.1f}% ± "
          f"{np.std(fold_accs)*100:.1f}%p")
    print(f"  CV 종합: {overall*100:.1f}% ({len(all_preds)} 샘플)")

    # 모음별
    by_v = defaultdict(lambda: {"correct": 0, "total": 0,
                                "errors": Counter()})
    for t, p, _ in all_preds:
        by_v[t]["total"] += 1
        if t == p:
            by_v[t]["correct"] += 1
        else:
            by_v[t]["errors"][p] += 1

    print(f"\n  모음별:")
    for v in VOWELS:
        d = by_v[v]
        if d["total"] == 0:
            continue
        acc = f"{d['correct']}/{d['total']}"
        err = (", ".join(f"{p}×{n}" for p, n in d["errors"].most_common())
               or "—")
        print(f"    {v}: {acc:>5s}  {err}")

    return overall, all_preds


# ══════════════════════════════════════════
# 실험 2: 합성 다화자 LOSO
# ══════════════════════════════════════════

def experiment_loso_synthetic(files, mode="static"):
    """합성 다화자 LOSO.

    화자 셋:
      "본인"   : 원본
      "남성"  : ×0.83
      "아동"  : ×1.20

    LOSO: 한 화자 빼고 학습, 그 화자 cal (모음당 3 샘플) → 분류.
    """
    print("─" * 60)
    print(f"실험 2: 합성 다화자 LOSO (feature={mode})")
    print("─" * 60)

    # 본인 데이터
    base_X, base_y, base_t = [], [], []
    for v, t, path in files:
        audio = load_wav(path)
        feat = extract_features(audio, mode)
        if np.any(np.isnan(feat)):
            continue
        base_X.append(feat)
        base_y.append(v)
        base_t.append(t)
    base_X = np.array(base_X)
    base_y = np.array(base_y)
    base_t = np.array(base_t)

    speakers = {
        "본인": (base_X, base_y, base_t),
        "남성": (transform_features(base_X, 0.83), base_y, base_t),
        "아동": (transform_features(base_X, 1.20), base_y, base_t),
    }

    for held in speakers.keys():
        train_dict = {k: (X, y) for k, (X, y, _) in speakers.items()
                      if k != held}
        model = LobanovLDA()
        model.fit(train_dict)

        X_held, y_held, t_held = speakers[held]
        # cal: 모음당 3 샘플 (takes 1, 2, 3)
        cal_mask = np.isin(t_held, [1, 2, 3])
        test_mask = ~cal_mask
        X_cal, X_test = X_held[cal_mask], X_held[test_mask]
        y_test = y_held[test_mask]

        preds, _ = model.predict_for_new_speaker(X_cal, X_test)
        acc = float(np.mean(preds == y_test))
        print(f"  held-out={held:<6s}: cal={len(X_cal)} test={len(X_test)} "
              f"acc={int(np.sum(preds == y_test))}/{len(X_test)} "
              f"= {acc*100:.1f}%")
    print()


# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = collect_files()

    print("=" * 60)
    print("Phase B v1 — Lobanov + LDA")
    print("=" * 60)
    print(f"  데이터: {len(files)} wav")
    print()

    feature_modes = ["static", "multi", "dynamic"]
    summary = {}
    all_preds_by_mode = {}

    for mode in feature_modes:
        print(f"\n{'='*60}")
        print(f"FEATURE MODE: {mode}")
        print(f"{'='*60}")
        acc, preds = experiment_within_speaker(files, mode)
        summary[mode + "_within"] = acc
        all_preds_by_mode[mode] = preds
        print()
        experiment_loso_synthetic(files, mode)

    # ── 종합 ──
    print("=" * 60)
    print("종합")
    print("=" * 60)
    print(f"  {'feature':<12s} {'within-speaker 5-fold CV':>25s}")
    for mode in feature_modes:
        acc = summary[mode + "_within"]
        print(f"  {mode:<12s} {acc*100:>24.1f}%")
    print()

    # 비교 baseline
    print("이전 결과 비교:")
    print(f"  Phase A 통합 (cal-free): 22.9~54.3% (화자별)")
    print(f"  Phase v3 (cal+GMM 14-wav split): 78.6%")
    print(f"  Phase v5 (cal+GMM 5-fold CV): 80.0%")
    print()

    best_mode = max(feature_modes, key=lambda m: summary[m + "_within"])
    best_acc = summary[best_mode + "_within"]
    print(f"  최고: feature={best_mode}, accuracy={best_acc*100:.1f}%")
    print()

    # ── MD ──
    md_path = RESULTS / "phase_b_v1_lobanov_lda.md"
    L = ["# Phase B v1 — Lobanov + LDA",
         "",
         "**작성**: 2026-05-06",
         "",
         "## 가설",
         "Nature 2023 (LDA 94% 다화자) 재현 시도.",
         "휴리스틱 매칭 (Phase A) 대신 학습된 분류기 + Lobanov 정규화.",
         "",
         "## 결과 — 본인 within-speaker 5-fold CV",
         "",
         "| Feature 모드 | 정확도 |",
         "|---|---:|"]
    for mode in feature_modes:
        L.append(f"| {mode} | **{summary[mode + '_within']*100:.1f}%** |")
    L += ["",
          "## 비교",
          "",
          "| 방법 | 정확도 |",
          "|---|---:|",
          "| Phase A 통합 (cal-free) | 22.9~54.3% |",
          "| v3 (cal+GMM 14-wav split) | 78.6% |",
          "| v5 (cal+GMM 5-fold CV)    | 80.0% |",
          f"| **Phase B v1 (Lobanov+LDA, best={best_mode})** | "
          f"**{best_acc*100:.1f}%** |",
          "",
          "## 해석",
          "",
          "(채울 자리)",
          ""]
    md_path.write_text("\n".join(L), encoding="utf-8")
    print(f"산출물: {md_path}")


if __name__ == "__main__":
    main()
