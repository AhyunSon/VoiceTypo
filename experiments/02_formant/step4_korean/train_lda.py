"""
train_lda.py — 화자별 Lobanov 정규화 + LDA 학습

입력: step4_korean/vowel_features.npz
       X, y, spk

출력: step4_korean/lda_korean_multispeaker.pkl
       {"lda": ..., "grand_mean": ..., "grand_std": ...}

검증: speaker-out cross-validation (한 화자 제외하고 학습 → 그 화자로 평가)
"""

import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def speaker_stats(X: np.ndarray, spk: np.ndarray) -> dict:
    """화자별 mean/std 계산."""
    stats = {}
    for s in np.unique(spk):
        mask = spk == s
        m = X[mask].mean(axis=0)
        sd = np.maximum(X[mask].std(axis=0, ddof=1), 1e-6)
        stats[s] = (m, sd)
    return stats


def lobanov_per_speaker(X: np.ndarray, spk: np.ndarray,
                        stats: dict) -> np.ndarray:
    """화자별 z-score 정규화 (제공된 stats 사용)."""
    X_norm = np.empty_like(X)
    for s, (m, sd) in stats.items():
        mask = spk == s
        X_norm[mask] = (X[mask] - m) / sd
    return X_norm


def grand_stats(stats: dict) -> tuple:
    """화자 stats 의 평균 → cal-free 추론용."""
    means = np.array([m for m, _ in stats.values()])
    stds = np.array([sd for _, sd in stats.values()])
    return means.mean(axis=0), stds.mean(axis=0)


def speaker_out_cv(X: np.ndarray, y: np.ndarray,
                   spk: np.ndarray, max_fold: int = 10) -> float:
    """speaker-out cross-validation 정확도.

    학습: 화자별 Lobanov 정규화 + LDA
    테스트 (cal-free): 학습 화자들의 grand mean/std 적용
    """
    speakers = np.unique(spk)
    if len(speakers) > max_fold:
        rng = np.random.default_rng(42)
        speakers = rng.choice(speakers, max_fold, replace=False)

    accs = []
    for test_spk in speakers:
        train_mask = spk != test_spk
        test_mask = spk == test_spk

        # 학습 데이터 화자별 stats
        train_stats = speaker_stats(X[train_mask], spk[train_mask])

        # 학습: 각 화자 본인 stats 로 정규화
        X_train_norm = lobanov_per_speaker(
            X[train_mask], spk[train_mask], train_stats
        )

        # cal-free 추론 시 사용할 grand stats (학습 화자들의 평균)
        g_mean, g_std = grand_stats(train_stats)
        g_std = np.maximum(g_std, 1e-6)

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_norm, y[train_mask])

        # 테스트: cal-free — 학습 grand stats 로 정규화
        X_test_norm = (X[test_mask] - g_mean) / g_std
        pred = lda.predict(X_test_norm)
        acc = (pred == y[test_mask]).mean()
        accs.append(acc)
        print(f"  test {test_spk}: {acc * 100:.1f}%")

    return float(np.mean(accs))


def train_final(X: np.ndarray, y: np.ndarray,
                spk: np.ndarray, out_path: Path) -> None:
    """전체 화자로 LDA 학습 후 저장."""
    stats = speaker_stats(X, spk)
    X_norm = lobanov_per_speaker(X, spk, stats)
    g_mean, g_std = grand_stats(stats)
    g_std = np.maximum(g_std, 1e-6)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_norm, y)

    joblib.dump({
        "lda": lda,
        "grand_mean": g_mean,
        "grand_std": g_std,
        "classes": list(lda.classes_),
        "n_speakers": len(np.unique(spk)),
        "n_samples": len(X),
    }, str(out_path))
    print(f"\n저장: {out_path}")
    print(f"  화자 {len(np.unique(spk))}, 샘플 {len(X)}, 클래스 {list(lda.classes_)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features",
                    default=str(Path(__file__).resolve().parent
                                / "vowel_features.npz"))
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent
                                / "lda_korean_multispeaker.pkl"))
    ap.add_argument("--cv_folds", type=int, default=10)
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"])
    spk = np.asarray(data["spk"])

    print(f"데이터: {len(X)} 샘플 × {X.shape[1]}D, "
          f"{len(np.unique(spk))} 화자, {len(np.unique(y))} 모음")
    print()
    print("Speaker-out CV:")
    cv_acc = speaker_out_cv(X, y, spk, max_fold=args.cv_folds)
    print(f"\n평균: {cv_acc * 100:.1f}%")

    print()
    print("전체 학습 + 저장:")
    train_final(X, y, spk, Path(args.out))
