"""
lobanov_lda.py — Lobanov 정규화 + LDA 분류 (Nature 2023 재현)

원리:
  Lobanov (1971) z-score 정규화:
    F_norm[v] = (F[v] - μ_speaker) / σ_speaker
    μ_speaker = 그 화자의 모음 발화 F 평균
    σ_speaker = 그 화자의 모음 발화 F 표준편차

  → 모든 화자가 평균 0, SD 1 로 정규화됨
  → 화자간 systematic shift (성도 길이 등) 제거

LDA (Linear Discriminant Analysis):
  - 학습: 정규화된 (X_train, y_train) 으로 모음 클래스 간 분산 최대 / 내 분산 최소
  - 분류: linear projection 후 가장 가까운 클래스 평균
  - Bayes 최적 (가우시안 가정 하)

핵심 차이 vs 우리 이전 시도:
  ✗ 이전: 학계 _REFS 와 Mahalanobis 거리 (휴리스틱 비교)
  ✓ 신규: 정규화된 데이터로 분류기 직접 학습 (statistical learning)

Nature 2023 결과:
  영어 11 모음, 22 화자 학습, 20 화자 테스트
  F1, F2, F3, duration + Lobanov + LDA → 94%
"""

from typing import Optional
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ══════════════════════════════════════════
# Lobanov 정규화
# ══════════════════════════════════════════

def compute_speaker_stats(features: np.ndarray) -> tuple:
    """화자의 모음 발화로부터 (mean, std) 산출.

    Args:
        features: (N, D) — 그 화자의 모든 모음 측정값.
                  N = 발화 수, D = feature 차원.
    Returns:
        (mean: (D,), std: (D,))  — D 차원별 통계.
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0, ddof=1)
    # std 0 방지
    std = np.maximum(std, 1e-6)
    return mean, std


def lobanov_normalize(features: np.ndarray,
                      speaker_mean: np.ndarray,
                      speaker_std: np.ndarray) -> np.ndarray:
    """z-score 정규화. (features - mean) / std."""
    return (features - speaker_mean) / speaker_std


# ══════════════════════════════════════════
# LDA 분류기 wrapper
# ══════════════════════════════════════════

class LobanovLDA:
    """Lobanov-normalized LDA 분류기.

    학습:
      각 화자의 (X_speaker, y_speaker) 입력 → 화자별 정규화 후 학습.
    추론:
      신규 화자: 그 화자의 cal 데이터로 mean/std 산출 → 정규화 → 분류.
    """

    def __init__(self, vowels: list = None):
        self.vowels = vowels or ["아", "에", "이", "오", "우", "으", "어"]
        self.lda: Optional[LinearDiscriminantAnalysis] = None
        self._train_speakers: dict = {}   # speaker_id → (mean, std)

    # ── 학습 ──────────────────────────────────────────────────

    def fit(self, speaker_data: dict) -> None:
        """학습 — 다화자 데이터로 학습.

        Args:
            speaker_data: {speaker_id: (X, y)} dict
                X shape (N, D), y shape (N,) — 모음 라벨.
        """
        X_all, y_all = [], []
        for spkr_id, (X, y) in speaker_data.items():
            X = np.asarray(X, dtype=float)
            y = np.asarray(y)
            mean, std = compute_speaker_stats(X)
            self._train_speakers[spkr_id] = (mean, std)
            X_norm = lobanov_normalize(X, mean, std)
            X_all.append(X_norm)
            y_all.append(y)

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X_all, y_all)

    # ── 추론 (신규 화자) ──────────────────────────────────────

    def predict_for_new_speaker(self,
                                X_cal: np.ndarray,
                                X_test: np.ndarray) -> tuple:
        """신규 화자 — cal 로 stats 추정 → test 분류.

        Args:
            X_cal: 그 화자의 cal 모음 발화 (N_cal, D).
            X_test: 분류할 발화 (N_test, D).
        Returns:
            (predictions: (N_test,), confidences: (N_test,))
        """
        if self.lda is None:
            raise RuntimeError("fit() 먼저 호출해야 함")

        mean, std = compute_speaker_stats(np.asarray(X_cal, dtype=float))
        X_test_norm = lobanov_normalize(np.asarray(X_test, dtype=float),
                                        mean, std)

        preds = self.lda.predict(X_test_norm)
        # confidence = max class prob
        probs = self.lda.predict_proba(X_test_norm)
        confs = probs.max(axis=1)
        return preds, confs

    # ── 학습 화자에 대한 예측 ─────────────────────────────────

    def predict_for_train_speaker(self,
                                  speaker_id,
                                  X: np.ndarray) -> tuple:
        """학습에 포함된 화자 — 저장된 stats 재사용."""
        if speaker_id not in self._train_speakers:
            raise KeyError(f"학습 안 된 화자: {speaker_id}")
        mean, std = self._train_speakers[speaker_id]
        X_norm = lobanov_normalize(np.asarray(X, dtype=float), mean, std)
        preds = self.lda.predict(X_norm)
        probs = self.lda.predict_proba(X_norm)
        confs = probs.max(axis=1)
        return preds, confs


# ══════════════════════════════════════════
# 단위 테스트
# ══════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("lobanov_lda.py 단위 테스트")
    print("=" * 60)

    np.random.seed(0)

    # 시뮬: 3 화자 × 7 모음 × 5 발화 = 105 샘플
    # 학계 모음 평균 (여성 기준)
    vowel_means = {
        "아": (978, 1397, 2600),
        "에": (548, 2125, 2980),
        "이": (352, 2787, 3180),
        "오": (487,  840, 2680),
        "우": (367,  660, 2270),
        "으": (435, 1404, 2720),
        "어": (671, 1212, 2640),
    }
    # 화자별 시프트 (성도 길이 차이)
    speaker_factors = {
        "여성":   1.00,
        "남성":   0.83,
        "아동":   1.20,
    }

    speaker_data = {}
    for spkr, factor in speaker_factors.items():
        X, y = [], []
        for v, (m1, m2, m3) in vowel_means.items():
            for _ in range(5):
                X.append([
                    m1 * factor + np.random.normal(0, 30),
                    m2 * factor + np.random.normal(0, 60),
                    m3 * factor + np.random.normal(0, 80),
                ])
                y.append(v)
        speaker_data[spkr] = (np.array(X), np.array(y))

    # 학습 + 평가 (학습 화자에 대해 — within-speaker)
    print("\n[학습] 3 화자 × 7 모음 × 5 발화 = 105 샘플")
    model = LobanovLDA()
    model.fit(speaker_data)

    print("\n[평가] within-speaker (학습 화자 분류)")
    for spkr in speaker_factors:
        X, y = speaker_data[spkr]
        preds, _ = model.predict_for_train_speaker(spkr, X)
        acc = float(np.mean(preds == y))
        print(f"  {spkr}: {acc*100:.1f}%")

    # leave-one-speaker-out (LOSO) 평가
    # cal = 모음당 1개씩 (7 모음 모두 포함, 공평한 mean/std 추정)
    print("\n[LOSO] 한 화자 빼고 학습 → 그 화자 분류 (cal=각 모음 1개)")
    vowels = list(vowel_means.keys())
    speakers = list(speaker_data.keys())
    for held in speakers:
        train_data = {k: v for k, v in speaker_data.items() if k != held}
        model = LobanovLDA()
        model.fit(train_data)

        X_held, y_held = speaker_data[held]
        # 모음당 1 샘플씩 cal, 나머지 test
        cal_idx, test_idx = [], []
        for v in vowels:
            v_idx = np.where(y_held == v)[0]
            cal_idx.append(v_idx[0])
            test_idx.extend(v_idx[1:].tolist())
        cal_idx = np.array(cal_idx)
        test_idx = np.array(test_idx)

        preds, _ = model.predict_for_new_speaker(
            X_held[cal_idx], X_held[test_idx])
        acc = float(np.mean(preds == y_held[test_idx]))
        print(f"  held-out={held:<6s}: cal={len(cal_idx)} test={len(test_idx)} "
              f"acc={acc*100:.1f}%")

    # 추가 — cal 에 모음당 더 많은 샘플 (3개) 사용 시 효과
    print("\n[LOSO] cal=각 모음 3 샘플 (더 안정 stats)")
    for held in speakers:
        train_data = {k: v for k, v in speaker_data.items() if k != held}
        model = LobanovLDA()
        model.fit(train_data)

        X_held, y_held = speaker_data[held]
        cal_idx, test_idx = [], []
        for v in vowels:
            v_idx = np.where(y_held == v)[0]
            cal_idx.extend(v_idx[:3].tolist())
            test_idx.extend(v_idx[3:].tolist())
        cal_idx = np.array(cal_idx)
        test_idx = np.array(test_idx)

        preds, _ = model.predict_for_new_speaker(
            X_held[cal_idx], X_held[test_idx])
        acc = float(np.mean(preds == y_held[test_idx]))
        print(f"  held-out={held:<6s}: cal={len(cal_idx)} test={len(test_idx)} "
              f"acc={acc*100:.1f}%")

    print("\n✓ 테스트 완료")
