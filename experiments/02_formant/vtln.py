"""
vtln.py — Vocal Tract Length Normalization (Phase A1)

원리:
  성도 공명 주파수 ∝ 1 / 성도 길이.
  성도 길이는 화자별로 다름 (남: ~17cm, 여: ~14cm, 아동: ~12cm).
  같은 모음이라도 화자별 F1/F2/F3 위치가 다른 주된 이유.

알고리즘:
  α = F3_canonical / F3_observed
  warped_F = F * α
  → 모든 화자가 동일 캐노니컬 스케일로 정규화 → 화자 무관 분류 가능.

캐노니컬 F3 = 2900 Hz (Yoon 2015 + Hillenbrand 1995 기반 성인 남녀 평균).

Cal-free 자동 정규화:
  사용자 발화 일부에서 F3 통계 누적 → α 추정 → 적용.
  EMA 로 점진 갱신 (실시간 사용).
"""

from typing import Optional
import numpy as np


# ══════════════════════════════════════════
# 상수
# ══════════════════════════════════════════

# 캐노니컬 F3 (Hz) — 성인 평균. 모든 화자를 이 스케일로 정규화.
# 근거: Yoon 2015 여성 F3 평균 ~3100, 남성 ~2600 → 평균 2900.
F3_CANONICAL = 2900.0

# F3 측정 유효 범위 (Hz). 범위 밖이면 무시.
F3_VALID_MIN = 1500.0
F3_VALID_MAX = 4500.0

# 워핑 계수 안전 범위. 1.0 = no warp. 0.7 ~ 1.4 가 사람 성도 변동 범위.
ALPHA_MIN = 0.7
ALPHA_MAX = 1.4

# EMA α (online 적응 속도).
EMA_ALPHA = 0.1

# Online 모드에서 α 산출 전 최소 샘플 수.
MIN_SAMPLES = 5


# ══════════════════════════════════════════
# 즉시 추정 (offline)
# ══════════════════════════════════════════

def compute_warping_factor(f3_observed: float,
                           f3_canonical: float = F3_CANONICAL) -> float:
    """단일 F3 측정값으로 워핑 계수 산출.

    Args:
        f3_observed: 화자 F3 (Hz, 단일 모음 또는 평균값).
        f3_canonical: 목표 스케일 F3 (기본 F3_CANONICAL).

    Returns:
        α ∈ [ALPHA_MIN, ALPHA_MAX]. 1.0 = no warp.
        f3_observed 무효 시 1.0 반환.
    """
    if (f3_observed is None
            or not np.isfinite(f3_observed)
            or not (F3_VALID_MIN < f3_observed < F3_VALID_MAX)):
        return 1.0
    alpha = f3_canonical / f3_observed
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def warp_formants(f1: Optional[float],
                  f2: Optional[float],
                  f3: Optional[float],
                  alpha: float) -> tuple:
    """선형 주파수 워핑.

    원리: 성도 길이 1/α 배 변하면 모든 공명이 α 배 시프트.
    """
    def _w(f):
        return f * alpha if (f is not None and np.isfinite(f)) else None
    return _w(f1), _w(f2), _w(f3)


# ══════════════════════════════════════════
# Online 적응 (실시간 사용)
# ══════════════════════════════════════════

class VTLNEstimator:
    """실시간 사용자 F3 통계 누적 → 워핑 계수 자동 산출.

    Lifecycle:
      예열 (n < MIN_SAMPLES): α = 1.0 (no warp).
      적응 (n ≥ MIN_SAMPLES): α = F3_canonical / F3_EMA.
      EMA 로 화자 F3 변동 흡수.

    Cal-free 보장:
      사용자가 따로 발화 안 시켜도, 자연스러운 모음 발화 누적으로 α 추정.
    """

    def __init__(self,
                 ema_alpha: float = EMA_ALPHA,
                 min_samples: int = MIN_SAMPLES,
                 f3_canonical: float = F3_CANONICAL):
        self._f3_ema: Optional[float] = None
        self._n: int = 0
        self._ema_alpha = ema_alpha
        self._min_samples = min_samples
        self._f3_canonical = f3_canonical

    # ── 갱신 ──────────────────────────────────────────────

    def update(self, f3: Optional[float]) -> None:
        """voiced 청크 F3 1개 공급. 무효값 무시."""
        if (f3 is None
                or not np.isfinite(f3)
                or not (F3_VALID_MIN < f3 < F3_VALID_MAX)):
            return
        if self._f3_ema is None:
            self._f3_ema = float(f3)
        else:
            self._f3_ema = ((1.0 - self._ema_alpha) * self._f3_ema
                            + self._ema_alpha * float(f3))
        self._n += 1

    def reset(self) -> None:
        self._f3_ema = None
        self._n = 0

    # ── 조회 ──────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """워핑 적용 가능 여부 (예열 완료)."""
        return self._n >= self._min_samples

    @property
    def warping_factor(self) -> float:
        """현재 추정 α. 예열 전이면 1.0 (no warp)."""
        if not self.is_ready or self._f3_ema is None:
            return 1.0
        alpha = self._f3_canonical / self._f3_ema
        return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))

    @property
    def f3_estimate(self) -> Optional[float]:
        """현재 EMA F3 (디버깅용)."""
        return self._f3_ema

    # ── 워핑 ──────────────────────────────────────────────

    def warp(self,
             f1: Optional[float],
             f2: Optional[float],
             f3: Optional[float]) -> tuple:
        """현재 α 로 워핑. 예열 전이면 입력 그대로 반환."""
        return warp_formants(f1, f2, f3, self.warping_factor)

    # ── 진단 ──────────────────────────────────────────────

    def status(self) -> str:
        if not self.is_ready:
            return f"VTLN 예열 중 ({self._n}/{self._min_samples})"
        a = self.warping_factor
        f3 = self._f3_ema or 0.0
        return f"VTLN α={a:.3f}  F3≈{f3:.0f}Hz (n={self._n})"


# ══════════════════════════════════════════
# 단위 테스트 (모듈 직접 실행)
# ══════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("vtln.py 단위 테스트")
    print("=" * 60)

    # ── 1. compute_warping_factor 기본 ──
    print("\n[1] 즉시 추정")
    cases = [
        ("성인 여성 F3=3100", 3100, 0.935),     # 2900/3100
        ("성인 남성 F3=2500", 2500, 1.16),      # 2900/2500
        ("아동 F3=3500",      3500, 0.829),     # 2900/3500
        ("None",              None, 1.0),
        ("범위 밖 100",       100,  1.0),
        ("범위 밖 5000",      5000, 1.0),
    ]
    for label, f3, expected in cases:
        a = compute_warping_factor(f3)
        ok = abs(a - expected) < 0.005
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label:25s} → α={a:.3f}  (expected {expected:.3f})")
        assert ok, f"FAIL: {label}"

    # ── 2. warp_formants ──
    print("\n[2] 워핑 적용 (성인 남성 F1/F2/F3)")
    f1, f2, f3 = 600, 1100, 2500
    a = compute_warping_factor(f3)
    w1, w2, w3 = warp_formants(f1, f2, f3, a)
    print(f"  raw    F1={f1}  F2={f2}  F3={f3}")
    print(f"  α={a:.3f}")
    print(f"  warped F1={w1:.0f} F2={w2:.0f} F3={w3:.0f} (F3 → 2900?)")
    assert abs(w3 - 2900) < 1, "F3 가 캐노니컬에 안 맞음"

    # ── 3. Online estimator ──
    print("\n[3] Online estimator")
    est = VTLNEstimator()
    print(f"  예열 전: ready={est.is_ready}  α={est.warping_factor}")
    assert est.warping_factor == 1.0

    # 남성 화자 시뮬: F3 ~2500
    np.random.seed(0)
    for _ in range(20):
        est.update(2500 + np.random.uniform(-200, 200))
    print(f"  20 청크 후: {est.status()}")
    assert est.is_ready
    assert 1.10 < est.warping_factor < 1.20, "남성 α 범위 이상"

    # 여성 화자 reset 후 시뮬
    est.reset()
    for _ in range(20):
        est.update(3100 + np.random.uniform(-200, 200))
    print(f"  reset+여성 20 청크 후: {est.status()}")
    assert 0.90 < est.warping_factor < 1.00, "여성 α 범위 이상"

    print("\n✓ 모든 테스트 통과")
