"""
multi_prototype.py — Multi-Prototype 모음 분류 (Phase A4)

개념:
  학계 _REFS["female"] 단일 prototype 대신, VTL scale 변형
  α = 0.85, 0.92, 1.00, 1.08, 1.15 → 5 prototype 동시 운영.
  각 입력 (f1, f2, f3) 에 대해 모든 prototype 시도 → 최고 confidence 선택.

VTLN vs Multi-Prototype:
  VTLN     : 화자 단일 α 추정 → 모든 발화에 동일 적용 (hard)
  Multi-P  : 발화마다 최적 α 동적 선택 (soft)
  결합 가능: VTLN 으로 1차 정규화 → 그 위에 multi-P 미세 조정

알고리즘 (per sample):
  for α in [0.85, 0.92, 1.00, 1.08, 1.15]:
      vowel, conf = classify_vowel(f1, f2, gender, f3, scale=α)
      record (vowel, α, conf)
  best = argmax(conf)
  return best.vowel, best.α, best.conf

α 선택 근거:
  성인 남성   F3 ≈ 2500 → α=1.16 → 1.15 prototype 매칭
  성인 여성   F3 ≈ 3100 → α=0.94 → 0.92 prototype 매칭
  아동       F3 ≈ 3500 → α=0.83 → 0.85 prototype 매칭
  중간/canonical → 1.00 prototype
"""

from typing import Optional
import numpy as np

from vowel_classifier import classify_vowel


# ══════════════════════════════════════════
# Prototype α 후보
# ══════════════════════════════════════════

# 5 단계 — 사람 성도 변동 범위 (0.83~1.20) 커버.
DEFAULT_ALPHAS = (0.85, 0.92, 1.00, 1.08, 1.15)


# ══════════════════════════════════════════
# 단일 샘플 분류
# ══════════════════════════════════════════

def classify_multi_proto(f1: float, f2: float,
                         f3: Optional[float] = None,
                         alphas: tuple = DEFAULT_ALPHAS,
                         gender: str = "female") -> tuple:
    """다중 α prototype 시도 → 최고 confidence 선택.

    Args:
        f1, f2, f3: raw formants (Hz).
        alphas: 시도할 scale 후보들.
        gender: classify_vowel 기준 _REFS 선택.

    Returns:
        (vowel, best_alpha, confidence, all_results)
        all_results: [(vowel, alpha, conf), ...] 디버깅용.
    """
    if f1 is None or f2 is None or f1 < 100 or f2 < 200:
        return "?", 1.0, 0.0, []

    results = []
    for a in alphas:
        v, c = classify_vowel(f1, f2, gender, f3=f3, scale=a)
        results.append((v, a, c))

    # 최고 confidence (단, "?" 제외)
    valid = [r for r in results if r[0] != "?"]
    if not valid:
        return "?", 1.0, 0.0, results

    best = max(valid, key=lambda r: r[2])
    return best[0], best[1], best[2], results


# ══════════════════════════════════════════
# 다중 청크 vote (Layer 5 결합)
# ══════════════════════════════════════════

def vote_multi_proto(samples: list,
                     alphas: tuple = DEFAULT_ALPHAS,
                     gender: str = "female") -> tuple:
    """5 시간점 등 다중 샘플에 대해 multi-proto + confidence-weighted vote.

    Args:
        samples: [(f1, f2, f3), ...] 청크별 측정.
        alphas: prototype scale 후보.

    Returns:
        (vowel, total_conf, n_voters, alpha_dist)
        alpha_dist: 각 청크에서 선택된 α 분포 (다양성 진단용).
    """
    from collections import defaultdict, Counter
    votes = defaultdict(float)
    nv = 0
    alpha_chosen = []

    for f1, f2, f3 in samples:
        if f1 is None or f2 is None:
            continue
        v, a, c, _ = classify_multi_proto(f1, f2, f3, alphas, gender)
        if v == "?" or c <= 0:
            continue
        votes[v] += c
        alpha_chosen.append(a)
        nv += 1

    if not votes:
        return "?", 0.0, 0, Counter()

    best = max(votes, key=votes.get)
    return best, votes[best], nv, Counter(alpha_chosen)


# ══════════════════════════════════════════
# 단위 테스트
# ══════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("multi_prototype.py 단위 테스트")
    print("=" * 60)

    # 본인 (canonical-near 여성) 표준 모음 측정값 시뮬
    test_cases = [
        # (vowel, f1, f2, f3, expected_alpha_near)
        ("아", 978, 1397, 2600, 1.00),   # canonical female 아
        ("이", 352, 2787, 3180, 1.00),
        ("어", 671, 1212, 2640, 1.00),
        # 가상 남성 (formants × 0.83)
        ("아_male", 812, 1159, 2158, 1.15),  # → α=1.15 prototype 매칭 기대
        ("이_male", 292, 2313, 2639, 1.15),
        # 가상 아동 (formants × 1.20)
        ("아_child", 1174, 1676, 3120, 0.85),  # → α=0.85 매칭
    ]

    for label, f1, f2, f3, expected_a in test_cases:
        v, a, c, all_r = classify_multi_proto(f1, f2, f3)
        mark = "✓" if abs(a - expected_a) < 0.05 else "△"
        print(f"  {mark} {label:12s} → vowel={v} α={a:.2f} conf={c:.2f} "
              f"(expected α≈{expected_a})")

    # vote test
    print("\n[vote test]")
    samples = [(978, 1397, 2600), (980, 1400, 2610), (970, 1395, 2595),
               (985, 1402, 2608), (975, 1390, 2598)]
    v, c, nv, ad = vote_multi_proto(samples)
    print(f"  5 청크 (canonical 아) → {v} (conf={c:.2f}, voters={nv}, α 분포={dict(ad)})")
    assert v == "아", f"기대 '아', got {v}"

    print("\n✓ 모든 테스트 통과")
