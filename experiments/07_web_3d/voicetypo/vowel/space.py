"""모음 공간 좌표계 변환.

포먼트 (F1, F2, F3) ↔ 정규화된 3D 좌표 (x, y, z).

좌표계:
    X =  norm(F2, 400~3200Hz)    # 전설(+1) / 후설(-1)
    Y = -norm(F1, 180~1050Hz)    # 고모음(+1) / 저모음(-1)
    Z =  norm(F3, 2200~3700Hz)
"""

import numpy as np

# 포먼트 정규화 범위 (Hz)
F1_MIN, F1_MAX = 180.0, 1050.0
F2_MIN, F2_MAX = 400.0, 3200.0
F3_MIN, F3_MAX = 2200.0, 3700.0


def _norm(value, lo, hi):
    """[lo, hi] → [-1, +1] 선형 매핑."""
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def _denorm(n, lo, hi):
    """[-1, +1] → [lo, hi] 역매핑."""
    return (n + 1.0) / 2.0 * (hi - lo) + lo


def formant_to_xyz(f1, f2, f3):
    """포먼트 → 모음 공간 xyz.

    Returns:
        np.ndarray shape (3,) — [x, y, z]
        NaN 입력이면 NaN 출력.
    """
    x = _norm(f2, F2_MIN, F2_MAX)     # 전설(+) / 후설(-)
    y = -_norm(f1, F1_MIN, F1_MAX)    # 고모음(+) / 저모음(-)
    z = _norm(f3, F3_MIN, F3_MAX)
    return np.array([x, y, z], dtype=np.float64)


def xyz_to_formant(xyz):
    """모음 공간 xyz → 포먼트 (f1, f2, f3).

    Args:
        xyz: array-like shape (3,)

    Returns:
        (f1, f2, f3) Hz 튜플.
    """
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    f2 = _denorm(x, F2_MIN, F2_MAX)
    f1 = _denorm(-y, F1_MIN, F1_MAX)
    f3 = _denorm(z, F3_MIN, F3_MAX)
    return (f1, f2, f3)


# ─── 참조점 유틸리티 ───

# 모음별 참조 포먼트 (남녀 평균)
VOWEL_FORMANTS = {
    '아': (798, 1369, 2748),
    '이': (255, 2524, 3366),
    '우': (335,  703, 2579),
    '오': (346,  726, 2593),
    '으': (354, 1485, 2606),
    '어': (511,  903, 2817),
    '에': (480, 2142, 2836),
}

# 참조 xyz (기본값, 캘리브레이션 전)
DEFAULT_CENTROIDS = {
    name: formant_to_xyz(*fmt) for name, fmt in VOWEL_FORMANTS.items()
}


def distance_to_centroids(xyz, centroids=None):
    """xyz에서 각 모음 중심점까지의 유클리드 거리.

    Args:
        xyz: np.ndarray shape (3,)
        centroids: dict[str, np.ndarray] — 없으면 DEFAULT_CENTROIDS 사용

    Returns:
        dict[str, float] — 모음별 거리
    """
    if centroids is None:
        centroids = DEFAULT_CENTROIDS
    return {name: float(np.linalg.norm(xyz - c))
            for name, c in centroids.items()}
