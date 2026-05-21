"""
vowel_classifier.py — 한국어 단모음 분류기

개선점:
  - Bark 스케일: 사람 청각 지각에 맞는 주파수 거리 계산
  - 성별 고유 참조값: 여성/남성 포먼트 분포 별도 적용
  - Mahalanobis 거리: 모음별 표준편차로 가중된 정규화 거리
  - F3 활용: 우/오 구분에 F3 추가 (입술 둥글기 효과로 F3 차이)
  - 개인 보정: set_calibration()으로 사용자 본인 기준값 설정 시 우선 사용
"""

import numpy as np

# ──────────────────────────────────────────────────────────────
# 참조값: (F1_μ, F1_σ, F2_μ, F2_σ, F3_μ, F3_σ)  단위: Hz
#
# 여성: Yoon(2015) + Yang(1996) + Oh(1995) 복수 연구 보정
# 남성: 여성 대비 F1 ×0.85, F2 ×0.82, F3 ×0.83
#
# 주요 조정:
#   우/오: σ2 축소(120→) + F2 중심 간격 확보 (F3로 보완)
#   으/어: σ1 확대로 포용력 향상
#   에: σ2 유지(화자 편차 큼)
# ──────────────────────────────────────────────────────────────
_REFS = {
    "female": {
        # config.py VOWEL_REFS 중심값 (시각화 타원과 일치)
        # σ = VOWEL_REFS 범위/3  (mean ± 1.5SD → 3SD = range)
        #        F1    σ1    F2    σ2    F3    σ3
        "아": (978,  100, 1397,  175, 2600,  260),
        "에": (548,  100, 2125,  185, 2980,  270),
        "이": (352,   78, 2787,  250, 3180,  265),
        "오": (487,   88,  840,  148, 2680,  230),
        "우": (367,   78,  660,  121, 2270,  225),
        "으": (435,   90, 1404,  217, 2720,  245),
        "어": (671,  109, 1212,  178, 2640,  235),
    },
    "male": {
        # VOWEL_REFS_MALE 중심값 (시각화 타원과 일치)
        #        F1    σ1    F2    σ2    F3    σ3
        "아": (831,   88, 1145,  143, 2158,  215),
        "에": (466,   88, 1743,  152, 2474,  224),
        "이": (299,   68, 2285,  205, 2639,  220),
        "오": (414,   78,  689,  121, 2224,  191),
        "우": (312,   68,  541,  100, 1884,  186),
        "으": (370,   79, 1151,  178, 2258,  203),
        "어": (570,   95,  994,  146, 2191,  195),
    },
}

# F3 가중치 (우/오 구분 보조)
_W_F3 = 0.30   # 0.25 → 0.30: 우/오 구분을 위해 F3 비중 증가


# ──────────────────────────────────────────────────────────────
# 개인 보정 데이터
# set_calibration()으로 설정하면 _REFS 대신 우선 사용
# ──────────────────────────────────────────────────────────────
_calib_data: dict = {}   # vowel → {f1, f2, f1_sd, f2_sd}


def set_calibration(data: dict) -> None:
    """
    개인 보정 데이터 설정.
    data 형식: {"아": {"f1": 850, "f2": 1320, "f1_sd": 45, "f2_sd": 80}, ...}
    """
    global _calib_data
    _calib_data = {k: v for k, v in data.items()}


def clear_calibration() -> None:
    global _calib_data
    _calib_data = {}


def _bark(f: float) -> float:
    """Zwicker & Terhardt (1980) Bark 스케일 변환"""
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _mahal_dist(b1, b2, b3,
                bm1, bsd1, bm2, bsd2,
                bm3=None, bsd3=None) -> float:
    """Bark 공간 Mahalanobis 거리 계산 (F3 선택)"""
    d_sq = ((b1 - bm1) / bsd1) ** 2 + ((b2 - bm2) / bsd2) ** 2
    if b3 is not None and bm3 is not None:
        d_sq = (1.0 - _W_F3) * d_sq + _W_F3 * ((b3 - bm3) / bsd3) ** 2
    return float(np.sqrt(d_sq))


def _confidence(best_dist: float, second_dist: float) -> float:
    base_conf  = max(0.0, 1.0 - best_dist / 4.0)
    separation = min(1.0, (second_dist - best_dist) / 1.5)
    return float(0.7 * base_conf + 0.3 * separation)


def classify_vowel(f1: float, f2: float,
                   gender: str = "female",
                   f3: float = None) -> tuple:
    """
    Bark 스케일 Mahalanobis 거리로 한국어 단모음 분류.
    개인 보정 데이터가 있으면 그것을 우선 사용.

    Args:
        f1, f2: 포먼트 Hz
        gender:  "female" | "male"
        f3:      F3 Hz (선택 — 우/오 구분 개선)

    Returns:
        (vowel: str, confidence: float 0–1)
    """
    if f1 is None or f2 is None or f1 < 100 or f2 < 200:
        return "?", 0.0

    b1 = _bark(f1)
    b2 = _bark(f2)
    use_f3 = (f3 is not None and 1500 < f3 < 4500)
    b3 = _bark(f3) if use_f3 else None

    # ── 개인 보정 데이터 우선 ──────────────────────────────────
    if _calib_data:
        return _classify_with_calib(b1, b2, b3)

    # ── 기본 모집단 참조값 ────────────────────────────────────
    refs = _REFS.get(gender, _REFS["female"])
    best_vowel  = "?"
    best_dist   = float("inf")
    second_dist = float("inf")

    for name, (m1, sd1, m2, sd2, m3, sd3) in refs.items():
        bm1  = _bark(m1)
        bm2  = _bark(m2)
        bsd1 = max(_bark(m1 + sd1) - bm1, 0.05)
        bsd2 = max(_bark(m2 + sd2) - bm2, 0.05)
        bm3_ = bsd3_ = None
        if use_f3:
            bm3_  = _bark(m3)
            bsd3_ = max(_bark(m3 + sd3) - bm3_, 0.05)

        d = _mahal_dist(b1, b2, b3, bm1, bsd1, bm2, bsd2, bm3_, bsd3_)

        if d < best_dist:
            second_dist = best_dist
            best_dist   = d
            best_vowel  = name
        elif d < second_dist:
            second_dist = d

    if best_dist > 4.0:
        return "?", 0.0

    return best_vowel, _confidence(best_dist, second_dist)


def _classify_with_calib(b1: float, b2: float, b3) -> tuple:
    """개인 보정 데이터로 분류 (Bark 공간 Mahalanobis)"""
    best_vowel  = "?"
    best_dist   = float("inf")
    second_dist = float("inf")

    for name, vals in _calib_data.items():
        m1  = float(vals["f1"])
        m2  = float(vals["f2"])
        # 보정 시 측정한 SD 사용; 최솟값 보장으로 과적합 방지
        sd1 = max(float(vals.get("f1_sd", 80.0)), 55.0)
        sd2 = max(float(vals.get("f2_sd", 150.0)), 110.0)

        bm1  = _bark(m1)
        bm2  = _bark(m2)
        bsd1 = max(_bark(m1 + sd1) - bm1, 0.05)
        bsd2 = max(_bark(m2 + sd2) - bm2, 0.05)

        # F3 보정값 없으므로 F3 항은 건너뜀
        d = _mahal_dist(b1, b2, None, bm1, bsd1, bm2, bsd2)

        if d < best_dist:
            second_dist = best_dist
            best_dist   = d
            best_vowel  = name
        elif d < second_dist:
            second_dist = d

    # 보정 기반 분류는 임계값을 약간 완화 (개인 편차 반영)
    if best_dist > 3.5:
        return "?", 0.0

    return best_vowel, _confidence(best_dist, second_dist)
