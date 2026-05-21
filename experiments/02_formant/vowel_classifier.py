"""
vowel_classifier.py — 한국어 단모음 분류기

개선점:
  - Bark 스케일: 사람 청각 지각에 맞는 주파수 거리 계산
  - 성별 고유 참조값: 여성/남성 포먼트 분포 별도 적용
  - Mahalanobis 거리: 모음별 표준편차로 가중된 정규화 거리
  - F3 활용: 우/오 구분에 F3 추가 (입술 둥글기 효과로 F3 차이)
  - 화자 self-reference: set_user_refs() 로 calibrator 출력 등록 시
    학계 _REFS 대신 그 화자의 user_refs 사용 (경로 C, 다화자 환경 적합)
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
# 화자 self-reference (경로 C)
# calibrator.VowelCalibrator.user_refs 를 set_user_refs() 로 등록.
# 등록 시 학계 _REFS 대신 화자 자기 7모음 측정값으로 분류.
# 형식은 _REFS["female"] 와 동일: {vowel: (F1_μ, F1_σ, F2_μ, F2_σ, F3_μ, F3_σ)}
# ──────────────────────────────────────────────────────────────

_USER_REFS: dict | None = None


def set_user_refs(refs: dict | None) -> None:
    """화자 self-reference 등록 (또는 해제).

    Args:
        refs: calibrator.user_refs 형식의 dict, 또는 None.
              None / 빈 dict 이면 학계 _REFS 로 복귀.
    """
    global _USER_REFS
    _USER_REFS = refs if refs else None


def clear_user_refs() -> None:
    """학계 _REFS 로 복귀."""
    global _USER_REFS
    _USER_REFS = None


def has_user_refs() -> bool:
    return _USER_REFS is not None


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
                   f3: float = None,
                   scale: float = 1.0) -> tuple:
    """
    Bark 스케일 Mahalanobis 거리로 한국어 단모음 분류.

    Args:
        f1, f2: 포먼트 Hz
        gender: "female" | "male"
        f3:     F3 Hz (선택 — 우/오 구분 개선)
        scale:  SpeakerF0Tracker.scale (ref_f0/speaker_f0).
                측정 포먼트를 집단 기준 공간으로 정규화 (기본 1.0 = 보정 없음)

    Returns:
        (vowel: str, confidence: float 0–1)
    """
    if f1 is None or f2 is None or f1 < 100 or f2 < 200:
        return "?", 0.0

    # 화자 포먼트 → 집단 기준 공간으로 정규화
    if scale != 1.0:
        f1 = f1 * scale
        f2 = f2 * scale
        if f3 is not None:
            f3 = f3 * scale

    b1 = _bark(f1)
    b2 = _bark(f2)
    use_f3 = (f3 is not None and 1500 < f3 < 4500)
    b3 = _bark(f3) if use_f3 else None

    # ── 참조값 선택: user_refs 우선, 없으면 학계 _REFS ───────
    if _USER_REFS is not None:
        refs = _USER_REFS
    else:
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

    # 으/어 접전: F1이 주요 변별 피처 (F2 차이 192Hz는 노이즈에 묻힘)
    # user_refs 활성 시 → 그 화자의 으/어 F1 중간값 사용 (동적)
    # 학계 _REFS 사용 시 → female=553, male=470 고정
    if best_vowel in {"으", "어"} and second_dist - best_dist < 0.8:
        if _USER_REFS is not None and "으" in refs and "어" in refs:
            f1_mid = (refs["으"][0] + refs["어"][0]) / 2.0
        else:
            f1_mid = 553.0 if gender == "female" else 470.0
        if f1 > f1_mid + 40:
            best_vowel = "어"
        elif f1 < f1_mid - 40:
            best_vowel = "으"

    return best_vowel, _confidence(best_dist, second_dist)


