"""LPC 기반 포먼트(F1/F2) 추출.

오디오 프레임에서 LPC 계수를 구하고,
다항식 근(root)에서 포먼트 주파수를 추출.
"""

import numpy as np

LPC_ORDER = 12  # LPC 차수 (일반적으로 sr/1000 + 2~4)


def _autocorrelate(x, order):
    """자기상관 함수."""
    n = len(x)
    r = np.zeros(order + 1)
    for i in range(order + 1):
        r[i] = np.sum(x[:n - i] * x[i:])
    return r


def _levinson_durbin(r, order):
    """Levinson-Durbin 알고리즘으로 LPC 계수 계산."""
    a = np.zeros(order + 1)
    e = r[0]
    a[0] = 1.0

    for i in range(1, order + 1):
        acc = sum(a[j] * r[i - j] for j in range(i))
        k = -acc / max(e, 1e-12)
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e *= (1 - k * k)
        if e <= 0:
            break

    return a[1:order + 1], e


def extract_formants(audio: np.ndarray, sr: int = 44100,
                     n_formants: int = 2) -> list:
    """오디오 프레임에서 포먼트 주파수 추출.

    Args:
        audio: float32 배열
        sr: 샘플레이트
        n_formants: 추출할 포먼트 수 (기본 2: F1, F2)

    Returns:
        [F1, F2, ...] Hz 리스트. 추출 실패 시 빈 리스트.
    """
    # 프리엠퍼시스
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    # 해밍 윈도우
    windowed = emphasized * np.hamming(len(emphasized))

    # LPC
    r = _autocorrelate(windowed, LPC_ORDER)
    if r[0] < 1e-10:
        return []

    lpc_coeffs, _ = _levinson_durbin(r, LPC_ORDER)

    # LPC 다항식의 근 구하기
    poly = np.concatenate(([1.0], lpc_coeffs))
    roots = np.roots(poly)

    # 허수부가 양수인 근만 (대칭이므로 절반만)
    roots = roots[np.imag(roots) > 0.01]

    if len(roots) == 0:
        return []

    # 근의 각도 → 주파수
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2.0 * np.pi))

    # 대역폭 필터 (너무 넓은 포먼트 제거)
    bandwidths = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.abs(roots))
    valid = bandwidths < 400  # 대역폭 400Hz 이하만
    freqs = freqs[valid]

    # 유효 범위 필터 (50~5500Hz)
    freqs = freqs[(freqs > 50) & (freqs < 5500)]

    if len(freqs) == 0:
        return []

    # 오름차순 정렬, 상위 n개
    freqs = np.sort(freqs)
    return freqs[:n_formants].tolist()
