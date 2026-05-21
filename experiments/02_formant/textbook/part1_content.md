# VoiceTypo 교재 — Part 1: 포먼트 추출 원리
저자: 허재원 | 참조 코드: realtime_formant 프로젝트

---

## 슬라이드 1: Part 1 표지 — 포먼트란 무엇인가?

**학습 목표**:
- 음성이 물리적으로 어떻게 만들어지는지 source-filter 모델로 이해한다
- 포먼트(formant)가 무엇이며 왜 모음 인식에 핵심인지 설명할 수 있다
- VoiceTypo가 포먼트를 추출하기 위해 어떤 파이프라인을 쓰는지 조감도를 그릴 수 있다

**핵심 개념**:
- 포먼트(formant): 성도(vocal tract)의 공명 주파수 — 목구멍+입 공간이 특정 주파수를 증폭
- F1, F2, F3: 낮은 순서로 번호를 매긴 첫 세 공명 주파수
- 모음마다 F1-F2 위치가 다르기 때문에 포먼트 = 모음의 지문
- 이 교재에서 다루는 세 방법: Praat Burg LPC, pyworld CheapTrick, scipy LPC 근

**다이어그램**:
[설명: 가로로 긴 블록 다이어그램. 왼쪽부터 순서대로: "성대(source)" → "성도 필터(filter)" → "출력 음파(speech)". 성대 블록 아래에는 "글라이드 + 기식음 + 무성음" 설명 텍스트. 성도 필터 블록 아래에는 포먼트 스펙트럼 미니 그래프를 그려 F1·F2·F3 세 개의 언덕 모양 피크를 표시. 전체적으로 깔끔한 흑백 박스-화살표 스타일.]

**발표자 노트**:
- "포먼트를 모르면 모음 인식을 이해할 수 없다"는 문장으로 시작
- 이 파트 끝에는 세 방법을 직접 코드로 볼 수 있음을 예고

---

## 슬라이드 2: Source-Filter 모델 — 목소리의 두 층

**핵심 개념**:
- Source(성대): 성대가 진동하며 만드는 글로탈 펄스(glottal pulse) — 거친 톱니파에 가까움
- Filter(성도): 목·구개·입술이 이루는 공동(cavity) — 특정 주파수를 공명·증폭
- 최종 음성 = Source 스펙트럼 × Filter(성도) 주파수 응답
- 포먼트 = Filter의 공명 주파수 = 스펙트럼 포락선(spectral envelope)의 봉우리

**수식**:
$$
S(f) = G(f) \cdot V(f) \cdot R(f)
$$
기호 설명:
- $S(f)$: 최종 음성 스펙트럼
- $G(f)$: 글로탈 소스(성대 진동)
- $V(f)$: 성도 필터 (포먼트를 만드는 주인공)
- $R(f)$: 입술 방사(radiation) 특성 (+6 dB/octave 보정)

**다이어그램**:
[설명: 세 행으로 구성된 스펙트럼 그림. 첫 번째 행 "G(f) — 성대 소스": 조밀하게 늘어선 고조파(harmonics) 스파이크들이 높은 주파수로 갈수록 점점 작아지는 모양. 두 번째 행 "V(f) — 성도 필터": 부드러운 포락선에 F1=500Hz, F2=1500Hz, F3=2500Hz 위치에 봉우리 세 개. 세 번째 행 "S(f) — 최종 음성": 고조파 스파이크가 V(f)의 봉우리 형태를 따라 높낮이가 조절된 모양. 세 그래프 사이에 "×" 기호로 곱셈 관계를 나타냄.]

**발표자 노트**:
- 실제 음성은 세 성분의 곱이지만, 포먼트 추출은 V(f)만 복원하는 작업
- LPC가 바로 G(f)를 제거하고 V(f)를 추정하는 방법임을 다음 슬라이드에서 설명

---

## 슬라이드 3: 한국어 모음 F1-F2 지도

**핵심 개념**:
- F1 ↑ = 입이 더 열림(open) / F1 ↓ = 입이 덜 열림(close)
- F2 ↑ = 혀가 앞쪽(front) / F2 ↓ = 혀가 뒤쪽(back)
- 한국어 단모음 7개: /아, 에, 이, 오, 우, 으, 어/
- 같은 모음이라도 화자(성별·나이)마다 절대값 다름 → 상대 위치가 더 중요

**수식**:
$$
\text{모음 공간} = (F1, F2) \text{ 좌표}
$$
기호 설명:
- F1 축: 세로축, 낮은 값이 위(고모음)
- F2 축: 가로축, 높은 값이 왼쪽(전설모음)
(전통적 음성학 도표 방향)

**다이어그램**:
[설명: 전통적인 F1-F2 모음 사각도. 가로축 = F2(Hz), 오른쪽→왼쪽으로 500~3200Hz. 세로축 = F1(Hz), 위→아래로 200~1100Hz. 7개 모음을 타원 영역으로 표시: "아"(F1≈978, F2≈1397) 오른쪽 아래, "이"(F1≈352, F2≈2787) 왼쪽 위, "우"(F1≈367, F2≈660) 오른쪽 위, "에"(F1≈548, F2≈2125) 왼쪽 중간, "오"(F1≈487, F2≈840) 오른쪽 중간, "어"(F1≈671, F2≈1212) 가운데 아래, "으"(F1≈435, F2≈1404) 가운데 중간. 각 모음 색상은 코드의 VOWEL_REFS 색상과 동일하게: 아=빨강, 에=주황, 이=노랑, 오=초록, 우=하늘, 으=파랑, 어=보라.]

**내 코드에서는**:
파일: `config.py`
줄번호: `110-129`
```python
VOWEL_REFS = {
    "아": dict(F1=(828, 1128), F2=(1135, 1660), color="#FF4444"),
    "에": dict(F1=(398,  698), F2=(1848, 2402), color="#FFAA22"),
    "이": dict(F1=(235,  469), F2=(2412, 3162), color="#FFFF44"),
    "오": dict(F1=(355,  619), F2=(618,  1062), color="#44FF88"),
    "우": dict(F1=(250,  484), F2=(479,   841), color="#44DDFF"),
    "으": dict(F1=(300,  570), F2=(1078, 1730), color="#4488FF"),
    "어": dict(F1=(508,  834), F2=(945,  1479), color="#CC55FF"),
}

VOWEL_REFS_MALE = {
    "아": dict(F1=(699,  963), F2=(931,  1359), color="#FF4444"),
    ...
}
```
설명: 여성 기준(Yoon 2015)과 남성 기준(여성의 약 82-85%)을 따로 관리합니다. 포먼트 절대값은 성별에 따라 달라지므로 두 기준표가 필요합니다. 튜플 `(min, max)`는 mean ± 1.5 SD 범위입니다.

**발표자 노트**:
- 모음 사각도는 음성학 교과서에서 자음 표와 함께 가장 중요한 그림
- "이" F2가 2787Hz인 이유: 혀가 앞으로 올라가서 앞쪽 공동이 짧아짐 → 공명 주파수 상승

---

## 슬라이드 4: 음성 전처리 파이프라인 개요

**핵심 개념**:
- 포먼트 추출 전 반드시 거쳐야 할 전처리 단계 4가지
- DC 제거 → 프리엠퍼시스(pre-emphasis) → 프레이밍(framing) → 윈도잉(windowing)
- 각 단계는 신호의 특정 문제를 해결하기 위해 존재
- 순서를 바꾸면 결과가 달라짐 — 순서가 중요

**다이어그램**:
[설명: 세로로 긴 흐름도. 입력 "마이크 원본 신호 (float32, 44100Hz)" → ① "DC 제거: x -= mean(x)" → ② "프리엠퍼시스: y[n] = x[n] - 0.97·x[n-1]" → ③ "프레이밍: 300ms 청크 (13230샘플)" → ④ "윈도잉: 함밍 창(Hamming window)" → ⑤ "FFT / LPC 분석". 각 단계 옆에 신호 파형 미니 그림: DC 제거 전에는 중심이 위로 치우친 파형, 후에는 0 중심. 프리엠퍼시스 후에는 고주파 성분이 부각된 모습.]

**발표자 노트**:
- VoiceTypo는 300ms 청크를 사용 (현재 기본값, 튜닝 포인트)
- 짧으면 빠르지만 정확도 하락, 길면 정확하지만 지연 증가

---

## 슬라이드 5: DC 제거와 프리엠퍼시스

**핵심 개념**:
- DC 성분: 마이크 직류 오프셋(0Hz 성분) → LPC/FFT 계산을 오염시킴 → 평균값 빼기로 제거
- 프리엠퍼시스(pre-emphasis): 고주파를 의도적으로 증폭하는 1차 고역통과 필터
- 왜 필요? 음성 스펙트럼은 고주파로 갈수록 약해짐(−6 dB/octave) → 증폭해야 포먼트가 잘 보임
- alpha = 0.97: 거의 모든 고주파를 증폭 (0에 가까울수록 약한 효과)

**수식**:
$$
y[n] = x[n] - \alpha \cdot x[n-1]
$$
기호 설명:
- $x[n]$: 입력 신호의 n번째 샘플
- $y[n]$: 프리엠퍼시스 적용 후 신호
- $\alpha$: 프리엠퍼시스 계수 (현재 기본값: 0.97, 튜닝 포인트)
- 주파수 응답: $H(z) = 1 - \alpha z^{-1}$ — 고역통과 특성

**Python 구현**:
```python
def preemphasis(signal, alpha=0.97):
    # 첫 번째 샘플은 그대로, 나머지는 차분
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])
```

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `35-36`
```python
def preemphasis(signal: np.ndarray, alpha: float = PREEMPH_ALPHA) -> np.ndarray:
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])
```
설명: alpha = PREEMPH_ALPHA = 0.97 (config.py 55번 줄). DC 제거는 `formant_engine.py` 253번 줄 `chunk - np.mean(chunk)`에서 수행하고, 이후 이 함수를 호출합니다.

**발표자 노트**:
- `np.append(signal[0], ...)`는 signal[0]을 그대로 쓰는 일반적인 관례
- Praat도 내부적으로 pre-emphasis를 수행하지만, VoiceTypo는 수동으로 먼저 적용 후 Praat pre_emphasis=50(≈0Hz)으로 설정해 이중 적용을 피함

---

## 슬라이드 6: 프레이밍과 윈도잉

**핵심 개념**:
- 프레이밍(framing): 연속 신호를 짧은 구간(프레임)으로 분할 — 음성은 단기적으로 정상(stationary)
- 윈도잉(windowing): 프레임 끝에서 발생하는 불연속점을 부드럽게 처리
- Hamming 창: 가장 흔히 쓰이는 윈도 — 주파수 누설(spectral leakage) 최소화
- VoiceTypo: 300ms 청크를 1개 프레임으로 처리 (프레임 분할 없음 — 실시간 단순화)

**수식**:
$$
w[n] = 0.54 - 0.46 \cos\!\left(\frac{2\pi n}{N-1}\right), \quad 0 \le n \le N-1
$$
기호 설명:
- $w[n]$: n번째 샘플에 곱하는 Hamming 창 값 (0~1)
- $N$: 프레임 길이 (샘플 수)
- 양 끝이 0에 가까워지므로 불연속 제거

**Python 구현**:
```python
import numpy as np

frame = signal[start:start + frame_len]
window = np.hamming(frame_len)
windowed = frame * window  # 요소별 곱
```

**발표자 노트**:
- Praat은 "To Formant (burg)" 내부에서 윈도잉을 자동 처리
- scipy LPC 방법(scipy_lpc_formants)은 원본 청크 전체를 사용 — Praat 내부 윈도잉과 다름
- 300ms는 모음 steady-state를 충분히 담기에 적합 (현재 기본값, 튜닝 포인트)

---

## 슬라이드 7: FFT와 스펙트럼 포락선

**핵심 개념**:
- FFT(Fast Fourier Transform): 시간 도메인 신호 → 주파수 도메인 스펙트럼으로 변환
- 스펙트럼: 각 주파수 성분의 크기를 보여줌
- 스펙트럼 포락선(spectral envelope): 고조파 스파이크들의 "윤곽선" = 성도 필터 V(f) 근사
- 포먼트 = 포락선의 봉우리 → FFT만으로도 대략 위치 파악 가능하지만 정확도 낮음

**수식**:
$$
X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j2\pi kn/N}
$$
기호 설명:
- $x[n]$: 시간 도메인의 $n$번째 샘플
- $X[k]$: 주파수 $k$에 해당하는 복소수 성분 (진폭 + 위상)
- $e^{-j2\pi kn/N}$: 주파수 $k$의 복소 회전 (Euler 공식)
- $N$: 프레임 길이 (샘플 수)
- $|X[k]|$: 크기 스펙트럼 (파워 = $|X[k]|^2$)
- 주파수 해상도 = $f_s / N$ Hz/bin

> Euler 공식: $e^{j\theta} = \cos\theta + j\sin\theta$ — FFT는 신호를 수많은 sin/cos의 합으로 분해하는 과정입니다.

**다이어그램**:
[설명: 두 행 구성. 위 행 "시간 도메인": x축=시간(ms 0~300), y축=진폭. 모음 /아/ 발성처럼 주기적인 파형 그림, 주기가 약 8ms(F0≈125Hz)임을 보여줌. 아래 행 "주파수 도메인(FFT)": x축=주파수(Hz 0~5000), y축=크기. 125Hz 간격으로 고조파 스파이크들이 늘어서 있고, 그 위에 부드러운 포락선(점선)을 그려 F1≈800Hz, F2≈1400Hz 위치에 봉우리가 있음을 표시. "이것이 포먼트" 라벨 화살표.]

**발표자 노트**:
- FFT 포락선만으로 포먼트를 찾으면 고조파와 포먼트를 혼동할 수 있음
- LPC는 이 문제를 해결 — 다음 슬라이드에서 설명

---

## 슬라이드 8: LPC 원리 — 성도를 수식으로 모델링

**핵심 개념**:
- LPC(Linear Predictive Coding, 선형 예측 부호화): 현재 샘플을 과거 p개의 샘플의 선형 조합으로 예측
- 핵심 아이디어: 성도(성도 필터)가 AR(AutoRegressive) 과정처럼 동작한다고 가정
- LPC 계수 a_1...a_p가 바로 성도 필터 계수 → 이 계수로 포먼트를 계산
- 차수 p = 성도가 얼마나 복잡한지를 결정 (더 높을수록 세밀하지만 과적합 위험)

**수식**:
$$
\hat{x}[n] = -\sum_{k=1}^{p} a_k \cdot x[n-k]
$$
$$
e[n] = x[n] - \hat{x}[n]
$$
기호 설명:
- $\hat{x}[n]$: n번째 샘플 예측값
- $a_k$: k번째 LPC 계수 (성도 필터 계수)
- $p$: LPC 차수 (현재 기본값: 16, 튜닝 포인트)
- $e[n]$: 예측 오차(잔차) — 이상적으로는 글로탈 펄스(성대 소스)
- 목표: $\sum e[n]^2$ 최소화 → $a_k$를 구하는 것이 LPC 문제

**다이어그램**:
[설명: 블록 다이어그램. 왼쪽 "x[n] 입력 신호" → "LPC 분석 (최소화: 예측 오차²)" → "계수 [a₁, a₂, ..., a_p]" → "AR 다항식 A(z) = 1 + a₁z⁻¹ + ... + aₚz⁻ᵖ" → "역수: 1/A(z) = 성도 필터". 성도 필터에서 오른쪽으로 화살표 → "스펙트럼 포락선 (포먼트 포함)".]

**발표자 노트**:
- LPC는 소스(성대)를 제거하고 필터(성도)만 남기는 방법
- 예측 오차 e[n]이 줄어들면 줄어들수록 a_k가 성도를 잘 모델링한 것

---

## 슬라이드 9: LPC 차수 p 결정법

**핵심 개념**:
- LPC 차수 p가 너무 작으면: 포먼트 놓침 (under-fitting)
- LPC 차수 p가 너무 크면: 가짜 공명 추가 (over-fitting)
- 실용적 경험칙: $p = 2 \times (\text{포먼트 수}) + 2$, 또는 샘플레이트(kHz) + 4~6
- 예: 44.1 kHz → p ≈ 44 + 5 = 49 이지만, 실제로는 다운샘플 후 16 정도 사용

**수식**:
$$
p \approx \frac{f_s}{1000} + 4 \sim 6
$$
기호 설명:
- $f_s$: 샘플레이트(Hz) — 단위를 kHz로 환산해서 사용
- 포먼트 1개당 복소근 쌍(pair)이 1개 → F1~F4를 잡으려면 최소 p=8
- 경험칙 선행값: 논문마다 다름, Praat 기본값 = 16kHz 리샘플링 후 p=10

> **VoiceTypo에서의 적용**:
> - 기본 공식: 16kHz 기준 + 4~6 → p = 10~12 권장
> - VoiceTypo 입력: **44100Hz** — 별도 다운샘플링 로직 없음 (44.1kHz 원본 그대로 사용)
> - 실제 코드에서 사용 중인 p 값: **order = 16** (고정값)
> - 파일: `formant_ensemble.py:222`
> - Praat Burg는 max_formants=5 → 내부적으로 p ≈ 10 (파일: `config.py:92,99`)
> - 44.1kHz에 p=16은 경험칙(≈49)보다 낮지만, 실시간 속도 우선 + 포먼트 F1~F3 추출에는 충분합니다.

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `219-222`
```python
def scipy_lpc_formants(chunk_pe: np.ndarray, gender: str):
    """
    최적화:
      - LPC 차수: 16 (포먼트 분석 충분, 속도 빠름)
    """
    p   = PARAMS[gender]
    sr  = float(SAMPLE_RATE)
    order = 16   # 포먼트 분석에 충분한 차수 (고정)
```
설명: 44100Hz 신호에 order=16을 사용합니다. 경험칙(44+5≈49)보다 낮지만, 실시간 처리 속도와 안정성을 우선시한 선택입니다. (현재 기본값, 튜닝 포인트)

**발표자 노트**:
- Praat Burg는 내부적으로 p=max_formants×2를 사용 (PARAMS의 max_formants=5 → p≈10)
- scipy 방법은 order=16 고정 — 더 세밀하게 조정 원하면 이 값 수정

---

## 슬라이드 10: Levinson-Durbin 알고리즘

**핵심 개념**:
- LPC 계수 계산 = 자기상관 행렬 방정식 풀기 → $O(p^3)$ 직접 풀면 느림
- Levinson-Durbin: 행렬이 Toeplitz 구조임을 이용 → $O(p^2)$로 효율적 계산
- 재귀(recursion) 방식: p=1부터 시작해 차수를 하나씩 늘려가며 반사 계수(reflection coefficient) 계산
- 안정성 보장: 각 단계에서 |km| < 1이면 필터가 안정 (모든 극점이 단위원 내부)

> **직관**: 행렬 역행렬을 계산하면 $O(p^3)$ 시간이 걸립니다. Levinson-Durbin은 토플리츠 행렬의 특수 구조를 이용해 1차 예측기부터 시작해 매 단계 한 차수씩 올리며 이전 계수를 보정합니다. 결과적으로 $O(p^2)$로 줄여주는 재귀 알고리즘입니다.

**수식**:
$$
k_m = \frac{-r[m] - \sum_{j=1}^{m-1} a_j^{(m-1)} r[m-j]}{E_{m-1}}
$$
$$
a_j^{(m)} = a_j^{(m-1)} + k_m \cdot a_{m-j}^{(m-1)}
$$
$$
E_m = E_{m-1}(1 - k_m^2)
$$
기호 설명:
- $r[k]$: k번째 자기상관값 $r[k] = \sum_n x[n]x[n+k]$
- $k_m$: m번째 반사 계수(reflection coefficient)
- $E_m$: m차 예측 잔차 에너지 (0에 가까울수록 과적합)
- $a_j^{(m)}$: m차 LPC 계수

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `185-207`
```python
def _levinson_durbin(r: np.ndarray, order: int):
    """Levinson-Durbin 재귀로 LPC 계수 계산 (표준 구현)"""
    a = np.zeros(order + 1)
    a[0] = 1.0
    E = float(r[0])
    if E < 1e-12:
        return a, E
    for k in range(1, order + 1):
        # 반사 계수
        num = -float(r[k])
        for j in range(1, k):
            num -= a[j] * float(r[k - j])
        km = num / E
        # 계수 갱신
        a_new = a.copy()
        for j in range(1, k):
            a_new[j] = a[j] + km * a[k - j]
        a_new[k] = km
        a = a_new
        E = E * (1.0 - km * km)
        if E <= 0:
            break
    return a, E
```
설명: 교과서의 Levinson-Durbin 수식을 그대로 구현한 코드입니다. `r`은 자기상관 벡터, `order`는 LPC 차수(=16). `E <= 0` 체크는 수치 불안정 방지용 안전장치입니다.

**발표자 노트**:
- 자기상관 계산 자체도 O(N²)이지만, FFT로 O(N log N)으로 단축 가능 — 다음 슬라이드 참고
- 반사 계수 |km| ≥ 1이면 필터 불안정 → E ≤ 0에서 강제 종료

---

## 슬라이드 11: FFT 기반 자기상관 최적화

**핵심 개념**:
- 자기상관(autocorrelation): $r[k] = \sum_n x[n]x[n+k]$ — 신호가 자기 자신과 얼마나 닮았는지
- 직접 계산: O(N²) — N=13230(300ms@44.1kHz)이면 1.75억 번 연산
- FFT 기반: Wiener-Khinchin 정리 이용 $r = \text{IFFT}(|X|^2)$ → O(N log N)으로 약 10배 빠름
- 실시간 처리에서 이 차이가 체감됨

**수식**:
$$
r[k] = \mathcal{F}^{-1}\!\left( |X(f)|^2 \right)
$$
기호 설명:
- $X(f) = \mathcal{F}(x[n])$: FFT
- $|X(f)|^2$: 파워 스펙트럼
- $\mathcal{F}^{-1}$: 역 FFT
- Wiener-Khinchin 정리: 자기상관 함수와 파워 스펙트럼은 FFT 쌍(pair)

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `228-232`
```python
# FFT 기반 자기상관 (O(N log N))
fft_sz = 1 << (2 * n - 1).bit_length()
X  = np.fft.rfft(sig, n=fft_sz)
r_full = np.fft.irfft(X * np.conj(X))
r  = r_full[:order + 1].real
```
설명: `fft_sz`를 2의 거듭제곱으로 맞추는 이유는 FFT가 2의 거듭제곱 크기에서 가장 빠르기 때문입니다. `X * np.conj(X)` = $|X|^2$ (파워 스펙트럼). vad.py의 자기상관도 동일 방식을 사용합니다.

**발표자 노트**:
- `(2 * n - 1).bit_length()`로 "2n-1 이상인 최소 2의 거듭제곱" 계산 — 파이썬 비트 트릭
- 선형 자기상관(linear)이 아닌 원형(circular) 자기상관이 되지 않도록 패딩 크기 주의

---

## 슬라이드 12: AR 다항식 근 → 포먼트 주파수 변환

**핵심 개념**:
- LPC 계수 $[1, a_1, ..., a_p]$가 AR 다항식 $A(z)$를 정의
- $A(z)$의 근(복소수) = 성도 필터의 극점(pole)
- 극점이 단위원 근처일수록 강한 공명 = 포먼트!
- 극점의 각도 → Hz 변환, 극점의 반지름 → 대역폭(BW) 계산

**수식**:
$$
A(z) = 1 + a_1 z^{-1} + a_2 z^{-2} + \cdots + a_p z^{-p}
$$
$$
F_k = \frac{\angle z_k}{2\pi} \cdot f_s = \frac{\arctan2(\text{Im}(z_k), \text{Re}(z_k))}{2\pi} \cdot f_s
$$
$$
BW_k = \frac{-\ln|z_k|}{\pi} \cdot f_s
$$
기호 설명:
- $z_k$: A(z)의 k번째 복소 근(극점)
- $\angle z_k$: 극점의 위상각 (라디안)
- $F_k$: k번째 포먼트 주파수 (Hz)
- $BW_k$: 대역폭 — 좁을수록 선명한 공명
- $|z_k| < 1$: 단위원 내부 조건 (안정 필터)

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `240-252`
```python
# 다항식 근 (16차 → 빠름)
roots = np.roots(a)

# 단위원 내부 + 양의 허수부만 선택
roots = roots[np.abs(roots) < 1.0]
roots = roots[np.imag(roots) >= 0]

if len(roots) == 0:
    return None, None, None

angles = np.arctan2(np.imag(roots), np.real(roots))
freqs  = angles * sr / (2.0 * np.pi)
bws    = -np.log(np.abs(roots)) * sr / np.pi
```
설명: `np.roots(a)`가 핵심 — 16차 다항식의 근을 계산합니다. "양의 허수부만 선택"은 켤레 복소수 쌍 중 하나만 선택(중복 방지). `arctan2`로 위상각 → Hz 변환, `-log(|z|)`로 BW 계산.

**발표자 노트**:
- 근이 단위원 밖에 있으면 → 불안정한 극점 → 실제 포먼트가 아님 → 제거
- 허수부 = 0인 근 = 실수 근 = 포먼트 아님 → `imag >= 0` 조건으로 걸러냄

---

## 슬라이드 13: Burg 알고리즘 — 양방향 LPC

**핵심 개념**:
- 표준 Levinson-Durbin: 앞→뒤 방향으로만 예측 (순방향)
- Burg 알고리즘: 순방향 + 역방향 예측 오차를 동시에 최소화
- 결과: Levinson-Durbin보다 짧은 데이터에서 더 안정적, 스펙트럼 왜곡 적음
- Praat이 "To Formant (burg)"에서 이 방법을 사용하는 이유

**수식**:
$$
k_m = \frac{-2\sum_{n=m}^{N-1} e_f^{(m-1)}[n] \cdot e_b^{(m-1)}[n-1]}{\sum_{n=m}^{N-1} \left(|e_f^{(m-1)}[n]|^2 + |e_b^{(m-1)}[n-1]|^2\right)}
$$
기호 설명:
- $e_f^{(m-1)}[n]$: m-1차 순방향 예측 오차
- $e_b^{(m-1)}[n]$: m-1차 역방향 예측 오차
- $k_m$: m번째 반사 계수 (|km| < 1 자동 보장)
- Levinson-Durbin 대비: 자기상관 추정이 필요 없음 → 짧은 신호에 유리

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `66-103`
```python
def praat_burg_formants(chunk_pe: np.ndarray, gender: str):
    """
    멀티-ceiling Praat Burg → BW 가중 중앙값
    Returns: (f1, f2, f3) or (None, None, None)
    """
    p = PARAMS[gender]
    snd = parselmouth.Sound(
        chunk_pe.astype(np.float64),
        sampling_frequency=float(SAMPLE_RATE),
    )
    for ceiling in FORMANT_CEILINGS:
        fmt = call(snd, "To Formant (burg)",
                   0.0, p["max_formants"], ceiling,
                   p["window_length"], p["pre_emphasis"])
```
설명: `parselmouth`를 통해 Praat의 Burg LPC를 호출합니다. Praat은 내부에서 Burg 알고리즘으로 LPC 계수를 계산하므로, 수동으로 구현하지 않아도 됩니다. `pre_emphasis=50` (≈0Hz)은 수동 프리엠퍼시스 이미 적용했으므로 내부 프리엠퍼시스를 최소화.

**발표자 노트**:
- Praat Burg = 안정적이고 검증된 표준 방법 → Method 1로 사용
- Burg의 단점: 스펙트럼 분할(spectral line splitting) 현상 — 멀티-ceiling으로 보완
- **왜 여성/고음 화자에 유리한가**: Burg는 프레임 양 끝을 0으로 가정하지 않고 양방향 예측을 수행합니다. 고음(짧은 주기) 화자는 같은 프레임 길이 안에 더 많은 주기가 들어오지만, 일반 LPC는 프레임 끝의 0 가정 때문에 추정이 불안정해집니다. Burg는 이 가정이 없어 짧은 프레임에서도 안정적으로 포먼트를 추정합니다.

---

## 슬라이드 14: Multi-Ceiling 전략 — 화자 성도 길이 차이 대응

**핵심 개념**:
- 문제: 성인 남성(성도 약 17cm) vs 여성(약 14cm) vs 어린이(더 짧음) — 성도 길이에 따라 포먼트 주파수 전체가 올라가거나 내려감
- Ceiling(상한값): Praat Formant 분석에서 탐색할 최고 주파수 — 너무 낮으면 F2·F3를 놓침, 너무 높으면 가짜 포먼트가 생김
- Multi-Ceiling: 여러 상한값으로 분석 후 대역폭(BW)이 가장 좁은 결과 선택
- BW가 좁다 = 공명이 선명하다 = 실제 포먼트일 가능성 높다

**수식**:
$$
\text{best\_ceiling} = \underset{c \in C}{\operatorname{argmin}} \frac{1}{2}\left(BW_{F1}(c) + BW_{F2}(c)\right)
$$
기호 설명:
- $C$: 시도할 Ceiling 값 집합 (현재: {3500, 4800, 5200} Hz, 튜닝 포인트)
- $BW_{F1}(c)$, $BW_{F2}(c)$: ceiling c에서 F1, F2의 대역폭
- F1·F2의 평균 BW를 최소화하는 ceiling을 선택

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `64-101`
```python
best_f, best_bw, best_score = None, None, float('inf')

for ceiling in FORMANT_CEILINGS:   # [3500, 4800, 5200] Hz
    fmt = call(snd, "To Formant (burg)",
               0.0, p["max_formants"], ceiling, ...)
    # ... 포먼트 추출 ...
    bw12 = [float(np.mean(bw_vals[fn]))
            for fn in [1, 2] if bw_vals[fn]]
    if bw12 and f_vals[1] and f_vals[2]:
        score = float(np.mean(bw12))
        if score < best_score:        # BW 최소 = 가장 선명한 공명
            best_score = score
            best_f, best_bw = f_vals, bw_vals
```
설명: `FORMANT_CEILINGS = [3500, 4800, 5200]` (config.py 61번 줄). 세 ceiling으로 분석 후 F1+F2 평균 BW가 가장 작은 ceiling의 결과를 채택합니다. 낮은 ceiling(3500)은 특히 한국어 /우, 오/처럼 F2가 낮은 모음에서 도움이 됩니다. (현재 기본값, 튜닝 포인트)

**발표자 노트**:
- FastTrack(Praat 플러그인)의 아이디어를 실시간 버전으로 단순화
- Ceiling 3500 추가 이유: /우/ F2 ≈ 660Hz — 높은 ceiling에서 잡기 어려움

---

## 슬라이드 15: pyworld CheapTrick — Source-Filter 분리 없는 스펙트럼 피크 탐색

**핵심 개념**:
- LPC: Source-Filter 모델을 가정하고 V(f)를 분리 추출
- CheapTrick(pyworld): Source-Filter 분리 없이 스펙트럼 포락선에서 직접 피크를 찾음
- 방법: F0를 이용해 고조파 영향을 줄인 스펙트럼 포락선 계산 → 봉우리 = 포먼트
- 장점: LPC 차수 선택 불필요 / 단점: F0가 없으면 사용 불가

**수식**:
$$
\hat{S}(f) = \text{CheapTrick envelope given } f_0
$$
$$
\text{포먼트} = \underset{f \in [F_\text{lo}, F_\text{hi}]}{\operatorname{argmax}} \hat{S}(f)
$$
기호 설명:
- $\hat{S}(f)$: CheapTrick 스펙트럼 포락선
- $f_0$: 기본 주파수 (F0, 이미 DIO+StoneMask로 추출된 값 재활용)
- 피크 탐색 범위: 성별별 F1/F2/F3 범위

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `133-178`
```python
def cheaptrick_formants(x_f64, f0_arr, t_arr, gender):
    sp = pw.cheaptrick(x_f64, f0_arr, t_arr, float(SAMPLE_RATE))
    mid = sp.shape[0] // 2
    envelope = sp[mid]   # 중간 프레임 (steady-state 구간)

    log_env = np.log(np.maximum(envelope, 1e-10))

    # 최소 피크 간격: 약 80Hz
    bin_per_hz = n_half / (SAMPLE_RATE / 2.0)
    min_dist   = max(1, int(80 * bin_per_hz))

    peaks, props = find_peaks(
        log_env,
        distance=min_dist,
        prominence=0.3,
    )
    peak_freqs = freqs[peaks]
    # 성별별 범위에서 가장 두드러진 피크 선택
    for fn, rk in [(1, "f1_range"), (2, "f2_range"), (3, "f3_range")]:
        lo, hi = p[rk]
        mask = (peak_freqs >= lo) & (peak_freqs <= hi)
        if mask.any():
            best_i = int(np.argmax(prominences[mask]))
            out[i] = float(peak_freqs[mask][best_i])
```
설명: pyworld의 `cheaptrick()`이 스펙트럼 포락선을 반환합니다. 중간 프레임을 선택하는 이유는 모음의 steady-state(안정 구간)이 중간에 있기 때문. `prominence=0.3`은 약한 봉우리를 걸러내는 임계값 (현재 기본값, 튜닝 포인트).

**발표자 노트**:
- f0_arr, t_arr는 DIO+StoneMask에서 이미 계산된 값 재활용 → 별도 F0 계산 없음
- DC 제거만 하고 프리엠퍼시스는 적용하지 않음 (CheapTrick에는 불필요)

---

## 슬라이드 16: 3-방법 앙상블 — 투표로 신뢰도 향상

**핵심 개념**:
- 세 방법(Praat Burg, CheapTrick, scipy LPC)은 각자 장단점이 다름
- 앙상블: 2개 이상이 ±허용 오차 Hz 이내에서 동의하면 그 값 채택
- 동의: 신뢰도 0.5~1.0 / 불일치: 중앙값 채택 + 신뢰도 0.1 / 단일 유효: 신뢰도 0.25
- F1/F2/F3별로 허용 오차 다름: F1=±80Hz, F2=±120Hz, F3=±160Hz (현재 기본값, 튜닝 포인트)

**수식**:
$$
\text{agreed\_vals} = \{(v_i, v_j) : |v_i - v_j| \le \text{tol}(F_n)\}
$$
$$
\text{result} = \text{median}(\text{agreed\_vals}), \quad \text{score} = \frac{|\text{agreed\_vals}|}{|\text{valid}| \times 2}
$$
기호 설명:
- $\text{tol}(F_n)$: 포먼트 번호별 허용 오차 (F1=80, F2=120, F3=160 Hz)
- agreed_vals: 동의 조건을 만족하는 값 쌍
- score: 0~1 신뢰도 점수

**내 코드에서는**:
파일: `formant_ensemble.py`
줄번호: `283-312`
```python
AGREE_TOL = {1: 80, 2: 120, 3: 160}   # F1 < F2 < F3 순으로 허용 폭 확대

def _ensemble_one(candidates: list, fn: int) -> tuple:
    tol   = AGREE_TOL[fn]
    valid = [f for f in candidates if f is not None and f > 50]

    if not valid:
        return None, 0.0
    if len(valid) == 1:
        return valid[0], 0.25   # 단일 방법 = 낮은 신뢰

    # 2-이상 동의 검사
    agreed_vals = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if abs(valid[i] - valid[j]) <= tol:
                agreed_vals.extend([valid[i], valid[j]])

    if agreed_vals:
        value = float(np.median(agreed_vals))
        score = len(agreed_vals) / (len(valid) * 2.0)
    else:
        value = float(np.median(valid))
        score = 0.1

    return value, score
```
설명: `candidates`는 [Praat값, CheapTrick값, scipy값]. 쌍별로 차이를 비교해 tol 이내면 동의로 판정. F2의 tol(120Hz)이 F1(80Hz)보다 넓은 이유: F2 범위 자체가 더 넓어 방법 간 오차도 더 크게 나타납니다.

**발표자 노트**:
- 앙상블 아이디어: 단일 방법의 실패를 다른 방법이 보완
- 신뢰도(agreement) 점수는 Kalman 필터의 측정 노이즈로 연결됨 → 다음 슬라이드

---

## 슬라이드 17: Part 1 요약 및 파이프라인 전체 조감

**핵심 개념**:
- Part 1에서 배운 포먼트 추출 파이프라인 7단계 정리
- 각 단계에서 사용한 방법과 그 이유
- 다음 파트(F0, VAD)와의 연결고리
- VoiceTypo의 포먼트 추출 = Praat Burg + CheapTrick + scipy LPC 의 앙상블

**다이어그램**:
[설명: 수직 흐름도(플로우차트). 단계별로 박스를 그리고 각 박스에 번호와 내용 기재. ① "마이크 입력 (44100Hz, float32)" → ② "VAD 판단 (Part 3에서 설명)" → [음성 아님 → 종료] / [음성 → 계속] → ③ "DC 제거 + 프리엠퍼시스 (α=0.97)" → ④ "300ms 청크 준비" → 세 개의 병렬 박스: ④-a "Praat Burg LPC (Multi-ceiling: 3500/4800/5200Hz)", ④-b "pyworld CheapTrick (F0 재활용)", ④-c "scipy LPC (Levinson-Durbin, p=16)" → ⑤ "3-방법 앙상블 (동의 ±80~160Hz)" → ⑥ "Kalman 필터 스무딩" → ⑦ "출력: F1, F2, F3 (Hz) + 신뢰도". 각 박스에 파일명을 작게 표기: ③~④ = formant_engine.py, ④-a/b/c = formant_ensemble.py, ⑤~⑦ = formant_engine.py.]

**발표자 노트**:
- Part 2에서는 이 파이프라인의 ② VAD에서 사용하는 F0(pyworld DIO+StoneMask) 설명
- Part 3에서는 VAD(음성 활동 탐지) 전체 설명
- 포먼트 추출의 핵심 철학: "한 방법을 완벽하게 만들기보다 세 방법을 앙상블"

---

*Part 1 완료 — 슬라이드 17장*
*다음: Part 2 (F0 추출, 10장) — 재원님의 검토 후 진행*
