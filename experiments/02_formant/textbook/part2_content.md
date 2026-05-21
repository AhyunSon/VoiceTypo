# VoiceTypo 교재 — Part 2: F0 추출 원리
저자: 허재원 | 참조 코드: realtime_formant 프로젝트

---

## 슬라이드 1: Part 2 표지 — F0란 무엇인가?

**학습 목표**:
- F0(기본 주파수, fundamental frequency)가 물리적으로 무엇인지 설명할 수 있다
- 성별·연령별 F0 범위를 알고, VoiceTypo에서 어떻게 활용하는지 이해한다
- pyworld DIO와 StoneMask의 역할과 동작 원리를 설명할 수 있다
- F0가 포먼트 추출 파이프라인에서 어떤 게이트 역할을 하는지 이해한다

**핵심 개념**:
- F0 = 성대 진동 주파수 = 우리가 "음높이(pitch)"라고 느끼는 것
- F0는 Part 1의 포먼트(성도 공명)와 완전히 다른 물리량
- VoiceTypo에서 F0의 역할: ① 유성음/무성음 판별 게이트, ② 성별 자동 판단, ③ CheapTrick 포먼트 추출 입력
- 사용 알고리즘: DIO(빠름) + StoneMask(정밀 보정)

**발표자 노트**:
- "포먼트가 성도의 모양이라면, F0는 성대가 얼마나 빠르게 떠는가"로 직관을 잡아주기
- Part 2는 Part 1보다 짧지만 포먼트 파이프라인의 입구(게이트)이므로 중요

---

## 슬라이드 2: F0의 물리적 정체 — 성대 진동

**핵심 개념**:
- 성대(vocal folds): 후두에 위치한 한 쌍의 막 — 숨이 통과할 때 진동
- 진동 주기 T0 = 한 번 열렸다 닫히는 데 걸리는 시간
- F0 = 1/T0 — 초당 진동 횟수(Hz)
- 낮은 F0 = 낮은 음 / 높은 F0 = 높은 음
- 무성음(s, f, k, p): 성대 진동 없음 → F0 = 0

**수식**:
$$
F_0 = \frac{1}{T_0}
$$
기호 설명:
- $F_0$: 기본 주파수(Hz) — 음높이의 물리적 단위
- $T_0$: 기본 주기(초) — 성대가 한 번 진동하는 데 걸리는 시간
- 예: F0 = 125Hz → T0 = 8ms (남성 일반 발화 음높이)
- 예: F0 = 220Hz → T0 ≈ 4.5ms (여성 일반 발화 음높이)

**다이어그램**:
[설명: 두 행 파형 비교. 위 행 "남성 F0=125Hz": x축=시간(0~32ms), y축=진폭. 약 8ms 간격으로 글로탈 펄스(뾰족한 삼각형 모양) 4개. 각 펄스 사이에 "T0=8ms" 양방향 화살표. 아래 행 "여성 F0=220Hz": 같은 시간 축에 약 4.5ms 간격으로 펄스 7개. 두 행 아래에 "F0가 높을수록 같은 시간에 펄스가 더 많다" 설명 텍스트.]

**발표자 노트**:
- 성대 진동이 만드는 파형 = 글로탈 펄스(glottal pulse) — 이것이 Part 1의 "Source G(f)"
- F0는 성도 형태(포먼트)와 독립적 — 같은 모음 /아/를 낮게도, 높게도 발음할 수 있음

---

## 슬라이드 3: 성별·연령별 F0 범위

**핵심 개념**:
- F0 범위는 성별·연령에 따라 뚜렷하게 다름
- 남성: 80~180Hz (일반 발화 기준, 베이스 가수는 65Hz까지)
- 여성: 165~255Hz (일반 발화 기준)
- 아이: 250~400Hz (성도가 짧아서 전체적으로 높음)
- 남녀 경계(165Hz 근처): 겹치는 구간 존재 → 단순 임계값으로 성별 판단 시 오류 가능

**다이어그램**:
[설명: 수평 막대 그래프(범위 표시). x축=주파수(Hz), 50~450Hz. 세 개의 수평 막대: ① "남성(베이스~테너)" 50~180Hz, 파란색. ② "여성(알토~소프라노)" 165~400Hz, 분홍색. ③ "아이" 250~450Hz, 노란색. 165~180Hz 구간에 수직 점선을 그어 "성별 겹침 구간" 라벨 표시. VoiceTypo 판단 임계값 165Hz 위치에 빨간 수직선 + "GENDER_THRESH_HZ=165" 라벨.]

**내 코드에서는**:
파일: `config.py`
줄번호: `24`
```python
GENDER_THRESH_HZ = 165      # F0 < 165 Hz → 남성
```
파일: `formant_engine.py`
줄번호: `143-149`
```python
f0_arr, t_arr = pw.dio(
    x, float(SAMPLE_RATE),
    f0_floor=50.0,    # 고정 하한: 초저음 남성 베이스 포착
    f0_ceil=500.0,    # 고정 상한: 전 성별 커버
    frame_period=PYWORLD_FRAME_PERIOD,
)
```
설명: DIO 탐색 범위를 50~500Hz로 고정한 이유가 중요합니다. 만약 초기 gender="female"일 때 f0_floor=150Hz로 설정하면 F0=65~100Hz인 초저음 남성 목소리를 아예 감지하지 못해, gender가 영원히 "female"에 고정되는 순환 오류가 발생합니다. (코드 주석 참고)

**발표자 노트**:
- VoiceTypo에서 165Hz 임계값은 "현재 기본값, 튜닝 포인트" — 화자에 따라 조정 가능
- 베이스 가수처럼 F0가 매우 낮은 경우를 위해 f0_floor=50Hz로 설정

---

## 슬라이드 4: pyworld DIO 알고리즘 원리

**핵심 개념**:
- DIO(Distributed Inline-filter Operation): pyworld 라이브러리의 F0 추출 알고리즘
- 핵심 아이디어: 저역통과 필터 뱅크(filter bank)를 통해 F0 후보를 탐색
- 저역통과 필터 → 고조파 제거 → 남은 성분의 주기 = F0
- 장점: 빠름, 실시간 처리에 적합 / 단점: 약간의 오차 → StoneMask로 보정

**수식**:
$$
F_0^{DIO}(t) = \underset{f \in [f_{\min}, f_{\max}]}{\operatorname{argmax}} \; \text{reliability}(f, t)
$$
기호 설명:
- $f_{\min}$, $f_{\max}$: 탐색 범위 (VoiceTypo: 50~500Hz)
- $\text{reliability}(f, t)$: 시각 $t$에서 후보 F0 $f$의 신뢰도
- 신뢰도: 필터 출력 파형의 주기성으로 계산
- frame_period: 프레임 간격 (현재 기본값: 10ms, 튜닝 포인트)

**다이어그램**:
[설명: DIO 처리 흐름도. 입력 "음성 신호 x[n]" → "여러 차단 주파수의 저역통과 필터 뱅크 (LPF₁, LPF₂, ..., LPFₙ)" → 각 필터 출력에서 "주기 탐색" → "가장 신뢰도 높은 주기 선택" → "F0 후보 시퀀스". 저역통과 필터를 여러 개 병렬로 배치하고 그 결과들이 합쳐지는 모양으로 그림. 오른쪽 끝에 F0 시계열 그래프(시간-Hz축): 모음 구간에서 약 150Hz 정도의 가로선이 나타남.]

**내 코드에서는**:
파일: `formant_engine.py`
줄번호: `143-150`
```python
f0_arr, t_arr = pw.dio(
    x, float(SAMPLE_RATE),
    f0_floor=50.0,    # 고정 하한: 초저음 남성 베이스 포착
    f0_ceil=500.0,    # 고정 상한: 전 성별 커버
    frame_period=PYWORLD_FRAME_PERIOD,   # 현재 기본값: 10ms
)
f0_arr = pw.stonemask(x, f0_arr, t_arr, float(SAMPLE_RATE))
```
설명: `pw.dio()`가 초벌 F0를 반환하고, `pw.stonemask()`가 이를 정밀 보정합니다. `frame_period=10ms`이므로 300ms 청크에서 약 30개의 F0 프레임이 나옵니다. (현재 기본값, 튜닝 포인트)

**발표자 노트**:
- DIO 이전에 사용된 방법: YIN, SWIPE, 자기상관 기반 등 — DIO는 이들보다 빠른 편
- `f0_arr`에서 0 값 = 무성음 프레임, 양수 값 = 유성음 프레임

---

## 슬라이드 5: StoneMask — DIO 오차 정밀 보정

**핵심 개념**:
- DIO의 약점: F0 후보를 빠르게 추정하지만 정확도가 약 ±수 Hz 수준
- StoneMask: DIO 출력을 입력으로 받아 더 정밀한 F0로 재추정
- 방법: DIO 추정값 주변의 스펙트럼을 분석해 실제 기본 주기를 재계산
- 결과: DIO 단독 대비 정확도가 크게 향상, 연산량 증가는 적음

**수식**:
$$
F_0^{refined}(t) = \text{StoneMask}\!\left(x,\; F_0^{DIO}(t),\; t\right)
$$
기호 설명:
- $F_0^{DIO}(t)$: DIO가 추정한 초벌 F0 시퀀스
- $F_0^{refined}(t)$: StoneMask가 보정한 최종 F0 시퀀스
- StoneMask는 DIO 없이 단독 사용 불가 — 반드시 DIO → StoneMask 순서

**다이어그램**:
[설명: 두 개의 F0 시계열 그래프를 위아래로 배치. 위 "DIO 출력": 시간축(0~300ms), F0값이 약간 들쭉날쭉한 계단 모양 (정수 격자에 snap된 느낌). 아래 "StoneMask 보정 후": 동일 시간축, F0값이 더 부드럽고 연속적으로 연결된 모양. 두 그래프 오른쪽에 "오차 ↓" 화살표. 색상: DIO=회색, StoneMask=파랑.]

**내 코드에서는**:
파일: `formant_engine.py`
줄번호: `150`
```python
f0_arr = pw.stonemask(x, f0_arr, t_arr, float(SAMPLE_RATE))
```
설명: 딱 한 줄이지만 DIO 오차를 크게 줄여줍니다. `t_arr`는 DIO에서 반환된 시간축 배열 — StoneMask가 각 프레임에서 스펙트럼을 재분석하는 데 사용합니다.

**발표자 노트**:
- pyworld의 권장 사용법: 반드시 DIO + StoneMask 세트로 사용
- StoneMask만 단독 사용은 불가 — 항상 DIO 결과를 먼저 얻어야 함

---

## 슬라이드 6: Harvest — 더 정확한 F0 추출기

**핵심 개념**:
- Harvest: pyworld의 또 다른 F0 추출 알고리즘 — DIO보다 정확하지만 느림
- 원리: 로그 스펙트럼의 조화 평균(harmonic mean)을 이용해 F0 추정
- DIO 대비: 동일 입력에서 F0 오차가 작고, 보이스리스 구간 탐지가 더 정확
- 실시간 처리: 300ms 청크에서 Harvest는 DIO 대비 약 3~5배 느림

**수식**:
$$
F_0^{Harvest}(t) = \underset{f}{\operatorname{argmin}} \sum_{h=1}^{H} \left| \log S(hf, t) - \bar{L}(f, t) \right|^2
$$
기호 설명:
- $S(hf, t)$: 시각 $t$에서 $h$번째 고조파 $hf$의 스펙트럼 값
- $\bar{L}(f, t)$: 고조파 스펙트럼 로그값의 평균
- $H$: 고려하는 고조파 최대 개수
- 직관: F0의 정수 배(고조파)들이 스펙트럼에서 가장 고르게 분포하는 주파수를 찾음

**Python 구현**:
```python
import pyworld as pw
import numpy as np

x = signal.astype(np.float64)
sr = 44100.0

# Harvest 사용 (정확하지만 느림)
f0_arr, t_arr = pw.harvest(x, sr, f0_floor=50.0, f0_ceil=500.0)
# DIO와 달리 StoneMask 없이도 높은 정확도
```

**발표자 노트**:
- VoiceTypo 현재 버전은 실시간 속도 우선으로 DIO+StoneMask를 사용
- 오프라인 분석이나 정확도 우선 모드로 전환할 때 Harvest를 고려할 수 있음
- Harvest는 DIO처럼 StoneMask 보정이 필수는 아님

---

## 슬라이드 7: DIO vs Harvest — 실시간 vs 정확도 트레이드오프

**핵심 개념**:
- 두 알고리즘은 목적에 따라 선택해야 함
- DIO + StoneMask: 실시간 처리 적합, 오차 약간 있지만 속도 빠름
- Harvest: 오프라인/고정밀 분석 적합, 느리지만 정확
- VoiceTypo의 선택 이유: 실시간 피드백(30fps)을 위해 DIO+StoneMask 선택

**다이어그램**:
[설명: 2×2 표(테이블). 가로축 헤더: "DIO+StoneMask" / "Harvest". 세로축 항목: "속도" / "F0 정확도" / "무성음 판단" / "StoneMask 필요" / "VoiceTypo 사용". 셀 내용: DIO+StoneMask—속도:빠름(★★★★), 정확도:보통(★★★), 무성음:보통(★★★), StoneMask:필요, 사용:현재 사용. Harvest—속도:느림(★★), 정확도:높음(★★★★★), 무성음:우수(★★★★), StoneMask:불필요, 사용:미사용. VoiceTypo 행에 체크 표시를 DIO+StoneMask 칸에.]

**내 코드에서는**:
파일: `config.py`
줄번호: `48-50`
```python
PYWORLD_VOICED_FRAC_MIN = 0.25   # 청크 내 유성음 프레임 비율 임계값 (완화: 초저음 남성 포착)
PYWORLD_FRAME_PERIOD    = 10.0   # ms — DIO 분석 프레임 간격
```
설명: `PYWORLD_FRAME_PERIOD=10ms`는 DIO의 시간 해상도입니다. 300ms 청크 = 약 30프레임. `PYWORLD_VOICED_FRAC_MIN=0.25`는 30프레임 중 8개 이상이 유성음이어야 "이 청크는 음성"으로 판단합니다. (현재 기본값, 튜닝 포인트)

**발표자 노트**:
- PYWORLD_VOICED_FRAC_MIN을 높이면 보수적 판단(잡음 감소), 낮추면 유연한 판단(초저음 포착)
- 현재 0.25는 낮춰진 값 — 초저음 남성 베이스(F0≈65Hz) 포착을 위해 0.5→0.25로 완화

---

## 슬라이드 8: 유성음 판단과 F0=0 처리

**핵심 개념**:
- DIO+StoneMask 출력 f0_arr: 유성음 프레임 = 양수(Hz), 무성음 프레임 = 0
- VoiceTypo의 유성음 판단: 청크 내 "유성음 프레임 비율"이 임계값 이상이면 유성음
- is_voiced=False이면 포먼트 추출 자체를 건너뜀 → 불필요한 연산 방지
- 이 메커니즘이 파이프라인의 "게이트(gate)"

**수식**:
$$
\text{voiced\_frac} = \frac{\sum_{t} \mathbf{1}[f_0(t) > 0]}{N_\text{frames}}
$$
$$
\text{is\_voiced} = (\text{voiced\_frac} \ge \theta)
$$
기호 설명:
- $\mathbf{1}[\cdot]$: 조건이 참이면 1, 거짓이면 0 (지시 함수)
- $N_\text{frames}$: 청크 내 전체 프레임 수 (300ms / 10ms = 30)
- $\theta$: 유성음 판단 임계값 (현재 기본값: 0.25, 튜닝 포인트)

**내 코드에서는**:
파일: `formant_engine.py`
줄번호: `152-169`
```python
voiced_mask = f0_arr > 0
voiced_frac = float(voiced_mask.mean()) if len(f0_arr) > 0 else 0.0
f0 = float(np.mean(f0_arr[voiced_mask])) if voiced_mask.any() else None
is_voiced = voiced_frac >= PYWORLD_VOICED_FRAC_MIN

hnr = None
if is_voiced:
    ap = pw.d4c(x, f0_arr, t_arr, float(SAMPLE_RATE))
    n_bins   = ap.shape[1]
    low_bins = max(1, int(n_bins * 2000.0 / (SAMPLE_RATE / 2.0)))
    ap_low   = ap[voiced_mask, :low_bins]
    mean_ap  = float(np.mean(ap_low))
    hnr = float(-10.0 * np.log10(max(mean_ap, 1e-6)))

return f0, hnr, is_voiced, f0_arr, t_arr
```
설명: `f0_arr > 0`으로 유성음 프레임을 마스킹합니다. `f0`는 유성음 프레임들의 평균 F0. `is_voiced=False`이면 `formant_engine.py` 245번 줄에서 즉시 반환 — 포먼트 계산을 건너뜁니다. D4C는 비주기성(aperiodicity)을 계산해 HNR(조화-잡음비)을 근사합니다.

**발표자 노트**:
- F0=0 처리가 없으면 무성음 구간에서도 포먼트를 추출하려 시도 → 잡음 값 출력
- HNR(Harmonic-to-Noise Ratio): 값이 낮으면 잡음 많음 → Kalman 필터 신뢰도 낮춤

---

## 슬라이드 9: HNR — 음성 품질과 포먼트 신뢰도 연결

**핵심 개념**:
- HNR(Harmonic-to-Noise Ratio, 조화-잡음비): 유성음 성분 대비 잡음 성분의 비율
- 높은 HNR = 맑고 안정적인 음성 / 낮은 HNR = 쉰 목소리, 잡음 많은 환경
- VoiceTypo 활용: HNR이 임계값 미만이면 Kalman 필터의 측정 노이즈를 2배로 설정 → 새 측정값을 덜 신뢰
- D4C(pyworld): aperiodicity를 계산해 HNR을 근사 — $HNR \approx -10 \log_{10}(\text{mean aperiodicity})$

**수식**:
$$
HNR = -10 \log_{10}\!\left(\frac{E_\text{noise}}{E_\text{harmonic}}\right) \approx -10\log_{10}(\bar{a}_\text{low})
$$
기호 설명:
- $E_\text{noise}$: 잡음 에너지
- $E_\text{harmonic}$: 고조파(조화) 에너지
- $\bar{a}_\text{low}$: 2kHz 이하 저주파 대역의 평균 비주기성(D4C 출력)
- 단위: dB — 높을수록 음성 품질 좋음 (HNR_MIN_DB = 5.0 dB, 튜닝 포인트)

**내 코드에서는**:
파일: `formant_engine.py`
줄번호: `276-281`
```python
# ── 4. HNR 저품질 시 BW 증폭 → Kalman 신뢰도 낮춤 ────
if hnr is not None and hnr < HNR_MIN_DB:
    for fn in [1, 2, 3]:
        if bw_avg[fn] is not None:
            bw_avg[fn] = bw_avg[fn] * 2.0
```
파일: `config.py`
줄번호: `42-43`
```python
HNR_MIN_DB   = 5.0    # HNR 이 미만이면 Kalman 측정 노이즈 2배
HNR_VOICE_MIN = 2.0   # 이 미만이면 숨소리로 간주
```
설명: HNR < 5.0dB이면 대역폭(BW)을 2배로 늘려 Kalman 필터가 이 프레임의 측정값을 덜 신뢰하게 만듭니다. BW가 크면 Kalman R(측정 노이즈)가 커지므로, 새 측정값 대신 이전 예측값을 더 많이 유지합니다. (현재 기본값, 튜닝 포인트)

**발표자 노트**:
- D4C = D4C(Deterministic plus stochastic model of residual) — pyworld의 세 번째 모듈
- HNR 계산에 전 주파수가 아닌 2kHz 이하 저주파만 쓰는 이유: 성도 공명과 관련된 대역이 저주파에 집중되어 있기 때문

---

## 슬라이드 10: Part 2 요약 — F0 파이프라인 전체 흐름

**핵심 개념**:
- Part 2에서 배운 F0 추출 파이프라인 정리
- F0는 독립적 정보이자 포먼트 추출 파이프라인의 입구(게이트)
- DIO+StoneMask → voiced_frac → is_voiced → (True이면) 포먼트 추출 진행
- HNR은 F0 추출의 부산물이지만 포먼트 신뢰도 조정에 활용

**다이어그램**:
[설명: 수직 흐름도. ① "마이크 입력 청크 (300ms, 44100Hz)" → ② "pw.dio() → f0_arr, t_arr (초벌 F0)" → ③ "pw.stonemask() → f0_arr 보정 (정밀 F0)" → ④ "voiced_frac 계산: 유성음 프레임 비율" → 분기: [voiced_frac < 0.25] → "is_voiced=False → 포먼트 추출 건너뜀 → None 반환" / [voiced_frac ≥ 0.25] → "is_voiced=True" → ⑤ "pw.d4c() → aperiodicity → HNR 계산" → ⑥ "F0 평균값 출력 + is_voiced=True" → "포먼트 추출로 진행 (Part 1 파이프라인)". 각 박스 오른쪽에 파일명:줄번호 작게 표기. ②③④ = formant_engine.py:143-155, ⑤ = formant_engine.py:158-165.]

**발표자 노트**:
- F0의 역할 세 가지 요약: ① 유성음 게이트, ② 성별 자동 판단(165Hz 임계), ③ CheapTrick 입력
- Part 3에서는 pyworld와 별개로 동작하는 VAD(AdaptiveVAD) 설명
- 두 시스템(pyworld F0 기반 + AdaptiveVAD)이 어떻게 협력하는지는 main.py에서 확인 가능

---

*Part 2 완료 — 슬라이드 10장*
*다음: Part 3 (VAD, 8장) — 재원님의 검토 후 진행*
