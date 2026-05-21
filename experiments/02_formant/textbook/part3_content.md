# VoiceTypo 교재 — Part 3: VAD (음성 활동 감지)
저자: 허재원 | 참조 코드: realtime_formant 프로젝트

---

## 슬라이드 1: Part 3 표지 — VAD란 무엇인가?

**학습 목표**:
- VAD(Voice Activity Detection, 음성 활동 감지)가 왜 필요한지 설명할 수 있다
- RMS, 자기상관, ZCR 세 조건이 각각 무엇을 걸러내는지 구분할 수 있다
- 세 조건을 AND로 결합하는 보수적 판정 방식의 의미를 이해한다
- 적응형 노이즈 바닥 갱신과 Onset/Hangover 안정화가 왜 필요한지 설명할 수 있다

**핵심 개념**:
- VAD = "지금 이 순간 음성이 있는가?"를 판단하는 이진(binary) 분류기
- 목적: 무음·잡음 구간에서 포먼트 추출을 건너뛰어 계산 낭비와 오류 방지
- VoiceTypo의 VAD는 세 가지 음향 특징의 AND 조건 — 하나라도 실패하면 무음 판정
- 보너스: 적응형 노이즈 바닥 갱신 + Onset/Hangover 상태 머신으로 안정화

**다이어그램**:
[설명: VAD의 역할을 보여주는 시계열 그림. x축=시간(0~3초), y축=진폭. 파형이 그려진 긴 스트립. 구간별로 배경색: 침묵 구간(회색 배경) + 발화 구간(파란 배경). 발화 구간 위에 "VAD=ON" 라벨, 침묵 구간 위에 "VAD=OFF" 라벨. "VAD=ON 구간에서만 포먼트 추출" 화살표 텍스트를 발화 구간에 연결.]

**발표자 노트**:
- "VAD 없이 항상 포먼트를 추출하면 어떻게 될까?" 질문으로 시작 — 잡음·숨소리가 포먼트 값으로 들어옴
- Part 2에서 배운 pyworld 유성음 판단과 VAD는 독립적으로 동작 — 2중 게이트

---

## 슬라이드 2: VAD가 필요한 이유 + 캘리브레이션

**핵심 개념**:
- 음성 신호는 "말하는 구간"과 "침묵·잡음 구간"이 뒤섞여 있음
- 침묵 구간에서 포먼트를 추출하면: 잡음이 F1/F2 값으로 표시 → 화면이 흔들림
- 계산 낭비: 포먼트 추출(Praat Burg + CheapTrick + LPC)은 연산 비용이 높음 → VAD가 통과시킨 프레임에만 수행
- 캘리브레이션(calibration): 시작 2초 동안 주변 잡음 RMS를 수집 → 개인·환경에 맞는 임계값 자동 설정

**수식**:
$$
\text{noise\_rms} = \text{percentile}_{80}(RMS_\text{calib frames})
$$
$$
\text{threshold} = \text{noise\_rms} \times \text{VAD\_RMS\_MULT}
$$
기호 설명:
- $RMS_\text{calib frames}$: 초기 2초 동안 수집된 RMS 값 목록
- $\text{percentile}_{80}$: 상위 20% 극단값을 제거한 안정적 추정
- $\text{VAD\_RMS\_MULT}$: 노이즈 바닥 대비 음성 판단 배수 (현재 기본값: 3.5, 튜닝 포인트)
- 직관: 주변 잡음보다 3.5배 이상 큰 신호만 음성 후보로 인정

**내 코드에서는**:
파일: `vad.py`
줄번호: `36-44`
```python
def calibrate(self, rms_list: list):
    """수집된 RMS 목록으로 노이즈 바닥 및 임계값 설정"""
    if rms_list:
        self.noise_rms = float(np.percentile(rms_list, 80))  # 상위 20% 제거
        self.threshold = self.noise_rms * VAD_RMS_MULT
    # 상태 리셋
    self._onset_cnt    = 0
    self._hangover_cnt = 0
    self._is_voiced    = False
```
파일: `ui_window.py`
줄번호: `690-692`
```python
if elapsed >= CALIB_SECS and self.calib_rms:
    self.vad.calibrate(self.calib_rms)
    self.calib_done = True
```
설명: 시작 2초 동안(CALIB_SECS=2.0) RMS를 수집한 후 `calibrate()`를 호출합니다. 80번째 백분위수를 쓰는 이유는 캘리브레이션 도중 우연히 소리를 낸 경우 극단값이 섞일 수 있기 때문입니다. (CALIB_SECS, VAD_RMS_MULT 모두 튜닝 포인트)

**발표자 노트**:
- "percentile 80" vs "평균": 평균은 큰 소리에 휩쓸림 → 안정적인 잡음 바닥 추정에는 백분위수가 적합
- 캘리브레이션이 끝난 후에도 침묵 구간마다 노이즈 바닥이 천천히 업데이트됨 → 슬라이드 6에서 설명

---

## 슬라이드 3: 조건 1 — RMS 에너지 기반 필터

**핵심 개념**:
- RMS(Root Mean Square): 신호의 "평균 에너지(진폭)"를 나타내는 값
- 직관: 아무 소리도 없으면 파형이 0 근처 → RMS 낮음 / 말을 하면 파형이 크게 진동 → RMS 높음
- VAD 조건 1: `rms < threshold`이면 즉시 무음으로 판정 — 가장 빠른 1차 필터
- 임계값은 캘리브레이션으로 설정 (개인·환경별 자동 조정)

**수식**:
$$
RMS = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}
$$
$$
\text{조건 1 통과} \Leftrightarrow RMS \ge \text{threshold}
$$
기호 설명:
- $x[n]$: n번째 샘플값 (float32, 범위 −1~+1)
- $N$: 청크 샘플 수 (300ms × 44100 = 13230)
- $\text{threshold} = \text{noise\_rms} \times 3.5$ (현재 기본값, 튜닝 포인트)
- RMS가 낮으면 나머지 두 조건을 검사하지 않고 즉시 반환 → 연산 절약

**다이어그램**:
[설명: 두 개의 파형 비교. 왼쪽 "침묵 (RMS 낮음)": 거의 수평선에 가까운 작은 진폭의 파형, RMS 값 예: 0.002. 오른쪽 "발화 (RMS 높음)": 진폭이 크게 출렁이는 파형, RMS 값 예: 0.08. 두 파형 아래에 수평 점선 "threshold = 0.007 (noise_rms × 3.5)". 왼쪽 파형은 점선 아래 → "VAD=OFF", 오른쪽 파형은 점선 위 → "조건 1 통과".]

**내 코드에서는**:
파일: `vad.py`
줄번호: `64-69`
```python
rms = float(np.sqrt(np.mean(chunk ** 2)))

# ── 1) RMS 에너지 조건 ──
if rms < self.threshold:
    self._adapt(rms)
    return self._update_state(False), rms
```
설명: `np.mean(chunk ** 2)`가 평균 제곱값, `sqrt`가 RMS입니다. 조건 불통과 시 `_adapt(rms)`로 노이즈 바닥을 업데이트하고 즉시 반환합니다. 조건 통과 시에는 다음 두 조건으로 넘어갑니다.

**발표자 노트**:
- RMS만으로는 충분하지 않음: 책상 두드리는 소리, 환경 잡음도 RMS가 높을 수 있음
- 조건 2(자기상관)와 조건 3(ZCR)이 "이게 진짜 음성인가?"를 추가로 검증

---

## 슬라이드 4: 조건 2 — 자기상관 주기성 검사

**핵심 개념**:
- 자기상관(autocorrelation): 신호가 자기 자신과 얼마나 닮았는지 측정
- 유성음(voiced sound): 성대 진동 → 주기적 파형 → 특정 시간 지연(lag)에서 자기상관값이 높음
- 무성음·잡음: 비주기적 → 어느 lag에서도 자기상관값이 낮음
- VAD 조건 2: 음성 F0 범위에 해당하는 lag에서 자기상관 비율이 임계값 이상이어야 통과

**수식**:
$$
r[k] = \sum_{n=0}^{N-1-k} x[n] \cdot x[n+k]
$$
$$
\text{ratio} = \frac{\max_{k \in [k_\min, k_\max]} r[k]}{r[0]}
$$
$$
\text{조건 2 통과} \Leftrightarrow \text{ratio} \ge \theta_{autocorr}
$$
기호 설명:
- $r[k]$: lag $k$에서의 자기상관값
- $r[0] = \sum x[n]^2$: 신호 에너지 (정규화 기준)
- $k_\min = f_s / f_{hi}$, $k_\max = f_s / f_{lo}$: F0 범위에 대응하는 lag 범위
- $\theta_{autocorr}$: 자기상관 임계값 (현재 기본값: 0.15, 튜닝 포인트)
- lag 범위: $f_{lo}=50Hz$, $f_{hi}=500Hz$ → lag 88~882샘플 @44100Hz

**다이어그램**:
[설명: 두 열로 구성된 자기상관 그래프. 왼쪽 열 "유성음 /아/": 위에 주기적 파형, 아래에 자기상관 그래프(x축=lag, y축=r[k]/r[0]). lag=0에서 1.0, lag=약 350샘플(≈8ms, F0=125Hz)에서 뚜렷한 봉우리 0.6. "ratio=0.6 > 0.15 → 통과" 라벨. 오른쪽 열 "잡음": 위에 불규칙한 파형, 아래 자기상관 그래프. lag=0에서 1.0이지만 그 이후는 거의 평탄하게 0에 가까움. "ratio=0.05 < 0.15 → 탈락" 라벨. x축에 k_min, k_max 수직 점선으로 탐색 범위 표시.]

**내 코드에서는**:
파일: `vad.py`
줄번호: `72-89`
```python
# ── 2) 자기상관 주기성 (유성음 확인) ──
lag_min = max(1, int(sr / pitch_hi))   # 500Hz에 해당하는 lag
lag_max = int(sr / pitch_lo)           # 50Hz에 해당하는 lag

# FFT 기반 자기상관: O(n log n), 직접 계산 O(n²)보다 ~10x 빠름
n = len(chunk)
fft_size = 1 << (2 * n - 1).bit_length()   # 다음 2의 거듭제곱
X = np.fft.rfft(chunk, n=fft_size)
r = np.fft.irfft(X * np.conj(X))[:n]
r0 = r[0]
if r0 <= 1e-12:
    return False, rms

ratio = float(np.max(r[lag_min:lag_max])) / r0
if ratio < AUTOCORR_THRESH:
    self._adapt(rms)
    return self._update_state(False), rms
```
설명: Part 1 슬라이드 11의 FFT 기반 자기상관과 동일한 방식입니다. `pitch_lo=50Hz`, `pitch_hi=500Hz`는 `ui_window.py` 595번 줄에서 호출 시 전달 (성별 무관 고정값). AUTOCORR_THRESH=0.15는 0.25에서 완화된 값 — 초저음 목소리도 포착하기 위함입니다. (현재 기본값, 튜닝 포인트)

**발표자 노트**:
- "자기상관 비율이 0이면 완전 랜덤 잡음, 1이면 순수 사인파" — 실제 음성은 0.3~0.8 사이
- lag_min/lag_max가 F0 범위와 역수 관계인 이유: 주기 T0 = 1/F0, lag = T0 × 샘플레이트

---

## 슬라이드 5: 조건 3 — ZCR 고주파 잡음 구별

**핵심 개념**:
- ZCR(Zero Crossing Rate, 영점 교차율): 신호가 초당 몇 번이나 0을 교차하는지
- 유성음: 낮은 주파수 주기 신호 → 영점 교차 횟수 적음 → ZCR 낮음
- 고주파 잡음(바람, 마찰음 /s/, /f/): 빠르게 진동 → ZCR 높음
- VAD 조건 3: `ZCR > ZCR_THRESH`이면 고주파 잡음으로 판정 → 무음 처리

**수식**:
$$
ZCR = \frac{1}{N} \sum_{n=1}^{N-1} \frac{|\text{sign}(x[n]) - \text{sign}(x[n-1])|}{2}
$$
$$
\text{조건 3 통과} \Leftrightarrow ZCR \le \theta_{ZCR}
$$
기호 설명:
- $\text{sign}(x)$: x > 0이면 +1, x < 0이면 −1
- $|\text{sign}(x[n]) - \text{sign}(x[n-1])| / 2$: 부호가 바뀌면 1, 안 바뀌면 0
- $\theta_{ZCR}$: ZCR 임계값 (현재 기본값: 0.35, 튜닝 포인트 — 0.20에서 완화)
- 직관: 유성음은 낮은 주파수 → 교차 적음 / 고주파 잡음은 교차 많음

**다이어그램**:
[설명: 세 행의 파형 비교. 각 행 오른쪽에 ZCR 값과 판정 표시. 첫 번째 행 "유성음 /아/ (F0≈125Hz)": 8ms 주기로 완만하게 오르내리는 파형. ZCR≈0.05 → "ZCR 낮음 → 조건 3 통과". 두 번째 행 "무성 마찰음 /s/": 고주파로 빠르게 진동하는 파형. ZCR≈0.42 → "ZCR 높음 → 탈락". 세 번째 행 "바람 소리·환경잡음": 불규칙하고 빠른 파형. ZCR≈0.38 → "ZCR 높음 → 탈락". 세 파형 아래에 수평 점선 "ZCR_THRESH=0.35" 표시.]

**Python 구현** (교과서 스타일 — 독립 함수 버전):
```python
def zero_crossing_rate(frame: np.ndarray) -> float:
    """프레임 내 영점 교차율 계산 (일반적인 교과서 구현)"""
    signs = np.sign(frame)                          # 부호 배열 (+1 / -1)
    crossings = np.sum(np.abs(np.diff(signs))) / 2  # 부호 변화 횟수
    return crossings / len(frame)                   # 프레임 길이로 정규화
```

**내 코드에서는**:
파일: `vad.py`
줄번호: `91-95`
```python
# ── 3) ZCR (유성음은 영점 교차율이 낮음) ──
zcr = float(np.sum(np.abs(np.diff(np.sign(chunk)))) / 2) / len(chunk)
if zcr > ZCR_THRESH:
    self._adapt(rms)
    return self._update_state(False), rms
```
파일: `config.py`
줄번호: `36`
```python
ZCR_THRESH = 0.35   # 영점 교차율 (유성음 = 낮음, 완화: 0.20→0.35)
```
설명:
- **독립 함수 없음**: vad.py에는 `zero_crossing_rate()` 독립 함수가 존재하지 않습니다. ZCR 계산이 `check()` 메서드 안에 한 줄로 인라인 처리됩니다 — 함수 호출 오버헤드 없이 흐름 제어(if 판단)까지 한 블록에서 처리합니다.
- **수식 대응**: `np.diff(np.sign(chunk))`는 부호가 바뀌는 지점에서 ±2, 안 바뀌는 곳에서 0을 반환합니다. `np.abs()` 후 합산하면 교차 횟수 × 2가 되어, `/2 / len(chunk)`로 정규화하면 수식의 ZCR과 동일합니다. 교과서 버전과 수학적으로 완전히 같습니다.
- **임계값 완화**: ZCR_THRESH를 0.20 → 0.35로 올린 이유는 거친 목소리나 노인 목소리처럼 유성음이어도 ZCR이 자연스럽게 높은 화자를 잡아내기 위함입니다. (현재 기본값, 튜닝 포인트)

**발표자 노트**:
- ZCR 하나만으로는 문제 있음: 조용한 환경에서 무성음(/s/, /f/)도 음성인데 탈락
- 무성음은 현재 VAD에서 탈락 — VoiceTypo는 유성 모음 인식이 목적이므로 허용 가능
- ZCR_THRESH 낮추면 보수적 (무성음도 걸러냄), 높이면 관대 (잡음도 통과 위험)

---

## 슬라이드 6: 3조건 AND 결합 + 적응형 노이즈 바닥 갱신

**핵심 개념**:
- AND 결합: 세 조건 모두 통과해야만 음성으로 판정 — 하나라도 실패하면 무음
- "보수적 판정"의 의미: 잘못된 무음 판정(false negative)이 잘못된 음성 판정(false positive)보다 낫다 — 포먼트에 잡음값이 섞이는 것을 방지
- 적응형 노이즈 바닥 갱신: 침묵 프레임마다 노이즈 추정값을 지수 이동 평균(EMA)으로 천천히 갱신 → 마이크 주변 환경이 바뀌어도 자동 적응

**수식**:
$$
\text{is\_voice} = \mathbf{1}[RMS \ge \theta_\text{RMS}] \;\wedge\; \mathbf{1}[\text{ratio} \ge \theta_\text{autocorr}] \;\wedge\; \mathbf{1}[ZCR \le \theta_\text{ZCR}]
$$

$$
\text{(침묵 프레임마다)}:\quad \text{noise\_rms} \leftarrow (1 - \alpha)\cdot\text{noise\_rms} + \alpha \cdot RMS
$$
기호 설명:
- $\wedge$: 논리 AND
- $\alpha$: 적응 속도 (현재 기본값: 0.008, 튜닝 포인트 — 작을수록 느린 적응)
- $\alpha=0.008$ 의미: 매 프레임 0.8%씩 갱신 → 약 125프레임(≈4초)에 걸쳐 절반 수렴
- 캘리브레이션 후에도 환경 변화(에어컨 켜짐 등)에 자동 추적

**내 코드에서는**:
파일: `vad.py`
줄번호: `47-50`
```python
def _adapt(self, rms: float):
    """침묵 프레임에서 노이즈 바닥을 천천히 추적"""
    self.noise_rms = (1 - ADAPT_RATE) * self.noise_rms + ADAPT_RATE * rms
    self.threshold = self.noise_rms * VAD_RMS_MULT
```
파일: `config.py`
줄번호: `30`
```python
ADAPT_RATE = 0.008   # 침묵 중 노이즈 바닥 적응 속도 (작을수록 느림)
```
설명: `_adapt()`는 RMS 조건(슬라이드 3), 자기상관 조건(슬라이드 4), ZCR 조건(슬라이드 5) 중 하나라도 실패한 침묵 프레임마다 호출됩니다. 이렇게 하면 오래 침묵하면 threshold가 내려가서 작은 소리도 감지하고, 시끄러운 환경에서는 threshold가 올라가서 잡음을 걸러냅니다. (ADAPT_RATE, VAD_RMS_MULT 모두 튜닝 포인트)

**발표자 노트**:
- EMA(Exponential Moving Average): $\alpha=0$ → 전혀 적응 안 함, $\alpha=1$ → 즉시 적응
- 0.008이 작아 보이지만, 30fps × 4초 = 120프레임이면 $1 - 0.992^{120} \approx 0.62$ — 절반 이상 수렴
- 적응이 너무 빠르면(α 크면) 음성 중에도 노이즈 바닥이 올라가서 음성을 잘라낼 수 있음

---

## 슬라이드 7: Onset/Hangover 상태 머신 — 안정화 (자율 추가)

**핵심 개념**:
- 문제: 3조건 AND만으로는 프레임 단위로 ON/OFF가 빠르게 교차하는 "지터(jitter)" 발생
- 예: 모음 발화 중 일시적으로 조건 하나가 실패 → 포먼트가 갑자기 끊김
- 해결: Onset 확인(연속 N프레임 통과 후 ON) + Hangover(OFF 후 M프레임 유지) 상태 머신
- 결과: 음성 시작은 신중하게, 음성 종료는 여운을 두고 처리 → 안정적인 출력

**수식**:
$$
\text{onset}: \quad \text{is\_voiced} \leftarrow \text{True} \quad \text{if} \; \text{onset\_cnt} \ge N_\text{onset}
$$
$$
\text{hangover}: \quad \text{is\_voiced} \leftarrow \text{False} \quad \text{after} \; N_\text{hangover} \text{ silent frames}
$$
기호 설명:
- $N_\text{onset}$: 음성 시작 확인에 필요한 연속 유성 프레임 수 (현재 기본값: 2, 튜닝 포인트)
- $N_\text{hangover}$: 음성 종료 후 유지할 프레임 수 (현재 기본값: 3 ≈ 90ms, 튜닝 포인트)
- onset: 짧은 충격음(기침, 문 닫힘)이 음성으로 오인되는 것 방지
- hangover: 모음 끝부분 에너지 감소로 인한 조기 종료 방지

**다이어그램**:
[설명: 상태 전이 다이어그램(state machine). 두 개의 원: "침묵 (is_voiced=False)" 과 "음성 (is_voiced=True)". 침묵→음성 화살표: "onset_cnt ≥ 2 (연속 2프레임 통과)". 음성→침묵 화살표: "hangover_cnt = 0 (3프레임 유지 후 소진)". 자기 루프: 음성 상태에서 raw_voice=False일 때 "hangover_cnt-- (아직 음성 유지)" 루프. 아래에 타임라인 예시: 실제 raw_voice 신호(들쭉날쭉)와 상태 머신 출력(부드러운 ON/OFF 구간)을 비교해서 상태 머신이 안정화함을 보여줌.]

**내 코드에서는**:
파일: `vad.py`
줄번호: `100-119`
```python
def _update_state(self, raw_voice: bool) -> bool:
    """
    순간 판단(raw_voice)을 onset/hangover 로직으로 안정화.
    - 음성 시작: ONSET_FRAMES 연속 voiced → 음성 ON
    - 음성 종료: raw_voice=False 후 HANGOVER_FRAMES 유지 → 음성 OFF
    """
    if raw_voice:
        self._onset_cnt += 1
        self._hangover_cnt = self.HANGOVER_FRAMES  # 행오버 리셋
        if self._onset_cnt >= self.ONSET_FRAMES:
            self._is_voiced = True
    else:
        self._onset_cnt = 0
        if self._hangover_cnt > 0:
            self._hangover_cnt -= 1
            # 행오버 기간 중에는 이전 상태 유지
        else:
            self._is_voiced = False
    return self._is_voiced
```
파일: `vad.py`
줄번호: `25-26`
```python
ONSET_FRAMES    = 2   # 음성 시작 확인에 필요한 연속 voiced 프레임 수
HANGOVER_FRAMES = 3   # Fix3: 음성 종료 후 유지 프레임 (5→3, ~90ms)
```
설명: 이 상태 머신 없이는 VAD 출력이 매 프레임 ON/OFF로 진동할 수 있습니다. ONSET_FRAMES=2는 2연속 통과(≈60ms @30fps), HANGOVER_FRAMES=3은 종료 후 3프레임(≈90ms) 유지. HANGOVER가 5→3으로 줄어든 이유: 너무 길면 무음 구간까지 포먼트를 출력해 F1/F2가 망가짐. (두 값 모두 튜닝 포인트)

**발표자 노트**:
- "Onset+Hangover" 패턴은 상업 VAD(WebRTC VAD 등)에서도 표준적으로 사용하는 방식
- Hangover를 길게 할수록 모음 끝을 잘 잡지만 후속 잡음도 포함될 위험 증가

---

## 슬라이드 8: Part 3 요약 + 전체 파이프라인 조감도

**핵심 개념**:
- Part 3에서 배운 VAD 파이프라인 5단계: 캘리브레이션 → RMS → 자기상관 → ZCR → Onset/Hangover
- VAD는 Part 1 포먼트 추출과 Part 2 pyworld F0 추출의 공통 입구
- VoiceTypo의 전체 실시간 파이프라인 = Part 1 + Part 2 + Part 3의 합

**다이어그램**:
[설명: 전체 실시간 파이프라인 플로우차트. 배경은 어두운 색. 세 개의 색상 구역으로 나눔.

**구역 A (초록): 오디오 수집 + 캘리브레이션**
"마이크 입력 (44100Hz, float32, 10ms 블록)"
→ "AudioStream deque 링버퍼 (300ms × 4 크기)"
→ "300ms 청크 추출"
→ "DC 제거: chunk -= mean(chunk)"
→ 분기: [calib_done=False → "캘리브레이션 중 (2초): RMS 수집 → AdaptiveVAD.calibrate()"] / [calib_done=True → 아래로 계속]

**구역 B (파랑): VAD 게이트 (Part 3)**
→ "AdaptiveVAD.check() — 3조건 AND"
  ├ 조건 1: RMS ≥ threshold (캘리브레이션 기준)
  ├ 조건 2: 자기상관 ratio ≥ 0.15 (F0 lag 범위 내)
  └ 조건 3: ZCR ≤ 0.35
→ "_update_state(): Onset(2프레임) + Hangover(3프레임)"
→ 분기: [is_voice=False → "무음 처리: None 반환 + _adapt()로 노이즈 바닥 갱신"] / [is_voice=True → 아래로 계속]

**구역 C (보라): F0 추출 게이트 (Part 2)**
→ "FormantEngine.extract() 진입"
→ "pyworld DIO (50~500Hz) + StoneMask → f0_arr"
→ "voiced_frac 계산: 유성음 프레임 비율"
→ 분기: [is_voiced=False → "무음 처리: None 반환"] / [is_voiced=True → 아래로 계속]
→ "pw.d4c() → aperiodicity → HNR"
→ "F0 평균값 확정 → 성별 자동 판단 (165Hz 임계)"

**구역 D (주황): 포먼트 추출 (Part 1)**
→ "DC제거 + 프리엠퍼시스 (α=0.97)"
→ 3개 병렬 방법:
  ├ "Method 1: Praat Burg LPC (Multi-ceiling: 3500/4800/5200Hz)"
  ├ "Method 2: pyworld CheapTrick 스펙트럼 피크 (f0_arr 재활용)"
  └ "Method 3: scipy LPC (Levinson-Durbin p=16 → AR 근 → arctan2)"
→ "앙상블: 2개 이상 동의 ±80/120/160Hz → 중앙값"
→ "HNR < 5dB이면 BW × 2 (Kalman 신뢰도 낮춤)"
→ "모음 전환 감지 (F1 ±120Hz, F2 ±200Hz) → Kalman 리셋"
→ "Kalman 필터 스무딩 (F1/F2/F3 독립)"

**출력 박스 (흰색)**:
"최종 출력: F0(Hz), F1/F2/F3(Hz), HNR(dB), confidence(0~1), agreement(0~1), jitter(%)"
→ "UI 갱신 (30fps): 시계열 그래프 + F1/F2 모음 공간 + 모음 판정 표시"

각 구역 오른쪽에 파일명 레이블:
구역 A: audio_stream.py / ui_window.py:565-588
구역 B: vad.py (AdaptiveVAD)
구역 C: formant_engine.py:130-169
구역 D: formant_engine.py:252-310 / formant_ensemble.py]

**발표자 노트**:
- 이 조감도가 Part 1~3 전체의 핵심 요약
- VAD(Part 3) → pyworld 유성음 판단(Part 2) → 포먼트 앙상블(Part 1)의 순서로 게이트가 겹쳐 작동
- 두 개의 게이트(VAD + pyworld)는 독립적 — VAD 통과해도 pyworld가 무성음으로 판정하면 포먼트 추출 안 함
- 다음 단계로 배울 내용: wav2vec2 모음 분류기(vowel_classifier.py), Kalman 필터 심화, 글자 변형 효과 연동

---

## 슬라이드 8 부록: Part 1~3 핵심 파라미터 한눈에 보기

| 파라미터 | 파일 | 줄 | 현재값 | 역할 | 튜닝 방향 |
|---|---|---|---|---|---|
| `PREEMPH_ALPHA` | config.py | 55 | 0.97 | 고주파 증폭 계수 | 낮추면 약한 증폭 |
| `FORMANT_CEILINGS` | config.py | 61 | [3500, 4800, 5200] | Multi-ceiling 탐색 범위 | 화자에 맞게 조정 |
| `AGREE_TOL` | formant_ensemble.py | 28 | {1:80, 2:120, 3:160} Hz | 앙상블 동의 허용 오차 | 넓히면 관대한 동의 |
| `PYWORLD_VOICED_FRAC_MIN` | config.py | 49 | 0.25 | 유성음 판단 최소 비율 | 높이면 엄격한 게이트 |
| `GENDER_THRESH_HZ` | config.py | 24 | 165 Hz | 성별 자동 판단 F0 임계 | 화자에 맞게 조정 |
| `VAD_RMS_MULT` | config.py | 29 | 3.5 | 노이즈 대비 음성 배수 | 높이면 엄격, 낮추면 민감 |
| `AUTOCORR_THRESH` | config.py | 35 | 0.15 | 자기상관 주기성 임계 | 높이면 유성음만 통과 |
| `ZCR_THRESH` | config.py | 36 | 0.35 | 영점 교차율 상한 | 낮추면 고주파 잡음 차단 강화 |
| `ADAPT_RATE` | config.py | 30 | 0.008 | 노이즈 바닥 적응 속도 | 높이면 빠른 환경 적응 |
| `HANGOVER_FRAMES` | vad.py | 26 | 3 | VAD 종료 후 유지 프레임 | 높이면 부드러운 종료 |

---

*Part 3 완료 — 슬라이드 8장 (+ 부록 파라미터 표)*
*Part 1~3 전체 교재 완성*
