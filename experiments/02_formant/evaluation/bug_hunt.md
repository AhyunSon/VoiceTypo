# VoiceTypo Bug Hunt 보고서


## 0. 분류기 초기화 상태 검증

```
wav2vec2 로딩 중...
  done    = True
  elapsed = 9.0s
  err     = None

wv.is_ready  = True     # property
wv._ready    = True     # private 속성 직접 확인
wv._error    = None

기본 prototype 빌드 대기...
wv.has_prototypes      = True
len(wv._default_proto) = 7
keys: ['아', '어', '에', '오', '우', '으', '이']

프로토타입 임베딩 검사 (정상이면 norm≈1.0):
  아: shape=(768,)  dtype=float32  norm=1.0000
  에: shape=(768,)  dtype=float32  norm=1.0000
  이: shape=(768,)  dtype=float32  norm=1.0000
  오: shape=(768,)  dtype=float32  norm=1.0000
  우: shape=(768,)  dtype=float32  norm=1.0000
  으: shape=(768,)  dtype=float32  norm=1.0000
  어: shape=(768,)  dtype=float32  norm=1.0000

_normalizer.ready = False
```


## 1. 단일 파일 단계별 추적: 아_01.wav


### 1.1 WAV 메타데이터

```
파일: 아_01.wav
샘플레이트: 44100 Hz   (시스템 SAMPLE_RATE = 44100 Hz)
길이: 66150 samples = 1.500s
dtype: float32
min/max: -0.1666259765625 / 0.17041015625
평균: 0.0000
```


### 1.2 FormantEngine.extract()

```
중앙 윈도우 추출: [26460 : 39690] (13230 samples = 300ms)
chunk RMS: 0.0248

  f0           = 236.078
  hnr          = 17.657
  is_voiced    = True
  agreement    = 0.100
  confidence   = 0.569
  raw_f1       = 947.755
  raw_f2       = 2099.812
  raw_f3       = 3355.646
  f1           = 947.755
  f2           = 2099.812
  f3           = 3355.646
  jitter       = 0.887
```


### 1.3 wav2vec2 K-NN 단계

```
_prep_audio: 4800 samples @ 16kHz (= 0.300s)
feat: shape=(768,)  norm=1.0000

코사인 유사도 vs default_proto:
  오: +0.9305  ←best
  으: +0.9038
  아: +0.8977
  우: +0.8714
  어: +0.8353
  에: +0.8340
  이: +0.3615

best=오  top sim=0.9305  margin=0.0267  conf=0.134
_knn_classify 임계 (top sim ≥ 0.45): PASS

K-NN 반환값: (오, 0.134)
```


### 1.4 formant_vowel_probs (Bark Mahalanobis)

```
입력: f1=947.8  f2=2099.8  gender=female
_normalizer.ready = False  (정규화 적용 여부)

  아: 0.7025  ←best
  에: 0.2926
  어: 0.0049
  으: 0.0000
  오: 0.0000
  이: 0.0000
  우: 0.0000
```


### 1.5 classify() 분기 결정

```
knn_vowel = 오  knn_conf = 0.134
fmt_best  = 아  fmt_conf  = 0.703

① knn_vowel == fmt_best AND knn_conf >= 0.10  → False
② fmt_conf > 0.35                            → True
③ knn_conf < 0.08                            → False

→ 경로 ②: 포먼트 단독 신뢰 → _formant_only() 위임
```


### 1.6 실제 호출 결과 (sanity check)

```
wv.classify(...) = ('아', 0.7025158840961682)

wv._formant_only(...) = ('아', 0.7025158840961682)
```


## 2. raw F1/F2 분포 vs 학계 평균

학계 평균: 하영우·오재혁 (2017), 여성 아나운서 8명 평균

⚠️ = |측정 − 학계| ≥ 100 Hz

| 모음 | n | F1 측정±SD | F1 학계 | F1 Δ | F2 측정±SD | F2 학계 | F2 Δ |
|---|---:|---|---:|---:|---|---:|---:|
| 아 | 4 | 789 ± 238 | 996 | -207 ⚠️ | 1411 ± 404 | 1503 | -92 |
| 에 | 5 | 449 ± 165 | 477 | -28 | 2581 ± 159 | 2514 | +67 |
| 이 | 5 | 496 ± 189 | 289 | +207 ⚠️ | 2562 ± 394 | 2716 | -154 ⚠️ |
| 오 | 2 | 515 ± 179 | 363 | +152 ⚠️ | 1170 ± 95 | 642 | +528 ⚠️ |
| 우 | 2 | 272 ± 5 | 332 | -60 | 1374 ± 74 | 832 | +542 ⚠️ |
| 으 | 5 | 281 ± 25 | 344 | -63 | 1920 ± 254 | 1711 | +209 ⚠️ |
| 어 | 4 | 558 ± 64 | 629 | -71 | 1971 ± 147 | 950 | +1021 ⚠️ |

## 3. 결론

포먼트 추출 성공률:
  raw_f1 추출됨: 27/35
  raw_f2 추출됨: 27/35

학계 대비 100Hz 초과 차이:
  F1: 3/7 모음
  F2: 5/7 모음
  최소 한 축 어긋남: 6/7 모음

**진단**

추출은 대부분 성공하지만 측정값이 학계와 광범위하게 어긋남 (6/7 모음에서 ≥100Hz 차이).

→ **포먼트 추출 정확도 자체가 낮음**. 분류기를 고쳐도 한계 명확.
추출은 대부분 성공하지만 측정값이 학계와 광범위하게 어긋남 (6/7 모음에서 ≥100Hz 차이).

→ **포먼트 추출 정확도 자체가 낮음**. 분류기를 고쳐도 한계 명확.