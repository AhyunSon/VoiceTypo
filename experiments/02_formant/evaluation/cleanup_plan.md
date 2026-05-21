# 죽은/유해 코드 정리 계획

baseline_simple.py 가 22.9% → 54.3% 로 보여준 결과를 기반으로,
복잡성을 키우면서 정확도를 떨어뜨린 모듈들을 정리한다.

**원칙**: 이 문서는 계획만. 실제 코드 변경은 별도 지시 후 진행.

---

## 1. 분류표 (요약)

| 파일/요소 | 상태 | 행동 |
|---|---|---|
| [mfcc_svm.py](mfcc_svm.py) | 호출 0건 | **삭제** |
| [calibration_dialog.py](calibration_dialog.py) | 호출 0건 | **삭제** |
| [formant_engine_ORIG.py](formant_engine_ORIG.py) | 0 byte (빈 파일) | **삭제** |
| [wav2vec_classifier_ORIG.py](wav2vec_classifier_ORIG.py) | 23 KB 백업본, 호출 0건 | **삭제** |
| [test_speaker_adaptation.py](test_speaker_adaptation.py) | 단독 실행 테스트, 의존 일부 | **삭제 보류** (3단계) |
| [diagnostic.py](diagnostic.py) | 단독 실행 (project import 없음) | 유지 (해롭지 않음) |
| [wav2vec_classifier.py](wav2vec_classifier.py) | 22.9% 원인 | **삭제** (정밀 분리 필요) |
| [formant_ensemble.py](formant_ensemble.py) | 앙상블 우회 후 사용 0건 | **삭제** |
| `formant_engine.py` Kalman | 단일 청크에서 무의미 | 코드 제거 |
| `formant_engine.py` ensemble 호출 (cheaptrick + scipy_lpc) | 정확도 손해 확인 | 코드 제거 |
| `vowel_classifier.py` set_calibration 주석 | 미구현 함수 언급 | docstring 정리 |
| `wav2vec_classifier.py` `_VowelNormalizer` | 효과 불명확 | wav2vec 제거 시 함께 삭제 |
| `wav2vec_classifier.py` 합성 KNN | 변별력 약함, 22.9% 기여 | wav2vec 제거 시 함께 삭제 |

**삭제 후 줄어드는 코드량**: 약 **~ 90 KB / 6 파일 + 두 모듈의 약 절반**.

---

## 2. 파일별 상세

### 2.1 [mfcc_svm.py](mfcc_svm.py) — **삭제**

- **호출 그래프**: `calibration_dialog.py:13` 한 곳만 import. 그러나 `calibration_dialog` 자체가 어디서도 import되지 않음 → 도달 불가능 코드.
- **영향받는 코드**: 없음.
- **순서**: 1단계, 단독 삭제 가능.

### 2.2 [calibration_dialog.py](calibration_dialog.py) — **삭제**

- **호출 그래프**: import 0건. `CalibrationDialog`, `load_calibration` 함수 호출 0건.
- **영향받는 코드**: 없음.
- **순서**: 1단계, mfcc_svm.py 와 함께 삭제.

### 2.3 [formant_engine_ORIG.py](formant_engine_ORIG.py) — **삭제**

- 0 byte 빈 파일.
- **순서**: 1단계.

### 2.4 [wav2vec_classifier_ORIG.py](wav2vec_classifier_ORIG.py) — **삭제**

- 23 KB 백업본. `set_calibration`, `load_calibration` 등의 죽은 함수가 docstring 검색 결과를 오염시키는 출처.
- **호출 그래프**: import 0건.
- **순서**: 1단계.

### 2.5 [vowel_classifier.py](vowel_classifier.py) `set_calibration` 관련 주석 — **정리**

- 함수 자체는 존재하지 않음. 다만 [vowel_classifier.py:9](vowel_classifier.py#L9), [L57](vowel_classifier.py#L57) 의 docstring/주석에 `set_calibration()` 이 언급됨.
- **행동**: 주석에서 그 줄들 제거 (함수 호출은 없으므로 동작 영향 0).
- **순서**: 1단계.

### 2.6 [wav2vec_classifier.py](wav2vec_classifier.py) 전체 — **삭제**

이 모듈 전체가 22.9% baseline 의 주범.

- 합성 prototype K-NN 은 실제 음성과 cos 유사도 0.83+ 모든 모음에 대해 발생 → 변별력 없음
- `_VowelNormalizer` 의 anchor 누적 방식은 본인 1인 데이터 기준 (`_POP_I_F2 = 2787`) → speaker 종속
- `_lip_round_score` 같은 헛 휴리스틱 다수
- evaluate.py 에서 wav2vec 우회/포함 정확도 동일 → 가치 0

**제거 영향 범위**:

| 외부 의존 위치 | 처리 방법 |
|---|---|
| [ui_window.py:45-46](ui_window.py#L45-L46) `import Wav2VecVowelClassifier`, `_wvc_mod` | 전부 삭제 |
| [ui_window.py:90-94](ui_window.py#L90-L94) `self.wav2vec_clf = ...` 초기화 + start_loading | 삭제 |
| [ui_window.py:793](ui_window.py#L793) `_wvc_mod._normalizer.seed_from_scale(...)` | 삭제 |
| [ui_window.py:886-888](ui_window.py#L886-L888) `get_normalizer_status()` 표시 코드 | 삭제 |
| [ui_window.py:135-136](ui_window.py#L135-L136) "AI 모델 로딩 중..." 메시지 | 캘리브레이션 표시로 대체 |
| `_apply_wav2vec_ready/error`, `_on_wav2vec_ready/error`, `_update_clf_label` 메서드 | 삭제 |
| `_tick()` 의 wv_vowel 로직 | classify_vowel 단독 결과만 사용 |

- **순서**: 2단계 (UI 정리와 함께).

### 2.7 [formant_ensemble.py](formant_ensemble.py) 전체 — **삭제**

method_comparison.md 결과:

| 방법 | 평균 \|F2 Δ\| |
|---|---:|
| Praat 단독 | 131 |
| ensemble | 397 |

cheaptrick / scipy_lpc 단독은 더 나쁨 (935, 292). 앙상블이 정확한 신호를 잘못된 신호로 평균내고 있음.

- **호출 그래프**: `formant_engine.py:25-29` 에서 `cheaptrick_formants`, `scipy_lpc_formants`, `ensemble_formants`, `_weighted_median`, `preemphasis` import.
- **처리**: `formant_engine.py` 를 새 단순 버전으로 교체하면서 동시에 제거.
- **`_weighted_median`/`preemphasis` 보존?**: 단순 Praat single-ceiling 으로는 불필요. 둘 다 떠나 보냄.
- **순서**: 2단계 (formant_engine 재작성과 함께).

### 2.8 [formant_engine.py](formant_engine.py) 의 Kalman, ensemble 호출 — **단순화**

`formant_engine.py` 는 유지하되 다음을 제거:

| 제거 대상 | 이유 |
|---|---|
| `KalmanFormant` 클래스 + `self.kf` 인스턴스 | 단일 청크 평가에서 의미 0. 실시간에서 슬라이딩 청크 간 매끄러움 효과는 EMA(이미 ui_window 에 있음)가 충분히 대체. |
| `cheaptrick_formants`, `scipy_lpc_formants`, `ensemble_formants` 호출 | method_comparison 에서 손해 확인 |
| `_pyworld_voiced` 의 D4C HNR 계산 | F0 만 필요 (시각화용 jitter, voice_type 판정용). HNR 게이팅 우회. |
| `compute_jitter` | **유지** — 시각화에서 사용 |
| `VOWEL_CHANGE_THRESH_F1/F2` (Kalman 리셋 트리거) | Kalman 제거하면 자동 무용 |
| `preemphasis` 함수 | Praat 자체 pre_emphasis_from=50 이 충분. 수동 한 번 더는 불필요. |

**남는 골격**:
```
def extract(chunk, gender):
    1. pyworld DIO/StoneMask → f0, f0_arr, jitter
    2. Praat To Formant (burg) ceiling=5500 → f1, f2, f3 (중앙 시점)
    3. is_voiced 는 pyworld 유성음 비율로 판정
    return dict(f0, jitter, f1, f2, f3, is_voiced)
```

- **순서**: 2단계.

### 2.9 [speaker_tracker.py](speaker_tracker.py) — **간소화**

현재 기능:
1. F0 EWM → voice_type ("female"/"male"/"child")
2. F0 → scale (집단 평균 / 화자 F0) — **이것이 22.9% 의 부수 요인**
3. voice_type 별 ceiling 목록

**유지**: voice_type 판정 (`praat_gender` property) — male/female 구분만. _REFS 선택에 필요.
**제거**: `scale`, `seed_from_scale`, ceiling 목록 (단일 5500 으로 통일).

- **순서**: 2단계.

### 2.10 [test_speaker_adaptation.py](test_speaker_adaptation.py) — **삭제 (3단계)**

- `_VowelNormalizer` 와 SpeakerF0Tracker.scale 동작 테스트.
- 둘 다 제거 대상이므로 의미 상실.
- **순서**: 3단계 (의존 모듈 정리 후).

### 2.11 [diagnostic.py](diagnostic.py) — **유지**

- 마이크 + 분석 파이프라인 단독 진단 스크립트.
- 프로젝트 모듈 import 없음 (sys, numpy 만).
- 해롭지 않음.

---

## 3. 안전한 제거 순서

```
1단계 (단독 파일 삭제 — 위험 0):
  ✗ mfcc_svm.py
  ✗ calibration_dialog.py
  ✗ formant_engine_ORIG.py
  ✗ wav2vec_classifier_ORIG.py
  ✎ vowel_classifier.py 주석 정리

2단계 (모듈 재작성 + UI 통합):
  ✎ formant_engine.py 단순 버전으로 교체
     ↳ Kalman, ensemble 호출 제거
  ✎ speaker_tracker.py 간소화
     ↳ scale 제거
  ✗ formant_ensemble.py
  ✗ wav2vec_classifier.py
  ✎ ui_window.py
     ↳ wav2vec 관련 모두 제거
     ↳ classify_vowel 단독 사용

3단계 (테스트/관망 후):
  ✗ test_speaker_adaptation.py
```

각 단계 끝마다 [evaluation/baseline_simple.py](evaluation/baseline_simple.py) 재실행 → 정확도 동일/상승 확인 후 다음으로.

---

## 4. 주의사항

- **시각화 데이터(F0, RMS, jitter)는 유지**. 작품의 핵심.
- **EMA 모음 안정화 ([ui_window.py:823-855](ui_window.py#L823-L855)) 유지**. Kalman 의 매끄러움 역할을 EMA 가 이미 수행 중.
- **VAD ([vad.py](vad.py)) 유지**. 잡음 환경 보호.
- **캘리브레이션 단계 ([ui_window.py:697-710](ui_window.py#L697-L710)) 유지**. 환경 노이즈 측정용. 단, "AI 모델 로딩 중" 메시지는 제거.
- **작품 시나리오 (불특정 화자)** 를 위해 male/female 자동 전환 (`SpeakerF0Tracker.praat_gender`) 은 유지.

---

## 5. 제거 후 예상 코드량

| 항목 | Before | After |
|---|---:|---:|
| Python 파일 수 | 16 | 9 |
| 총 코드 라인 수 (대략) | ~3000 | ~1200 |
| 시작 시간 (wav2vec 모델 로딩) | 10 ~ 30 s | 즉시 |
| 메모리 (wav2vec 모델) | ~ 400 MB | ~ 0 |

작품 기능 손실: **0** (시각화·인식·실시간성 모두 유지).
