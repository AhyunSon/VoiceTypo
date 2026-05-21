# 새 시스템 설계안

baseline_simple.py 의 단순한 구조를 메인 시스템으로 승격.

---

## 1. 설계 원칙

1. **단일 책임**: 각 모듈은 한 가지만 잘 함.
2. **앙상블 금지**: Praat Burg 단독.
3. **화자 정규화는 _REFS 선택까지만**: voice_type → male/female refs. F1/F2 절대값 변환 금지.
4. **시각화는 본래 기능 그대로**: F0, RMS, jitter 표시 유지.
5. **실시간 청크 슬라이딩 유지**: 외부 인터페이스는 현재 ui_window 와 동일하게.

---

## 2. 모듈 구조

```
realtime_formant/
├── main.py                     ⟵ 변경 없음
├── config.py                   ⟵ 일부 상수 정리 (FORMANT_CEILINGS → 단일 5500)
│
├── audio_stream.py             ⟵ 변경 없음 (pure I/O)
├── vad.py                      ⟵ 변경 없음 (적응형 VAD)
│
├── formant_engine.py           ⟵ 단순화 (Kalman, ensemble 제거)
├── vowel_classifier.py         ⟵ 변경 없음 (Bark Mahalanobis)
├── speaker_tracker.py          ⟵ 간소화 (scale 제거, voice_type 만)
│
├── ui_window.py                ⟵ wav2vec 관련 제거, EMA 유지
│
└── diagnostic.py               ⟵ 유지 (단독 진단 스크립트)

✗ formant_ensemble.py          (전체 삭제)
✗ wav2vec_classifier.py        (전체 삭제)
✗ mfcc_svm.py                  (전체 삭제)
✗ calibration_dialog.py        (전체 삭제)
✗ formant_engine_ORIG.py       (빈 파일 삭제)
✗ wav2vec_classifier_ORIG.py   (백업 삭제)
✗ test_speaker_adaptation.py   (의존 모듈 삭제 후 함께 제거)
```

---

## 3. 모듈별 책임

### 3.1 `formant_engine.py` (단순화)

**역할**: 청크 → (F0, F1, F2, F3, jitter, is_voiced)

**구현 골격**:
```
class FormantEngine:
    def extract(chunk, gender) -> dict:
        # 1) DC 제거
        # 2) pyworld DIO + StoneMask → f0, f0_arr
        # 3) is_voiced = (유성음 프레임 비율 ≥ 0.25)
        # 4) jitter = compute_jitter(f0_arr)
        # 5) Praat Burg, ceiling=5500, 중앙 시점에서 F1/F2/F3
        return {
            "f0": ..., "jitter": ...,
            "f1": ..., "f2": ..., "f3": ...,
            "is_voiced": ...,
        }
```

**삭제**:
- KalmanFormant 클래스
- 멀티 ceiling 루프
- bw_avg 가중 중앙값 (단일 ceiling이면 불필요)
- HNR 계산 (D4C aperiodicity)
- agreement, confidence 반환 필드 (앙상블 없으므로 무의미)
- raw_f1/raw_f2/raw_f3 ↔ f1/f2/f3 분리 (Kalman 없으므로 동일값)

**유지**:
- compute_jitter (시각화용)
- pyworld DIO (F0 추출, voice_type 판정 입력)

### 3.2 `vowel_classifier.py` (변경 없음)

`classify_vowel(f1, f2, gender, f3=None, scale=1.0)` 그대로 사용.

`scale` 인자는 유지하되, 새 시스템에서는 항상 `scale=1.0` 으로 호출.
(미래에 _REFS 가 학계 다중 출처로 보강될 때, 화자별 작은 보정으로만 사용 검토 가능.)

### 3.3 `speaker_tracker.py` (간소화)

**유지**:
- F0 EWM → voice_type 판정 (`female`/`male`/`child`)
- `praat_gender` property → "male" or "female" (_REFS 선택용)

**제거**:
- `scale` 계산 + `seed_from_scale`
- `formant_ceilings` (단일 5500 으로 통일)
- `_REF_F0` 기반 정규화 로직
- `_SCALE_MIN`/`_SCALE_MAX` clamp

**남는 인터페이스**:
```
class SpeakerF0Tracker:
    def update(f0): ...                  # EWM 갱신
    def reset(): ...
    @property voice_type → str           # "female" | "male" | "child"
    @property praat_gender → str         # "male" | "female"
    def status() → str                   # UI 표시용
```

### 3.4 `ui_window.py` 통합

**삭제할 부분**:
- `from wav2vec_classifier import ...` (line 45-46)
- `self.wav2vec_clf = ...` 초기화 + start_loading (line 90-94)
- `_on_wav2vec_ready/_error`, `_apply_wav2vec_*`, `_update_clf_label` 메서드
- `_tick()` 의 wv_vowel/wv_conf 분기 (line 808-819)
- `_wvc_mod._normalizer.seed_from_scale` (line 793)
- `get_normalizer_status` 표시 코드 (line 886-888)

**`_tick()` 의 모음 결정 로직 (변경)**:

```python
# 기존 (제거):
if wv_vowel != "?" and wv_conf > 0.15:
    vowel_raw = wv_vowel
else:
    vowel_raw, _ = classify_vowel(f1, f2, clf_gender, f3=raw_f3, scale=clf_scale)
    # agreement boost ...

# 새 버전:
if iv:
    vowel_raw, v_conf = classify_vowel(
        f1, f2, gender=tracker.praat_gender,
        f3=f3, scale=1.0,
    )
else:
    vowel_raw, v_conf = "?", 0.0
```

**유지**:
- 캘리브레이션 단계 (노이즈 측정)
- 시계열 그래프 (F0/RMS/F1·F2·F3/jitter)
- 모음 공간 (F1/F2 ellipse + 측정 점)
- EMA + 히스테리시스 (best-vowel commit/release/switch)
- voice_type 표시 (남성/여성/아동 색상)
- 화자 분석 status text (단, _normalizer 부분 제외)

**캘리브레이션 메시지 단순화**:
- "AI 모델 로딩 중..." 제거
- "노이즈 측정 중... → 보정 완료 noise=..." 만 표시

### 3.5 `config.py` (소폭 정리)

**제거**:
- `FORMANT_CEILINGS = [3500, 4800, 5200]` → 단일 상수 `FORMANT_CEILING = 5500`
- `MAX_BW`, `SAMPLE_POS` (멀티 샘플링 안 함)
- `KALMAN_PROCESS_NOISE`, `KALMAN_MEAS_NOISE_DEF`
- `HNR_MIN_DB`, `HNR_VOICE_MIN` (HNR 게이팅 안 함)
- `PYWORLD_VOICED_FRAC_MIN` (단순화 가능)
- `PARAMS["male"]["pre_emphasis"]/window_length/max_formants` 등 일부는 Praat 호출에서 직접 사용

**유지**:
- SAMPLE_RATE, BLOCK_SIZE, ANALYSIS_WIN_SEC, UPDATE_MS
- VAD_RMS_MULT, ADAPT_RATE
- VOWEL_REFS, VOWEL_REFS_MALE (시각화용 ellipse 좌표)

---

## 4. 외부 인터페이스 호환

`ui_window._analysis_loop` 가 호출하는 형태:
```python
res = engine.extract(chunk, gender)
```

이 인터페이스 자체는 변경 없음. 내부 구현만 단순화. 외부에서 `res["f1"]`, `res["f0"]`, `res["is_voiced"]`, `res["jitter"]` 모두 그대로 접근 가능.

따라서 `ui_window.py` 의 시각화 데이터 큐 (q_f0, q_f1, q_f2, q_f3, q_rms, q_jitter) 는 변경 없음.

---

## 5. 시각화 기능 보존

작품의 핵심 시각화 데이터는 다음 경로로 그대로 유지:

| 시각화 | 출처 | 상태 |
|---|---|---|
| 시계열 F0 | pyworld DIO 평균 | 유지 |
| 시계열 RMS | VAD 측정 | 유지 |
| 시계열 jitter | compute_jitter(f0_arr) | 유지 |
| F1/F2/F3 scatter | Praat 추출 | 유지 |
| 모음 공간 ellipse | VOWEL_REFS / VOWEL_REFS_MALE | 유지 |
| 모음 공간 궤적 | 슬라이딩 청크 누적 | 유지 |
| 추정 모음 표시 | classify_vowel + EMA | 유지 (단순화된 경로) |
| 떨림(jitter) 색 표시 | jitter 임계값 분기 | 유지 |
| 목소리 유형 표시 | speaker_tracker.voice_type | 유지 |
| 신뢰도 표시 | classify_vowel 의 conf 값 | 유지 |

**손실**: agreement (앙상블 합의 점수) 표시. 이건 사용자 가시 정보가 아니었음.

---

## 6. 새 시스템의 한계와 향후 방향

### 한계
- _REFS 가 단일 출처(Yoon 2015 추정) 기반 → 화자 다양성에 약함
- 5500Hz 단일 ceiling 은 평균적으로 좋지만 모음별 최적은 아님 (오/우는 [3500, 4800, 5200] 이 더 정확)
- 단순 Mahalanobis 는 모음 경계가 직선/타원에 갇힘

### 향후 가능 (우선순위 순)
1. **다양한 화자 데이터 수집** → 화자 독립성 검증 ([speaker_independence_plan.md](speaker_independence_plan.md))
2. **_REFS 학계 다중 출처 보강** ([refs_validation.md](refs_validation.md))
3. **EMA 모음 안정화 파라미터 튜닝** (현재 상수가 이전 22.9% 시스템 기준)
4. **선택적**: 모음별 가변 ceiling (오/우 만 다른 ceiling) — 단, 복잡도 증가 트레이드오프 신중

### 절대 하지 말 것 (재확인)
- **본인 데이터로 _REFS 갱신** → speaker overfitting
- 합성 KNN 부활
- F0 기반 scale 부활
- 앙상블 부활

---

## 7. 변경 후 baseline 재측정

각 단계 후 [evaluation/baseline_simple.py](evaluation/baseline_simple.py) 재실행하여 54.3% 동일 또는 상승 확인.

또한 다양한 화자 데이터셋 확보 후 동일 평가를 모두 적용해 화자 독립성 측정 (별도 plan 참조).
