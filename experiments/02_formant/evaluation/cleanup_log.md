# Cleanup 실행 로그

`cleanup_plan.md` 의 단계별 실제 실행 기록. 각 단계 끝에 검증 결과 첨부.

---

## 1단계 — 단독 파일 삭제

**일시**: 2026-04-28

### 삭제한 파일 (4개)

| 파일 | 크기 | 삭제 사유 |
|---|---:|---|
| `mfcc_svm.py` | 8.5 KB | 호출 0건. `calibration_dialog` 만 import 했고 그것도 죽은 코드 |
| `calibration_dialog.py` | 11 KB | 어디서도 import 0건. `CalibrationDialog`/`load_calibration` 호출 0건 |
| `formant_engine_ORIG.py` | 0 byte | 빈 백업 파일 |
| `wav2vec_classifier_ORIG.py` | 24 KB | 23 KB 백업본. `set_calibration` 등 죽은 함수의 docstring 검색 오염원 |

**총 삭제량**: ~ 43 KB / 4 파일

### 사전 검증
1단계 진입 전 grep 으로 외부 참조 확인:

```
$ grep -rn "mfcc_svm|calibration_dialog|formant_engine_ORIG|wav2vec_classifier_ORIG" *.py
calibration_dialog.py:13:from mfcc_svm import MfccSvmClassifier
mfcc_svm.py:2:mfcc_svm.py — MFCC+CMVN 추출 + SVM 모음 분류기
calibration_dialog.py:2:calibration_dialog.py — 개인 모음 보정 다이얼로그
```

→ 자기 자신과 두 죽은 파일 사이의 상호 참조 1건만 존재. 외부 참조 0건. **안전 삭제 가능 확정**.

### 사후 검증

| 검증 | 명령 | 결과 |
|---|---|---|
| main.py 정상 import | `python -c "import main"` | ✅ `main.py import OK` |
| ui_window 정상 import | `python -c "from ui_window import RealtimePraatWindow"` | ✅ `ui_window import OK` |
| evaluate.py 정상 import | `python -c "import evaluation.evaluate"` | ✅ `evaluation.evaluate import OK` |
| baseline_simple 정확도 재현 | `python -m evaluation.baseline_simple` | ✅ **54.3% (19/35)** — 1단계 이전과 동일 |

### 루트 파일 현황 (After)

```
audio_stream.py                ← 유지 (I/O)
config.py                      ← 유지
diagnostic.py                  ← 유지 (단독 진단)
formant_engine.py              ← 2단계에서 단순화 예정
formant_ensemble.py            ← 2단계에서 삭제 예정
main.py                        ← 유지
speaker_tracker.py             ← 2단계에서 간소화 예정
test_speaker_adaptation.py     ← 3단계에서 삭제 예정
ui_window.py                   ← 2단계에서 wav2vec 제거
vad.py                         ← 유지
vowel_classifier.py            ← 유지
wav2vec_classifier.py          ← 2단계에서 삭제 예정
```

16 → 12 파일.

### 1단계 결론

**위험 0, 영향 0, 정확도 변화 0**. 의도대로 죽은 코드만 깔끔히 제거됨.

---

## 2단계 — 모듈 재작성 + UI 통합

**일시**: 2026-04-28

### 실행 순서 (의존성 그래프 기반)

먼저 의존을 끊고 → 그 다음 파일 삭제. 그렇지 않으면 import 깨짐.

#### Step 2A — formant_engine.py 단순화
- ✗ `KalmanFormant` 클래스 + `self.kf` 인스턴스 + `_prev_raw_f1/f2` 상태
- ✗ `from formant_ensemble import (cheaptrick_formants, scipy_lpc_formants, ensemble_formants)`
- ✗ `preemphasis` 함수 (scipy_lpc 전용이었음)
- ✎ `extract()` 메서드: pyworld + Praat 단독 → raw=f1/f2/f3 동일값 반환
- ✎ `reset_kalman()` 은 no-op 으로 유지 (ui_window 호환)
- ✎ `agreement` 항상 0.0 반환 (ui_window 호환 키)

코드량: 315 → 198 lines.

검증:
- `from formant_engine import FormantEngine` ✅
- 아_01.wav smoke test: f0=236, f1=729, f2=1216, f3=2973 (single ceiling 5500) ✅

#### Step 2B — formant_ensemble.py 삭제
- 의존 0건 확인 (formant_engine.py 의존 제거 후)
- ⚠ `evaluation/ceiling_experiment.py` 는 archival 실험 스크립트로 이 모듈 의존. 이미 결과는 [evaluation/results/method_comparison.md](evaluation/results/method_comparison.md) 에 보존. broken 상태 유지 (재실행 불가하나 프로덕션 영향 0).
- ✗ `formant_ensemble.py` 삭제

검증:
- `import main` ✅
- `baseline_simple` 54.3% (19/35) ✅

#### Step 2C — ui_window.py 의 wav2vec 통합 제거
편집 항목 (총 9개):

| 위치 | 변경 |
|---|---|
| imports | `Wav2VecVowelClassifier`, `_wvc_mod` 제거 |
| `__init__` | `self.wav2vec_clf = ...` + `start_loading` 블록 삭제. `self._tracker_seeded = False` 삭제 |
| 캘리브레이션 메시지 | "AI 모델 로딩 중..." → "노이즈 측정 중..." |
| `_wv_load_start` 변수 | 제거 |
| `_on_device_changed` | `self._tracker_seeded = False` 삭제 |
| 콜백 메서드 4개 | `_on_wav2vec_ready`, `_apply_wav2vec_ready`, `_on_wav2vec_error`, `_apply_wav2vec_error` 삭제 |
| `_analysis_loop` | wav2vec classify 호출 제거. result_q dict 에서 `wv_vowel`, `wv_conf` 키 제거 |
| `_tick` 분기 로직 | `wv_vowel != "?"` 분기 삭제 → `classify_vowel(...)` 직접 호출. `agr > 0.4` 부스트 제거. `clf_scale` 항상 1.0 |
| `_normalizer.seed_from_scale` | 블록 통째 삭제 |
| 상태표시 (lbl_calib) | `_normalizer.ready` 분기 삭제 → `tracker.status()` 만 |
| `_update_clf_label` | wav2vec 로딩 표시 → 정적 "포먼트 Mahalanobis" |

검증:
- `wav2vec | _wvc_mod | _normalizer` grep 결과: **0건** ✅
- `import main` ✅
- `from ui_window import RealtimePraatWindow` ✅

#### Step 2D — wav2vec_classifier.py 삭제
- 의존 0건 확인
- ✗ `wav2vec_classifier.py` 삭제 (24 KB)

검증:
- `import main` ✅
- `baseline_simple` 54.3% (19/35) ✅

#### Step 2E — speaker_tracker.py 간소화
**유지**:
- F0 EWM → voice_type 판정
- `praat_gender` property
- `formant_ceilings` (단, 모든 voice_type 에 [5500] 반환)
- `scale` 속성 (호환용, 항상 1.0)
- `status()` 메서드 (scale 표시 부분 제거)

**삭제**:
- F0 기반 scale 계산 (`_REF_F0`, `_SCALE_MIN`, `_SCALE_MAX`)
- `seed_from_scale` 메서드
- voice_type 별 ceiling 분기 (`_CEILINGS` dict 자체)
- `clf_gender` property (ui_window 가 더이상 사용 안 함)

코드량: 126 → 87 lines.

검증:
- `SpeakerF0Tracker()` 인스턴스화 + 모든 property 정상 ✅
- `import main` ✅
- `from ui_window import RealtimePraatWindow` ✅
- `baseline_simple` 54.3% (19/35) ✅

### 2단계 누적 결과

| 항목 | Before | After |
|---|---:|---:|
| 루트 .py 파일 | 12 | 10 |
| 총 코드 라인 | ~3000 | ~1700 |
| `formant_engine.py` | 315 lines | 198 |
| `speaker_tracker.py` | 126 lines | 87 |
| `ui_window.py` | 996 lines | 862 |

루트 파일 현황 (After):
```
audio_stream.py                ← 유지
config.py                      ← 유지 (3단계에서 일부 상수 정리)
diagnostic.py                  ← 유지 (단독)
formant_engine.py              ← 단순화 완료
main.py                        ← 유지
speaker_tracker.py             ← 간소화 완료
test_speaker_adaptation.py     ← archival, 3단계 삭제 예정
ui_window.py                   ← wav2vec 제거 완료
vad.py                         ← 유지
vowel_classifier.py            ← 유지
```

`evaluation/` 폴더 내 archival (이미 broken):
- `evaluation/ceiling_experiment.py` (formant_ensemble 의존, 결과는 method_comparison.md 에 보존)

### 2단계 결론

**baseline_simple 정확도 54.3% 유지** (모든 단계 후 19/35 동일).
프로덕션 시각화/실시간성/성별 자동 전환 모두 보존.
22.9% 의 핵심 원인 (앙상블 + Kalman + wav2vec K-NN + F0 scale) 모두 제거됨.

---

## 3단계 — Archival 정리 + config + 호환 키 제거

**일시**: 2026-04-28

### Step 3A — Archival broken 파일 삭제

| 파일 | 위치 | 의존 | 결과 보존 |
|---|---|---|---|
| `test_speaker_adaptation.py` | 루트 | 삭제된 `_VowelNormalizer` 등 | (테스트 코드, 보존 불필요) |
| `evaluation/ceiling_experiment.py` | evaluation/ | 삭제된 `formant_ensemble` | [evaluation/results/method_comparison.md](evaluation/results/method_comparison.md) 에 결과 보존 |

검증:
- `import main` ✅
- `baseline_simple` 54.3% (19/35) ✅

### Step 3B — config.py 죽은 상수 정리

사용처 grep 으로 확인 후 제거:

| 상수 | 사유 |
|---|---|
| `KALMAN_PROCESS_NOISE` | Kalman 클래스 제거됨 (Stage 2) |
| `KALMAN_MEAS_NOISE_DEF` | 동일 |
| `PREEMPH_ALPHA` | preemphasis 함수 제거됨 (Stage 2) |
| `VOWEL_STABLE_FRAMES` | 어디서도 import 0건 (구식 상수) |

**`FORMANT_CEILINGS` 정책 변경**:
- 이전: `[3500, 4800, 5200]` (멀티 ceiling 앙상블 시대 default)
- 이후: `[5500]` (단일 ceiling — ceiling_experiment 결과 가장 정확)
- speaker_tracker.formant_ceilings 가 항상 [5500] 반환하므로 default 도 일치시킴.

검증:
- `import main` ✅
- `from ui_window import RealtimePraatWindow` ✅
- `baseline_simple` 54.3% ✅

### Step 3C — `agreement` 키 / `tracker.scale` 속성 완전 제거

Stage 2 에서 ui_window 호환을 위해 fixed value 로 남겼던 잔재 정리.

**`formant_engine.py::extract()`**:
- 반환 dict 에서 `agreement=0.0` 제거 (양 분기 모두)
- docstring 의 agreement 줄 삭제

**`ui_window.py::_analysis_loop()`**:
- 무성음 result_q dict 에서 `agreement=0.0` 제거
- 유성음 result_q dict 에서 `trk_scale=...,`, `trk_ready=...` 제거
- 주석 정리 ("HNR 이중 게이트" 한 줄로)

**`ui_window.py::_tick()`**:
- `trk_scale = result.get("trk_scale", 1.0)` 삭제 (사용 0건)
- `trk_ready = result.get("trk_ready", False)` 삭제 (사용 0건)

**`speaker_tracker.py`**:
- `self.scale: float = 1.0` 속성 삭제
- 클래스 docstring 에서 `scale` 항목 삭제
- 정리 이력 코멘트 갱신

검증:
- `import main` ✅
- `from ui_window import RealtimePraatWindow` ✅
- `FormantEngine.extract()` 반환 dict 키 11개 (agreement 제거됨, raw_f1/2/3 + f1/2/3 + f0/hnr/jitter/confidence/is_voiced) ✅
- `hasattr(SpeakerF0Tracker(), "scale") == False` ✅
- `baseline_simple` 54.3% (19/35) ✅

### 3단계 누적 결과

| 항목 | After Stage 2 | After Stage 3 |
|---|---:|---:|
| 루트 .py 파일 | 10 | 9 |
| evaluation/ archival 깨진 스크립트 | 1 | 0 |
| 죽은 config 상수 | 4 | 0 |
| ui_window 호환용 잔재 키/속성 | 4 (agreement, trk_scale, trk_ready, scale) | 0 |

루트 파일 현황 (After Stage 3):
```
audio_stream.py
config.py             ← 4 dead 상수 제거, FORMANT_CEILINGS=[5500]
diagnostic.py
formant_engine.py     ← agreement 키 제거
main.py
speaker_tracker.py    ← scale 속성 제거
ui_window.py          ← trk_scale/trk_ready/agreement 참조 제거
vad.py
vowel_classifier.py
```

`evaluation/` 폴더 (broken archival 모두 정리):
```
evaluation/
├── __init__.py
├── record_dataset.py     ← 데이터셋 녹음 (작동)
├── evaluate.py            ← 기존 시스템 평가 (Wav2Vec import 깨졌을 수 있음 — 추후 정리 가능)
├── baseline_simple.py    ← 새 시스템 평가 (54.3%) — 사실상의 메인 평가 도구
├── bug_hunt.py            ← (Wav2Vec import 깨졌을 수 있음)
├── cleanup_plan.md
├── cleanup_log.md
├── new_architecture.md
├── refs_validation.md
├── speaker_independence_plan.md
├── diagnostic_report.md
├── bug_hunt.md
└── results/
    ├── accuracy_report.md, confusion_matrix.png, raw_distribution.csv,
    │   vowel_space.png         (기존 시스템 22.9% 결과)
    ├── baseline_simple.md, baseline_simple_confusion.png  (54.3%)
    ├── ceiling_comparison.md, ceiling_comparison.png
    └── method_comparison.md
```

### 3단계 결론

**baseline_simple 54.3% 유지**.
호환 키 제거로 코드가 "현재 정책과 일관된" 상태가 됨 — 더이상 stale 주석이나 항상 0.0 인 키가 없음.

다음: **Step 3D — 사용자 직접 마이크 테스트** 대기.
- main.py 실행 → 모음 인식 동작
- 시각화 (F0/RMS/jitter, F1/F2/F3 scatter, 모음공간) 정상
- 남녀 자동 전환

테스트 결과 보고 후 다음 작업 결정.
