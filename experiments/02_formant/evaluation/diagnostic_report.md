# VoiceTypo 시스템 진단 보고서

**작성 시점**: 2026-04-27
**조사 범위**: `realtime_formant/` 루트 (코드 읽기 전용, 수정 없음)

---

## 1. 죽은 코드 확인

| 항목 | 결과 | 근거 |
|------|------|------|
| `mfcc_svm.py`가 `main.py`에서 import | **없음** | [main.py:12-14](main.py#L12-L14) — `sys`, `QApplication`, `RealtimePraatWindow` 만 import |
| `mfcc_svm.py`가 `ui_window.py`에서 import | **없음** | [ui_window.py:16-47](ui_window.py#L16-L47) — `mfcc_svm` 부재. import 목록은 `audio_stream`, `vad`, `formant_engine`, `vowel_classifier`, `wav2vec_classifier`, `speaker_tracker` 만 |
| `mfcc_svm` 모듈을 import하는 다른 파일 | **있음 (1곳, 죽은 경로)** | [calibration_dialog.py:13](calibration_dialog.py#L13) — `from mfcc_svm import MfccSvmClassifier`. 단 `calibration_dialog` 자체가 어디서도 import되지 않음 (아래 항목 참조) |
| `calibration_dialog.py`가 어디서 import되는가 | **없음** | grep 결과 `calibration_dialog` 토큰은 자기 자신 docstring([calibration_dialog.py:2](calibration_dialog.py#L2))에서만 등장. main.py / ui_window.py 어디에도 없음 |
| `CalibrationDialog`를 인스턴스화하는 코드 | **없음** | grep `CalibrationDialog` 결과 [calibration_dialog.py:87](calibration_dialog.py#L87)의 클래스 **정의** 1건뿐. 호출/인스턴스화 0건 |
| `vowel_classifier.py`에 `set_calibration` 정의 | **없음** | grep `def set_calibration` 결과 정의 위치는 백업 파일 [wav2vec_classifier_ORIG.py:190](wav2vec_classifier_ORIG.py#L190) 1건뿐. 현행 [vowel_classifier.py](vowel_classifier.py) 안에는 함수 자체가 존재하지 않으며, 단지 [vowel_classifier.py:9](vowel_classifier.py#L9)·[L57](vowel_classifier.py#L57) **docstring·주석에만 이름이 남아 있음** (stale comments) |
| `load_calibration` 호출처 | **없음** | grep `load_calibration\(` 결과 [calibration_dialog.py:282](calibration_dialog.py#L282) **정의** 1건만. 호출 0건 |

### 종합 판정 — 죽은 코드 사슬

```
mfcc_svm.py ─────► (import 1건) calibration_dialog.py
                                  │
                                  ├─ CalibrationDialog 클래스 (인스턴스화 0건)
                                  └─ load_calibration()       (호출 0건)
                                  ▲
                                  └─ 어디서도 import 0건
```

→ **`mfcc_svm.py`, `calibration_dialog.py` 두 파일 + `vowel_classifier.py`의 docstring/주석 잔재**가 죽은 코드.
→ `set_calibration`은 **현행 코드 어디에도 정의가 없는 유령 함수**. 주석만 남아 혼동을 줌.
→ 백업 파일 `formant_engine_ORIG.py`, `wav2vec_classifier_ORIG.py`도 import 그래프 바깥(별도 확인 필요시 추가 조사).

---

## 2. 보정 데이터 흔적

| 파일 | 존재 여부 |
|------|---------|
| `user_proto.npz` | **없음** |
| `user_svm_model.pkl` | **없음** |
| `user_calibration.json` | **없음** |

`ls *.npz *.pkl *.json` 결과: 매칭 파일 0건. 루트 폴더에 보정 산출물이 없음.

참고: [wav2vec_classifier.py:26](wav2vec_classifier.py#L26)에 `PROTO_FILE = Path(__file__).parent / "user_proto.npz"` 상수가 정의되어 있으나, 현행 클래스([wav2vec_classifier.py:302](wav2vec_classifier.py#L302) 이하)에서 이 상수를 참조하는 코드가 없음 — 이것도 죽은 상수 가능성 높음 (상세 검증은 다음 단계 권장).

---

## 3. 분류 흐름 정리

### 3-1. `ui_window.py::_tick()` — `vowel_raw` 결정 블록

위치: [ui_window.py:796-822](ui_window.py#L796-L822)

```
800  # 성별별 분류기 입력 결정
801  if trk_vtype == "male":
802      clf_gender = "male"
803      clf_scale  = 1.0                           # 남성: scale 미적용
804  else:
805      clf_gender = "female"
806      clf_scale  = trk_scale if trk_ready else 1.0
807
808  if iv:                                          # is_voice_final
809      if wv_vowel != "?" and wv_conf > 0.15:      # ← wav2vec2 채택 조건
810          vowel_raw, v_conf = wv_vowel, wv_conf
811      else:                                       # ← classify_vowel 폴백
812          raw_f3 = result.get("raw_f3")
813          vowel_raw, v_conf = classify_vowel(
814              f1, f2, clf_gender, f3=raw_f3,
815              scale=clf_scale,
816          )
817          agr = result.get("agreement", 0.0)
818          if agr > 0.4 and vowel_raw != "?":
819              v_conf = min(1.0, v_conf + agr * 0.15)
820  else:                                           # 무음
821      vowel_raw, v_conf = "?", 0.0
```

- **wav2vec2 결과 채택 조건**: `iv == True` AND `wv_vowel != "?"` AND `wv_conf > 0.15` ([ui_window.py:809](ui_window.py#L809))
- **`classify_vowel` 폴백 조건**: `iv == True` AND (`wv_vowel == "?"` OR `wv_conf <= 0.15`) ([ui_window.py:811-816](ui_window.py#L811-L816))
- **`_formant_only()`가 호출되는 모든 경로**: `_tick()`에서는 **직접 호출 안 함**. 오직 `wav2vec_classifier.classify()` 내부에서만 호출됨 (아래 3-2 참조).

### 3-2. `wav2vec_classifier.py::classify()` — KNN/포먼트 분기

위치: [wav2vec_classifier.py:426-470](wav2vec_classifier.py#L426-L470)

```
429  if not self._ready:                              # 모델 로딩 중
430      if f1>80 and f2>250 and gender=="female":
431          _normalizer.update(f1, f2)
432      return self._formant_only(...)              # ← 경로 A: 모델 미준비
434
437  if self._default_proto:                          # 기본 합성 prototype 준비됨
438      knn_vowel, knn_conf = self._knn_classify(a16k, self._default_proto)
439
440      if f1>80 and f2>250:
441          if gender=="female":
442              _normalizer.update(f1, f2)
443          fmt_prob = formant_vowel_probs(f1, f2, gender)
444          fmt_best = max(fmt_prob, key=fmt_prob.get)
445          fmt_conf = fmt_prob[fmt_best]
446
447          # KNN+포먼트 일치 → KNN 결과 부스트해서 직접 반환
448          if knn_vowel == fmt_best and knn_conf >= 0.10:
449              return knn_vowel, min(knn_conf*1.3, 1.0)   # ← KNN 직접 반환
450
451          # 포먼트 단독 신뢰 → _formant_only 위임
452          if fmt_conf > 0.35:
453              return self._formant_only(...)              # ← 경로 B
454
455      # 포먼트 없거나 낮은 확률 → KNN/포먼트 최후 폴백
456      if knn_conf < 0.08:
457          if f1>80 and f2>250:
458              return self._formant_only(...)              # ← 경로 C
459          return "?", knn_conf                            # ← KNN 직접 반환 (?)
460      return knn_vowel, knn_conf                          # ← KNN 직접 반환
461
462  # default_proto 미준비 → 포먼트 폴백
463  if f1>80 and f2>250 and gender=="female":
464      _normalizer.update(f1, f2)
465  return self._formant_only(...)                          # ← 경로 D
```

내부적으로 `_knn_classify` ([wav2vec_classifier.py:406-422](wav2vec_classifier.py#L406-L422))는 코사인 유사도 최댓값이 0.45 미만이면 `("?", confidence)` 반환.

### 3-3. 의사코드 (전체 분류 흐름)

```
function decide_vowel(audio_chunk, f1, f2, f3, gender, trk):

    # ── ui_window._tick() 레벨 ────────────────────────────────
    if not is_voice_final:
        return "?"

    # 성별 결정
    if trk.voice_type == "male":
        clf_gender, clf_scale = "male", 1.0
    else:
        clf_gender = "female"
        clf_scale  = trk.scale if trk.ready else 1.0

    # wav2vec2 분류기 호출 (결과는 _analysis_loop에서 미리 계산됨)
    wv_vowel, wv_conf = wav2vec_clf.classify(audio_chunk, f1, f2, f3, gender)

    # 1차: wav2vec2 결과 사용
    if wv_vowel != "?" and wv_conf > 0.15:
        return wv_vowel, wv_conf

    # 2차: 포먼트 Mahalanobis 폴백
    vowel, conf = classify_vowel(f1, f2, clf_gender, f3=raw_f3, scale=clf_scale)
    if agreement > 0.4:
        conf = min(1.0, conf + agreement * 0.15)
    return vowel, conf


# ──────────────────────────────────────────────────────────────
# wav2vec_clf.classify() 내부 — wv_vowel/wv_conf 계산
# ──────────────────────────────────────────────────────────────
function wav2vec_clf.classify(audio, f1, f2, f3, gender):

    # 경로 A: 모델 로딩 중
    if not model_ready:
        return _formant_only(f1, f2, f3, gender, audio)

    # KNN: 합성 prototype 기반 코사인 유사도
    if default_proto_ready:
        knn_vowel, knn_conf = _knn_classify(audio, default_proto)
            # → 최댓값 < 0.45 이면 ("?", conf) 반환

        if f1, f2 valid:
            fmt_best, fmt_conf = formant_vowel_probs.argmax(f1, f2, gender)

            # KNN과 포먼트 일치 → KNN 부스트 후 직접 반환
            if knn_vowel == fmt_best and knn_conf >= 0.10:
                return (knn_vowel, knn_conf * 1.3)

            # 포먼트 단독 신뢰도 충분 → 포먼트 전용 분류 위임
            if fmt_conf > 0.35:
                return _formant_only(f1, f2, f3, gender, audio)   # 경로 B

        # KNN 신뢰도 매우 낮음 → 최후 폴백
        if knn_conf < 0.08:
            if f1, f2 valid:
                return _formant_only(...)                          # 경로 C
            return ("?", knn_conf)

        # 그 외 → KNN 결과 그대로
        return (knn_vowel, knn_conf)

    # 경로 D: default_proto 미준비
    return _formant_only(f1, f2, f3, gender, audio)
```

**KNN 결과가 직접 반환되는 조건 (3가지)**
1. `knn_vowel == fmt_best` AND `knn_conf >= 0.10` (라인 [448-449](wav2vec_classifier.py#L448-L449)) — 부스트 1.3×
2. `knn_conf < 0.08` AND 포먼트 무효 (라인 [459](wav2vec_classifier.py#L459)) — `("?", conf)`
3. 위 어느 조건도 안 맞을 때 (라인 [460](wav2vec_classifier.py#L460)) — knn 그대로

**`_formant_only()`로 위임되는 조건 (4가지 경로)**
- A: 모델 미로딩 ([wav2vec_classifier.py:433](wav2vec_classifier.py#L433))
- B: KNN/포먼트 불일치 + `fmt_conf > 0.35` ([:455](wav2vec_classifier.py#L455))
- C: `knn_conf < 0.08` + 포먼트 유효 ([:460](wav2vec_classifier.py#L460))
- D: `default_proto` 미준비 ([:467](wav2vec_classifier.py#L467))

---

## 4. _REFS 값 비교 (vowel_classifier.py female vs 학계)

출처: [vowel_classifier.py:25-37](vowel_classifier.py#L25-L37) `_REFS["female"]`
학계: 하영우·오재혁(2017) 여성 아나운서 8명 평균

| 모음 | 코드 F1 | 학계 F1 | F1 차이 | 코드 F2 | 학계 F2 | F2 차이 |
|------|--------:|--------:|--------:|--------:|--------:|--------:|
| 아 | 978 | 996 | −18 | 1397 | 1503 | −106 ⚠️ |
| 에 | 548 | 477 | +71 | 2125 | 2514 | −389 ⚠️ |
| 이 | 352 | 289 | +63 | 2787 | 2716 | +71 |
| 오 | 487 | 363 | +124 ⚠️ | 840 | 642 | +198 ⚠️ |
| 우 | 367 | 332 | +35 | 660 | 832 | −172 ⚠️ |
| 으 | 435 | 344 | +91 | 1404 | 1711 | −307 ⚠️ |
| 어 | 671 | 629 | +42 | 1212 | 950 | +262 ⚠️ |

⚠️ 표시: |차이| ≥ 100 Hz

### 차이가 큰 항목 요약

- **F1 차이 ≥ 100 Hz** : 오 (+124)
- **F2 차이 ≥ 100 Hz** : 아(−106), 에(−389), 오(+198), 우(−172), 으(−307), 어(+262)
- **F1·F2 둘 다 ≥ 100** : **오** (F1 +124, F2 +198)
- **F2 절대오차 최대값** : 에 (−389 Hz) → 코드값이 학계보다 389Hz 낮음
- **F2 절대오차 최소값** : 이 (+71 Hz)

### 패턴 관찰

1. 코드의 _REFS는 **F1을 일관되게 높은 쪽으로**(7개 중 6개가 +) 설정.
2. 코드의 _REFS는 **F2를 절대값 차이가 매우 크게** 설정 (4개가 |Δ|≥170).
3. **에**의 F2(2125 vs 학계 2514)와 **으**의 F2(1404 vs 학계 1711)가 학계보다 크게 낮게 잡혀 있어, 실제 발화의 에·으가 _REFS상 인접 모음(이/어)으로 오분류될 가능성 시사.
4. **오 vs 우**의 F2 차이가 코드(660→840=180)와 학계(832→642=−190, 부호 반대) 사이에서 **부호가 반대**: 학계는 우(832)>오(642)인데 코드는 오(840)>우(660). 이는 _REFS 출처(Yoon 2015 등)와 학계(하·오 2017) 간 모집단 차이일 가능성.

---

## 끝

코드 수정 0건. 신규 파일은 `evaluation/diagnostic_report.md` 1개뿐. 다음 단계 지시 대기 중.
