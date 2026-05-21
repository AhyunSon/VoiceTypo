# 01 실시간 한국어 단모음 분석기 (초기 버전)

작성자: jaewon
브랜치: jaewon
업로드: 2026-05-21

## 무엇 / 왜

포먼트·wav2vec2 기반 실시간 한국어 7모음(아/에/이/오/우/으/어) 인식 시스템.
마이크 입력 → 실시간 인식 → GUI 시각화 + 개인 캘리브레이션.

원래 VoiceTypo `main` 의 `vowel_recognition/method_7_realtime_integrated` 에 있던 것.
("method_7" 은 폴더에 들어갈 때 붙은 번호일 뿐 — 방법 1~6 과는 무관한 별개 작업.)

> **이 폴더는 `02_formant` 프로젝트의 "정리 전 초기 버전"입니다.**
> 파일 구성이 02_formant 와 거의 같고, 02_formant 가 이 코드를 정리·발전시킨 것입니다.
> 시도·방향 전환의 **전체 흐름은 [02_formant README](../02_formant/README.md) 에 정리**돼 있습니다.
> 이 폴더는 그 출발점을 히스토리로 남겨둔 스냅샷입니다.

## 이 버전의 구조

복잡한 다층 구조 — 분류기 여러 개 + 앙상블 + Kalman 안정화를 한 번에 얹은 형태.

```
마이크 입력 (audio_stream)
  │
  ▼
VAD — 음성 구간 검출 (vad)        3-조건: RMS + 자기상관 + ZCR
  │
  ▼
특징 추출
  ├─ 포먼트 F0/F1/F2/F3  (formant_engine + formant_ensemble 3-방법 앙상블)
  └─ wav2vec2 임베딩 768d (wav2vec_classifier)
  │
  ▼
분류
  ├─ wav2vec2 layer-8 K-NN    ← 주 분류기
  ├─ 포먼트 Bark Mahalanobis  ← 보조 / 폴백
  └─ MFCC+CMVN SVM            ← 교차 검증
  │
  ▼
캘리브레이션 보정 (calibration_dialog)
  │
  ▼
실시간 UI (ui_window)  시계열 그래프 + F1/F2 모음 공간
```

| 요소 | 내용 |
| --- | --- |
| 주 분류기 | wav2vec2 layer-8 은닉벡터 K-NN (코사인 유사도) |
| 포먼트 추출 | 3-방법 앙상블 (Praat Burg / pyworld CheapTrick / scipy LPC) + Kalman 안정화 |
| 보조 분류 | 포먼트 Bark Mahalanobis (성별별 참조값, F3 로 우/오 구분) |
| 교차 검증 | MFCC+CMVN+SVM |
| VAD | 적응형 3-조건 (RMS + 자기상관 주기성 + ZCR) |
| 캘리브레이션 | 개인 모음 보정 다이얼로그 |

## 결과 — 그리고 02_formant 로 이어짐

이 초기 버전은 평가에서 **약 22.9%** 에 그침.

이후 진단(02_formant `evaluation/diagnostic_report.md`)에서 드러난 문제:
- 분류기 여러 개 + 앙상블 + Kalman 의 **복잡도가 오히려 정확도를 깎고 있었음**
- wav2vec2 K-NN 과 포먼트 분류기의 분기 로직이 얽혀 결과가 불안정
- 죽은 코드(미사용 `mfcc_svm`, `calibration_dialog` 등) 다수

→ 이 코드를 복사해 복잡한 로직을 걷어낸 것이 **02_formant**.
   앙상블·Kalman·wav2vec K-NN 을 제거하자 **22.9% → 54.3%** 로 상승.
   그 뒤 Phase A/B 까지의 전체 여정은 [02_formant README](../02_formant/README.md) 참고.

## 코드 구성

| 파일 | 역할 |
| --- | --- |
| `main.py` | 실행 진입점 |
| `ui_window.py` | GUI — 시계열 그래프 + F1/F2 모음 공간, 백그라운드 분석 스레드 |
| `audio_stream.py` | sounddevice 마이크 캡처 (deque 링버퍼) |
| `vad.py` | 적응형 3-조건 음성 구간 검출 |
| `formant_engine.py` | 포먼트 추출 (pyworld F0 + Praat LPC + Kalman 안정화) |
| `formant_ensemble.py` | 3-방법 앙상블 포먼트 추출기 |
| `wav2vec_classifier.py` | wav2vec2 layer-8 K-NN 모음 분류 (주 분류기) |
| `vowel_classifier.py` | Bark Mahalanobis 포먼트 분류 (보조) |
| `mfcc_svm.py` | MFCC+CMVN+SVM 교차 검증 |
| `calibration_dialog.py` | 개인 모음 보정 다이얼로그 |
| `diagnostic.py` | 마이크 + 분석 파이프라인 단계별 진단 |
| `config.py` | 설정값 중앙 관리 |

## 상태

보존용 스냅샷 — 02_formant 의 정리 전 출발점.
실험은 02_formant 에서 이어짐 (Phase A/B, 최신 92.9%).
