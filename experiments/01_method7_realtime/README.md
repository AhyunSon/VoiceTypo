# 01 실시간 한국어 단모음 분석기 (통합 시도)

작성자: jaewon
브랜치: jaewon
날짜:  2026-05-21

## 무엇 / 왜

실시간 한국어 7모음(아/에/이/오/우/으/어) 분석기.
포먼트·MFCC SVM·wav2vec 등 여러 분류기를 한 파이프라인에 통합해
실시간 인식 + 캘리브레이션 + UI 까지 묶은 초기 통합 버전.

원래 VoiceTypo `main` 의 `vowel_recognition/method_7_realtime_integrated` 에
있던 작업을 개인 브랜치로 옮긴 것. (main 은 통합 프로젝트 전용이므로
개인 실험 코드는 개인 브랜치에서 관리)

## 구성

- `main.py` / `ui_window.py` — 실행 진입점, 실시간 UI
- `audio_stream.py` / `vad.py` — 오디오 입력, 음성 구간 검출
- `formant_engine.py` / `formant_ensemble.py` — 포먼트 추출
- `vowel_classifier.py` / `mfcc_svm.py` / `wav2vec_classifier.py` — 분류기들
- `calibration_dialog.py` — 화자 캘리브레이션 UI
- `diagnostic.py` — 진단 도구

## 실험 로그   (위가 최신, 계속 누적 — 실패도 기록)

- 5/21  main 에서 개인 브랜치로 이관 (initial upload)
- (이전 작업 내역은 작성자가 채워주세요)

## 상태

보류 — 이후 02_formant 로 포먼트 접근을 분리해 이어감
