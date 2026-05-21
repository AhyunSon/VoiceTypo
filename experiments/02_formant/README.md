# 02 포먼트 기반 7모음 인식

작성자: jaewon
브랜치: jaewon
날짜:  2026-05-21

## 무엇 / 왜

포먼트(F1/F2/F3)만으로 한국어 7모음을 실시간 인식.
미디어아트 설치 특성상 학습된 표현(DNN) 대신 연속적인 포먼트 값이
필요(실시간 모핑용)해서 포먼트 접근을 별도로 깊게 탐색.

## 구성

- `main.py` / `ui_window.py` — 실행 진입점, 실시간 UI
- `formant_engine.py` — 포먼트 추출 (Praat Burg F1/F2/F3)
- `vowel_classifier.py` / `multi_prototype.py` — 모음 분류
- `vtln.py` / `lobanov_lda.py` — 화자 정규화 (VTLN, Lobanov+LDA)
- `cal_setup.py` / `cal_dialog.py` / `calibrator.py` — 캘리브레이션
- `vad.py` / `audio_stream.py` — 음성 구간 검출, 오디오 입력
- `diagnose.py` / `diagnose_live.py` / `diagnostic.py` — 진단
- `evaluation/` — 평가 스크립트, `step4_korean/` — 한국어 데이터 처리

## 실험 로그   (위가 최신, 계속 누적 — 실패도 기록)

- 5/21  개인 브랜치로 정리 업로드 (initial upload)
- Phase B  Lobanov + LDA + dynamic feature → 92.9% (합성 데이터 기준, 라이브 검증 필요)
- Phase A  cal-free 천장 54.3% 확인 → VTLN 만 채택, 나머지 기법 폐기
- 5/6    라이브 테스트: cal + GMM 도 미흡 확인
- (세부 수치/날짜는 작성자가 보강해주세요)

## 상태

진행중 — 다음: Phase B 결과의 라이브 검증
