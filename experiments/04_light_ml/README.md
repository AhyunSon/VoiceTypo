# 04 경량 ML 접근

작성자: jaewon
브랜치: jaewon
날짜:  2026-05-21

## 무엇 / 왜

대형 SSL 모델(Whisper 등) 대신 가벼운 ML 모델로 7모음을 분류하는 접근.
적은 연산으로 실시간 동작 가능한지 확인하기 위한 실험.

## 구성

- `features.py` — 특징 추출
- `dataset.py` / `augment.py` — 데이터셋 구성, 증강
- `model.py` — 경량 모델 정의
- `train.py` / `train_v4.py` — 학습
- `evaluate.py` / `evaluate_v4.py` / `live_eval.py` — 평가, 라이브 평가

## 실험 로그   (위가 최신, 계속 누적 — 실패도 기록)

- 5/21  개인 브랜치로 정리 업로드 (initial upload)
- (이전 작업 내역·결과 수치는 작성자가 채워주세요)

## 상태

(작성자가 현재 상태를 적어주세요)
