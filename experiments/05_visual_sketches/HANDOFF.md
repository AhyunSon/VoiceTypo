# HANDOFF — 05 시각 스케치

## 진행 로그

### 2026-05-29 (집 컴퓨터, 셋업 직후)
- 05.29 스터디로그(Levin·Lieberman·Reas·Maeda·Rozin·Akten)를 **전부 실험**하는 게 목표.
- 02_formant 의 venv(python3.12) 재사용. `websockets` 추가 설치.
- 작성:
  - `voice_input.py` — 공통 입력(마이크→VoiceSignal, vowel_weights softmax)
  - `sketch_01` 매핑 어휘집 / `sketch_02` 글자꼴(본 파이프라인) / `sketch_03` 거울 /
    `sketch_04` 키네틱 타이포 / `sketch_05` 우리 Messa di Voce
  - `ai_generative/asemic_strokes.py` — Akten 개념 프로토타입(CPU 절차적)
  - `web_p5js/bridge.py` + `index.html` — 웹(p5.js) + WebSocket 전송 레이어
  - `REFERENCES.md`(크레딧/저작권) · `README.md`
- 검증: 7개 Python 파일 **py_compile 통과**, vowel_weights 계산 정상(합=1, "아"근처→"아").
  GUI/마이크 실행은 본인 화면에서 확인 필요(에이전트 환경에선 창 못 봄).

## 다음 작업
1. **실제 마이크로 각 스케치 실행** → 느낌 확인. 특히 sketch_02·05.
2. **매핑 표 튜닝**(README 의 표) — 이게 핵심 실험. 변수 하나씩.
   - 예: F1 정규화 범위(250~1000) 화자에 맞게, 회전 각도(±40°), 모음 규칙 오프셋.
3. jitter 근사 품질 점검(현재 최근 F0 상대표준편차) — 떨림 시각화가 자연스러운지.
4. 마음에 드는 방향 1개 선정 → §4 규칙대로 main 새 feature 브랜치로 재구현 → PR.

## 알려진 한계 / 메모
- `vowel_weights` 는 02_formant 의 정식 분류기(Bark Mahalanobis/LDA)가 아니라
  ref 중심까지의 Bark 거리 softmax(가벼운 근사). 시각화엔 충분, 정확도용 아님.
- Akten 레이어는 **학습형 아님**(GPU 필요) — 절차적 스탠드인. 연구실에서 학습형으로 교체 예정.
- 집 마이크 데이터는 결과로 쓰지 않음(§10). 여기 실험은 전부 코드·튜닝용.
