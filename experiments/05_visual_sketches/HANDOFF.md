# HANDOFF — 05 시각 스케치

## 진행 로그  (위가 최신 — 누적, 실패도 기록)

### 2026-06-01 (집) — 실행·디버그·모음인식 진단
- 5개 스케치 + AI + 웹 전부 실제 실행 단계 진입.
- **버그 수정**: 스케치들이 `sig.jit` 로 잘못 접근(실제 필드 `sig.jitter`) → 소리 날 때 sketch_05 크래시.
  sketch_01·04·05 수정, 전 스케치 `sig.` 속성 전수검사로 재발 차단.
- **무음 오인식**: rms_gate 0.004→0.013 (조용할 때 으/어 false trigger 제거).
- **개인 캘리브레이션 추가**: `calibrate.py`(7모음 녹음→`my_vowels.json`),
  voice_input 이 있으면 개인 기준값 사용. 평균값(cal-free) 대신 본인 목소리 기준.
- **모음 인식 잘 안 됨 — 원인 진단(본인 콘솔 데이터 근거):**
  1. (최대) 고음(F0 220~360Hz) → 닫힌모음 F1 이 F0 배음으로 오인. 예: `f0=347 f1=348`.
  2. 포먼트 추출 프레임마다 튐. 같은 "아"에서 f1 1094→248→819 (LPC 오추적).
  3. 전설모음(이·에) F2 덜 잡힘(이 F2가 1800 천장, 기대 2700). → 중앙모음으로 무너짐.
  4. 분류기가 평균값 거리재기(cal-free, 02 기록상 54% 천장).
  5. 어/오/우 후설 중첩(본질적 한계).

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

## 다음 작업 (내일 연구실에서 — 모음인식 개선 우선)
1. **`python calibrate.py` 1회** — 7모음 등록(= 진단 겸함, 본인 F1/F2 실측 확인).
   - 특히 "이" F2 가 정말 낮은지, 닫힌모음 F1 이 F0 와 붙는지 눈으로 확인.
2. **F0/성별-aware 포먼트 추출 적용** (원인 1·2 직격) — config.py 의 검증된 female 파라미터
   (max_formants=4, window 50ms)로 배음→F1 오인 방지. + 다중시점 중앙값으로 프레임 튐 제거.
   → voice_input._analyze 개선. (또는 02_formant/formant_engine.py 채택 검토)
3. 개선 후 sketch_02 재실행 → 모음별 인식 비교(어느 게 좋아졌나 로그).
4. **매핑 표 튜닝**(README 표) — F1/F2 정규화 범위, 회전 각도, 모음 규칙 오프셋. 변수 하나씩.
5. 마음에 드는 방향 1개 → §4 규칙대로 main 새 feature 브랜치로 재구현 → PR.

> 해결책 풀(우선순위): ①캘리브레이션 ②F0/성별-aware 추출 ③다중시점 중앙값
> ④시간평활+신뢰도 게이트 ⑤formant_engine 채택 ⑥Lobanov+LDA 이식(무거움) ⑦시각화를 연속가중치로.

## 알려진 한계 / 메모
- `vowel_weights` 는 02_formant 의 정식 분류기(Bark Mahalanobis/LDA)가 아니라
  ref 중심까지의 Bark 거리 softmax(가벼운 근사). 시각화엔 충분, 정확도용 아님.
- Akten 레이어는 **학습형 아님**(GPU 필요) — 절차적 스탠드인. 연구실에서 학습형으로 교체 예정.
- 집 마이크 데이터는 결과로 쓰지 않음(§10). 여기 실험은 전부 코드·튜닝용.
