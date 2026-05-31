# 05 시각 스케치 — 스터디로그 실험 (목소리 → 시각화)

작성자: jaewon
브랜치: jaewon
기간:  2026-05-29 ~ 진행중

## 무엇 / 왜

지금까지(01~04)는 전부 **"인식"**(ㅏ를 알아채기) 트랙이었다.
이 폴더는 그 반대편 — **"시각화"**(ㅏ를 무엇으로 보일 것인가) 를 실험한다.
05.29 스터디로그의 작가들(Levin·Lieberman·Reas·Maeda·Rozin·Akten)의 **개념을 우리 코드로 직접 구현**.

> 마이크 정책(§10)상 집에선 측정이 아니라 **코드·튜닝**만 — 시각화 실험이 정확히 그 범주.
> 라이브 마이크는 "느낌 확인"용. 결과 데이터로 쓰지 않음.

## 우리 파이프라인

```
Mic Input → Audio Analysis → Visual Mapping → Realtime Rendering
            (voice_input.py)   (각 sketch)      (pygame / p5.js)
```

**우리 매핑 (핵심 — 자유롭게 수정):**

| 음성 | → 시각 |
| --- | --- |
| F1 (개/폐) | 글자 높이 |
| F2 (전/후설) | 글자 폭 |
| Pitch (F0) | 회전 |
| Volume (RMS) | 크기 |
| 모음 | 형태 (vowel_weights 연속 블렌딩) |
| jitter | 떨림/거칠기 |

## 공통 모듈

- **`voice_input.py`** — 마이크 → `VoiceSignal`(f0·f1·f2·f3·rms·jitter·vowel·**vowel_weights**).
  02_formant 의 포먼트 추출을 재사용하되 자체 완결(독립 실행). 단독 실행 시 콘솔에 값 출력.
  `vowel_weights` 는 단정적 분류가 아닌 7모음 연속 가중치 → Reas 식 규칙 블렌딩의 입력.

## 스케치 (멘토 ↔ 실행 파일)

| 파일 | 멘토 | 내용 |
| --- | --- | --- |
| `sketch_01_mapping_vocabulary.py` | Levin / Rozin | 매핑 어휘집: 파라미터→형태(블롭). f0·rms·f1·f2·jitter |
| `sketch_02_vowel_letterform.py` ⭐ | Lieberman / Reas / Maeda | **우리 파이프라인 본체** — 글자에 F1/F2/Pitch/Volume/모음 5매핑 |
| `sketch_03_voice_mirror.py` | Rozin | 목소리 거울 — 타일 밭에 1:1 반사 |
| `sketch_04_kinetic_typo.py` | Maeda | 키네틱 타이포 — 입자가 글자로 모였다 흩어짐 |
| `sketch_05_messa_di_voce.py` | Levin / Lieberman | **우리 버전 Messa di Voce** — 말하면 형태가 태어나 흐름 |
| `ai_generative/asemic_strokes.py` | Akten | 생성형 손글씨(개념 프로토타입, CPU) |
| `web_p5js/` | Reas / Processing | 같은 매핑의 웹(p5.js) 버전 + WebSocket 전송 레이어 |

## 실행

```bash
# 02_formant 의 가상환경 재사용 (pygame·parselmouth·numpy·sounddevice·websockets)
cd experiments/05_visual_sketches
source ../02_formant/.venv/bin/activate    # (윈도: ..\02_formant\.venv\Scripts\activate)

python voice_input.py                  # 먼저 마이크/추출 동작 확인 (콘솔 값)
python sketch_02_vowel_letterform.py   # 본체
python sketch_05_messa_di_voce.py      # 통합 작품
# 웹: python web_p5js/bridge.py  → web_p5js/index.html 을 브라우저로 열기
```

각 창은 **ESC** 로 종료. (sketch_05·asemic 은 **C** 로 화면 비움)

## 상태

- 스케치 5개 + AI 개념 프로토타입 + 웹 파이프라인 **작성·문법검증 완료**.
- 다음: 실제 마이크로 돌려보며 **매핑 표 튜닝**(이게 진짜 실험 — Reas 식 "규칙이 형태를 만든다").
- 채택되면 §4 규칙대로 main 에서 새 feature 브랜치로 깔끔히 재구현 → PR.

크레딧·저작권 안전 노트는 **`REFERENCES.md`** 참조. 진행 로그는 **`HANDOFF.md`**.
