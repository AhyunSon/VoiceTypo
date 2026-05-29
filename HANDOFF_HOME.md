# 🏠 집 컴퓨터 초기 셋업 핸드오프

> **이 파일을 집 Claude 에게 그대로 전달하세요.** 프로젝트 맥락 + 셋업 작업 + 동기화 규칙이 한 번에 들어 있습니다.

작성: 2026-05-21 · 작성자: jaewon (`jwh2700@gmail.com`)

---

## 1. 프로젝트 한 줄

**VoiceTypo** — 한국어 7모음(아/에/이/오/우/으/어) 실시간 인식 + 시각화 프로젝트.
6명 학부 인턴팀 (리더: **ahyun**, 본인: **jaewon**).
레포: <https://github.com/AhyunSon/VoiceTypo> (Private)

## 2. 팀 / 브랜치 현재 상태 (2026-05-21 기준)

| 브랜치 | 내용 |
| --- | --- |
| `main` | **재구축 중** — README 만. 통합 코드는 `ahyun` 브랜치로 이관됨 |
| `ahyun` | `experiments/01_voicetypo_integrated/` — 이전 통합 프로젝트 보존 |
| **`jaewon` (본인)** | `experiments/01~04/` — 본인 작업 4개 |
| `eunbin` `jinhee` `seoeun` `sunmin` | 빈 초기 구조 (README + .gitignore + experiments/) |

## 3. jaewon 브랜치의 4개 실험

| 폴더 | 내용 |
| --- | --- |
| `01_method7_realtime` | 02 의 초기 버전 (~22.9%) — 동작은 함. 보존용 스냅샷 |
| **`02_formant`** | **본 트랙** — 포먼트 기반, Phase A/B 까지. 합성 다화자 92.9%, 실제 다화자 미통과 |
| `03_whisper_ssl` | Whisper SSL + MLP — 화자 일반화 성공, 정확도 0.66 천장 |
| `04_light_ml` | MFCC+CNN/Whisper-tiny — v1~v4. **현재 중단** (본 방향이 포먼트) |

각 폴더 `README.md` 에 상세 (방향 전환·실험 로그·결과·다음 작업).
가장 자세한 건 `experiments/02_formant/README.md`.

## 4. 협업 규칙 (반드시 지킬 것)

| 규칙 | 내용 |
| --- | --- |
| **main 직접 push 금지** | feature 브랜치 + PR 로만. 다른 사람 브랜치도 절대 안 건드림 |
| **개인 브랜치 = 1인 1개** | jaewon 안에 서브 브랜치 X. 실험은 `experiments/번호_주제/` 폴더로 분리 |
| **feature 브랜치는 최신 main 에서 분기** | 채택 실험을 새로 깔끔히 재구현 → PR → main. 개인 브랜치 직접 merge 안 함 |
| **커밋 메시지** | Conventional Commits. type/scope = 영어, **설명은 한국어 명사형 끝맺음**. 예: `exp(formant): 임계값 0.3→0.5 테스트` |
| **재구현 커밋엔 `Origin:` 푸터** | 실험 폴더를 가리킴 (feature 브랜치 X). 예: `Origin: jaewon/experiments/01_formant_cal` |
| **대용량 파일 커밋 금지** | 음성·모델·.venv 는 `.gitignore` 가 자동 제외 |
| **변수는 한 번에 하나만** | 비교 시 마이크·split·하이퍼파라미터 동시 변경 X |
| **HANDOFF.md 작성 습관** | 실험 코드엔 `HANDOFF.md` (진행 로그·결과·다음 작업) 같이 유지 |

## 5. 본 트랙이 왜 포먼트(02)인가

연구실 요구: **DNN 학습 표현 비사용** + 실시간 모핑을 위해 **연속적인 F1/F2 값** 필요.
→ 분류 모델(03·04)은 모음 라벨만 줄 뿐 모핑용 연속 포먼트를 못 줌 → 비교용 탐색.
→ **02_formant 가 본 방향**, 03·04 는 휴면/중단.

---

## 6. 집 컴퓨터 초기 셋업 (Claude 가 도와줄 작업)

### Step 1. 필수 도구

| 도구 | 설치 |
| --- | --- |
| Git | <https://git-scm.com> |
| Python 3.10+ | <https://python.org> |
| VS Code (권장) | <https://code.visualstudio.com> |

설치 안 돼 있으면 안내해 주세요.

### Step 2. 레포 clone + jaewon 체크아웃

```bash
cd Desktop
git clone https://github.com/AhyunSon/VoiceTypo.git
cd VoiceTypo
git checkout jaewon
git pull origin jaewon
```

### Step 3. git 사용자 정보 설정

```bash
git config user.name "hurjwon"
git config user.email "jwh2700@gmail.com"
```

### Step 4. 본 트랙 환경 구축 — `02_formant` 우선

```bash
cd experiments/02_formant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

필요시 03·04 도 같은 방식 (03 학습은 GPU 필요 — 집 GPU 없으면 추론만).

### Step 5. 작동 확인

```bash
# .venv 활성화된 상태에서
python -c "from formant_engine import FormantEngine; print('OK')"
```

→ `OK` 출력되면 환경 정상.

### Step 6. 데이터 (필요할 때만)

음성 `.wav` 와 모델 체크포인트는 git 미포함:
- **USB 로 한 번 옮기기** (가장 빠름)
- 공개 코퍼스는 자체 다운로드 스크립트 사용

---

## 7. 일상 동기화 워크플로 (반복)

### 시작할 때 — pull

```bash
cd Desktop/VoiceTypo
git checkout jaewon
git pull origin jaewon
```

### 끝낼 때 — commit + push

```bash
git add .
git commit -m "exp(formant): 임계값 조정 테스트"
git push origin jaewon
```

### 주말 사이클 예시

```
금 연구실 끝낼 때:  pull → 작업 → commit → push
토 집에서 시작:    pull → 작업 → commit → push
일 집에서 끝낼 때: pull → 작업 → commit → push
월 연구실 시작:    pull → (집에서 한 거 그대로 받음)
```

### 종료 1줄 체크

> **"오늘 commit 하고 push 했나?"** ← 이거만 확인

---

## 8. 흔한 사고 + 대처

| 사고 | 해결 |
| --- | --- |
| **push 깜빡함** | 다음날 대기 또는 VS Code Remote SSH 로 원격 push |
| **작업 중간이라 commit 애매** | `git commit -m "WIP: ..."` 으로라도 올림. 안 올린 것보다 백 배 나음 |
| **양쪽에서 같은 파일 수정 → 충돌** | `git pull` 시 표시. VS Code 가 충돌 부분 하이라이트 → 한 쪽 선택 |
| **요구사항 추가됨** | 양쪽 컴퓨터에 `pip install -r requirements.txt` 다시 |

---

## 9. git 이 안 옮기는 것 (집에 따로 준비)

| 항목 | 처리 |
| --- | --- |
| `.venv/` (가상환경) | 각 컴퓨터에 따로. `requirements.txt` 동일 → 환경 동일 |
| 음성 데이터 (`*.wav`) | USB 한 번 → 그 뒤 자주 안 바뀜 |
| 모델 체크포인트 (`*.pt`, `*.npz`) | 연구실 GPU 학습 → Google Drive 백업 → 집에서 추론 |
| Zeroth-Korean 등 공개 코퍼스 | 다운로드 스크립트가 자동으로 받음 |

---

## 10. 마이크 정책

- **집에서 녹음한 데이터는 실험 결과로 쓰지 않음** — 마이크·환경 다르면 비교 불가
- **집에선 코드·튜닝만, 측정은 연구실** 에서
- 평가용 마이크는 추후 **AT2020USB+ 1대 공용** 구매 예정 (~17~21만)
- 임시 단계엔 추가 비용 0 — 노트북/이어폰 마이크로 충분

---

## 11. 진행 중 / 다음 우선순위

| | |
| --- | --- |
| **본 트랙 (02_formant)** | Phase B(Lobanov+LDA 92.9% 합성)의 **실제 다화자 검증**이 다음. `evaluation/speaker_independence_plan.md` 의 6~26명 모집 |
| 휴면 트랙 (03) | Whisper SSL 정확도 0.66→0.90 — 인코더 교체부터 |
| 중단 트랙 (04) | MFCC+CNN 비교 연구. 본 방향이 포먼트라 우선순위 낮음 |

---

## 12. 집 Claude 에게 지시 예시

> "위 문서 기반으로 집 컴퓨터에 환경 셋업해줘. Step 1 부터 Step 5 까지. 막히는 부분 있으면 그때그때 물어봐."

또는 더 좁게:

> "git clone 부터 02_formant 가상환경까지 만들고 작동 확인까지 해줘."

---

## 13. 추가 참고 자료

- 팀 협업 가이드 전문: 노션에 별도 정리됨 (필요시 jaewon 에게 링크 요청)
- 본인 메일: `jwh2700@gmail.com`
- 연구 키워드: 한국어 7모음 / formant / Praat Burg / Lobanov 정규화 / LDA / VTLN / Whisper SSL / MLP probe / speaker-disjoint split

---

**셋업 완료 후 매번 따라야 할 것은 §7 의 동기화 워크플로 하나뿐입니다.**
