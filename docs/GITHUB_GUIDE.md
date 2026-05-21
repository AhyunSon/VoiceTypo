# VoiceTypo — GitHub 협업 가이드

**팀 공식 규칙** — 작업 시작 전 반드시 읽어 주세요.

최종 갱신: 2026-05-21

---

## 0. 절대 규칙 (대전제)

`main` 은 **팀 통합본 브랜치**입니다. 현재는 재구축을 위해 비어 있고(README 만),
앞으로 회의에서 채택된 기능을 깨끗하게 재구현해 쌓아 올립니다.
(이전 통합 프로젝트는 `ahyun` 브랜치 `experiments/01_voicetypo_integrated/` 에 보존돼 있습니다.)

> **반드시 지킬 것**
> 1. `main` 에 직접 push 하지 않는다 — feature 브랜치 + PR 로만 갱신
> 2. 개인 브랜치를 `main` 에 그대로 merge 하지 않는다 — 재구현으로만
> 3. 실험은 개인 브랜치, `main` 반영은 feature 브랜치 — 섞지 않는다

---

## 1. 브랜치 종류

이 3가지만 기억하면 됩니다.

| 종류 | 이름 예시 | 용도 | 직접 작업 |
| --- | --- | --- | --- |
| `main` | `main` | 팀 통합본 (재구축 중, 보호 브랜치) | 금지 — feature PR 로만 |
| 개인 브랜치 | `jaewon` (1인 1개) | 실험 — `experiments/` 폴더로 분리 (§3) | 자유 — main 에 안 감 |
| feature 브랜치 | `feature/formant-cal` | 회의 통과 후 main 재구현 | 작업 → PR → main |

> **핵심:** `main` 은 보호 · 실험은 개인 브랜치(폴더로 분리) · main 반영은 feature 브랜치. 셋을 섞지 않는다.
> 개인 브랜치는 **서브브랜치 안 만들고** 폴더로 실험 분리 — 자세히 §3.

---

## 2. 전체 흐름 (4단계)

| 단계 | 무엇 |
| --- | --- |
| 1. 평소 실험 | 개인 브랜치 |
| 2. 주간 회의 | 후보 선정 |
| 3. main 재구현 | feature → PR |
| 4. 학기말 | 정제본 |

### 1단계. 평소 — 개인 브랜치에서 실험

| 순서 | 명령 | 의미 |
| --- | --- | --- |
| 1 | `git checkout jaewon` → `git pull` | 내 브랜치 (이미 있음, 서브브랜치 X) |
| 2 | `experiments/01_formant_cal/` 폴더 + README 작성 | 실험 = 폴더로 분리 |
| 3 | `git add .` → `git commit -m "exp(formant): ..."` | 변경 저장 |
| 4 | `git push origin jaewon` | 내 브랜치에 올림 |

> **이 단계 규칙:** 서브브랜치 X · 실험은 `experiments/번호_주제/` 폴더 · 폴더마다 README · main 신경 X · PR X. (자세히 §3)

### 2단계. 매주 회의

| 할 일 | 내용 |
| --- | --- |
| 데모 | 각자 자기 브랜치 시연 |
| 후보 기록 | 회의록에 "main 재구현 후보" 체크박스 |

> **예시:** `jaewon` / `experiments/01_formant_cal` — cal 정확도 개선, main 반영 검토

### 3단계. 회의 후 — 담당자가 main 에 재구현

| 순서 | 명령 | 의미 |
| --- | --- | --- |
| 1 | `git checkout main` → `git pull` | 최신 main 받기 |
| 2 | `git checkout -b feature/formant-cal` | feature 브랜치 생성 |
| 3 | (채택 실험 보고 깔끔히 다시 구현) | merge 아님 — 새로 작성 → main 안 깨짐 |
| 4 | `git commit -m "feat(formant): ..." -m "Origin: jaewon/01_formant_cal"` | 출처 = Origin 푸터 (§4-4) |
| 5 | `git push -u origin feature/formant-cal` | feature 브랜치 올림 |
| 6 | PR → 팀 1명 확인 → merge → 브랜치 삭제 | main 반영 |

> **이 단계 규칙:** 출처는 `Origin:` 푸터로 명시 (§4-4). main 에서 작동 확인 후 merge.

### 4단계. 학기말

| 결과물 | 상태 |
| --- | --- |
| `main` | 깔끔하게 재구현된 최종 통합본 |
| 개인 브랜치들 | 각자의 탐색 과정 기록 (삭제 X — 연구 이력) |

---

## 3. 개인 브랜치 운영

**원칙: 1인 1브랜치 · 실험은 폴더로 분리 · 서브브랜치 안 만듦.**

### 3-1. 브랜치 (이미 생성됨)

| 형식 | 용도 | 예시 |
| --- | --- | --- |
| `<이름>` | 본인 실험 전부 (1개만) | `jaewon` |

대상: `ahyun` `eunbin` `jaewon` `jinhee` `seoeun` `sunmin`

> 서브브랜치(`jaewon/xxx`) 만들지 않는다. main 에서 따올 필요도 없다 (실험은 main 과 독립).

### 3-2. 실험 = 폴더로 분리

```
jaewon 브랜치 (1개)
└── experiments/
    ├── 01_formant_cal/
    │   ├── README.md
    │   └── (코드)
    ├── 02_whisper_dropout/
    └── 03_lobanov_norm/
```

폴더명 규칙

| 규칙 | 설명 |
| --- | --- |
| `번호_주제` | `01_formant_cal` — 번호=정렬, 주제=식별 |
| 소문자 + 언더바 | 띄어쓰기·한글·특수문자 금지 |
| 모호한 이름 금지 | `test`, `v2`, `new` 금지 |

### 3-3. 폴더마다 README (작성자 + 실험 로그)

```
# 01 포먼트 cal sanity check

작성자: jaewon
브랜치: jaewon
날짜:  2026-05-19

## 무엇 / 왜
cal 오염 제거로 정확도 향상 가설

## 실험 로그   (위가 최신, 계속 누적 — 실패도 기록)
- 5/19  sanity ±3σ → 55→61% (개선)
- 5/19  ±2σ 강화 → 58% (과함, ±3σ 복귀)
- 5/18  baseline → 55%

## 상태
진행중 — 다음: 라이브 검증
```

> 폴더만 봐도 **누가·언제·뭘·왜·결과·상태** 파악. 실패 기록도 자산.

### 3-4. 작업 흐름

| 순서 | 명령 |
| --- | --- |
| 1 | `git checkout jaewon` → `git pull` |
| 2 | `experiments/04_새실험/` 폴더 + README 작성 |
| 3 | `git add .` → `git commit -m "exp(formant): 임계값 0.3→0.5 테스트"` |
| 4 | `git push origin jaewon` |

> 브랜치 갈아타기 없음. 폴더만 추가.

### 3-5. 실험 현황 (노션 Database, 주 1회 갱신)

| 이름 | 실험 | 위치 | 목표 | 결과 | 상태 | 다음 | 갱신 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jaewon | 포먼트 cal | `jaewon`/`01_formant_cal` | 정확도↑ | 55→61% | 진행중 | 라이브 검증 | 5/19 |
| eunbin | whisper dropout | `eunbin`/`02_whisper_dropout` | 일반화 | 66→69% | 결과있음 | main후보 검토 | 5/19 |

상태 값

| 상태 | 뜻 | 다음 |
| --- | --- | --- |
| 진행중 | 실험 중 | 계속 |
| 결과있음 | 수치 나옴 | 회의서 main후보 판단 |
| main후보 | 회의 채택, 재구현 대기 | 담당자 배정 |
| 재구현완료 | feature 로 main 반영됨 | 종료 |
| 폐기 | 효과 없음 | 기록만 (삭제 X) |

> 노션 `/database` 표 보기로 생성 · "상태" = 선택 속성 · 자기 행은 자기가 주 1회 갱신.

### 3-6. feature 브랜치 (main 반영 전용 — 개인 브랜치와 별개)

| 형식 | 용도 | 예시 |
| --- | --- | --- |
| `feature/<내용>` | 새 기능 재구현 | `feature/formant-cal` |
| `fix/<내용>` | 버그 수정 | `fix/uo-confusion` |
| `docs/<내용>` | 문서 | `docs/github-guide` |

> 소문자 + 하이픈 · feature 브랜치는 **항상 최신 `main` 에서** 분기 (개인 브랜치에서 X)
> 실험 채택 → 담당자가 main 에서 feature 브랜치 → 깔끔히 재구현 → PR → main.
> 개인 브랜치는 절대 main 에 직접 merge 안 함.

---

## 4. 커밋 메시지 규칙

**Conventional Commits 표준.** type·scope 는 영어(업계 표준 키워드), 설명(description)은 한국어.

### 4-1. 형식

실제 커밋 예시:

```
feat(formant): cal 다이얼로그 추가
└┬─┘└───┬───┘  └────────┬────────┘
 type   scope        description
```

이걸 4부분으로 분해:

| 부분 | 무슨 뜻 | 위 예시 | 규칙 |
| --- | --- | --- | --- |
| type | 작업 종류 | `feat` | 영어 소문자 (4-2 참고) |
| (scope) | 어느 모듈 (생략 가능) | `(formant)` | **모듈명** (formant/whisper/cal). 숫자 X |
| `:` | 구분자 | `:` | 콜론 고정 (슬래시 `/` 는 브랜치 이름에만) |
| description | 한 줄 요약 | `cal 다이얼로그 추가` | **한국어 · 간결하게 · 명사형 끝맺음 · 마침표 X** |

> 설명 규칙: 한국어로 간결하게, **명사형으로 끝맺음**, 끝에 `.` 없음.
> 예) `... 추가`, `... 수정`, `... 개선`, `... 제거`, `... 테스트`

### 4-2. 유형(type)

| type | 언제 | 어디서 |
| --- | --- | --- |
| `exp` | 개인 실험·시도 (대부분 이거) | 개인 브랜치 |
| `feat` | 새 기능 (main 반영) | feature 브랜치 |
| `fix` | 버그 수정 | feature 브랜치 |
| `docs` | 문서만 | docs/feature 브랜치 |
| `refactor` | 동작 그대로 코드 정리 | feature 브랜치 |
| `chore` | 의존성·설정 등 잡일 | 아무 데나 |
| `test` | 테스트 코드 | 아무 데나 |

> `exp` 는 Conventional Commits 가 허용하는 **커스텀 유형** (스펙: feat/fix 외 다른 유형 사용 가능). 표준 위반 아님.
> 핵심: 개인 브랜치 실험 = `exp` / main 반영 = `feat`·`fix` 등.

### 4-3. 예시 (설명=한국어 · scope=모듈)

| 상황 | 커밋 메시지 |
| --- | --- |
| 실험 시작 | `exp(formant): cal sanity check 추가` |
| 값 바꿔 재실행 | `exp(formant): 임계값 0.3→0.5 테스트` |
| 결과 기록 | `exp(formant): sanity 적용해 정확도 55→61% 개선` |
| 실험 중단 | `exp(lobanov): 효과 없어 폐기` |
| main 재구현 | `feat(formant): cal 다이얼로그 추가` (+ 아래 4-4 푸터) |
| 버그 수정 | `fix(whisper): dropout NaN 수정` |
| 문서 | `docs: 협업 가이드 갱신` |

> 실험번호(01 등)는 **폴더명**(`experiments/01_formant_cal/`)에 있음. 커밋 scope 는 모듈명만.

### 4-4. 원본 출처 = 푸터 (표준 방식)

main 재구현 커밋: 제목 다음 **빈 줄**, 그 아래 `Origin:` 푸터.

```
feat(formant): cal 다이얼로그 추가

Origin: jaewon/experiments/01_formant_cal
```

git 명령 (`-m` 두 번 = 제목 / 빈줄 뒤 본문):

```
git commit -m "feat(formant): cal 다이얼로그 추가" -m "Origin: jaewon/experiments/01_formant_cal"
```

**왜 필요한지 — 예로 설명:**

| 상황 | 설명 |
| --- | --- |
| 평소 | jaewon 이 자기 브랜치 `experiments/01_formant_cal/` 에서 실험 |
| 채택 | 회의서 "이거 좋다 → main 넣자" |
| 재구현 | 담당자가 그 실험 **보고 새로 깔끔히 작성** (복붙 아님) |
| 문제 | 새로 작성이라 git 이 "이게 어디서 왔는지" 모름 |
| 해결 | 커밋에 `Origin: jaewon/01_formant_cal` 푸터 → 나중에 "이 코드 누구 실험서 왔지?" 추적 가능 |

→ 즉 `Origin:` 푸터 = **main 코드 ↔ 원래 실험 연결 꼬리표.** **재구현 커밋에만** 씀 (개인 실험 커밋엔 안 씀).

### 4-5. 언제 커밋하나

**한 가지 일 끝낼 때마다.** 몰아서 한 번에 X.

| 커밋한다 (O) | 안 한다 (X) |
| --- | --- |
| 값 하나 바꿔 돌려봤다 | 3일치 작업 한 커밋에 |
| 한 기능이 작동한다 | `fix`, `update` 만 (내용 없음) |
| 결과 수치 나왔다 (숫자 적기) | 안 돌아가는 코드 뭉텅이 |

### 4-6. 꼭 지킬 것

> 1. type = 영어 소문자 · description = 한국어, 간결하게 (팀 전체 통일)
> 2. scope = 모듈명 (숫자 X) · 실험번호는 폴더명에
> 3. `exp` = 개인 브랜치 / `feat`·`fix` = feature 브랜치
> 4. main 재구현 커밋엔 `Origin:` 푸터로 출처 명시

---

## 5. 금지 사항

| 금지 | 이유 |
| --- | --- |
| main 직접 push | PR·재구현으로만 갱신 |
| 개인 브랜치를 main 에 그대로 merge | 깨질 위험 · 재구현만 허용 |
| 실험을 feature 브랜치에서 진행 | feature 는 main 반영 전용 |
| 다른 사람 코드·브랜치 삭제 | 탐색 이력 보존 |
| 음성·모델·대용량 파일 커밋 | 저장소 비대화 (.gitignore 준수) |

---

## 6. 데이터 / .gitignore

음성 코퍼스 · 학습 데이터 · 모델 가중치 · 가상환경 · 캐시는 **git 에 올리지 않는다.**

| 분류 | 제외 패턴 |
| --- | --- |
| 음성 | `*.wav`, `*.flac`, `*.mp3` |
| 모델·데이터 | `*.npz`, `*.pt`, `*.pkl` |
| 폴더 | `data/`, `results/`, `external_data/`, `.venv/`, `__pycache__/` |

> 대용량은 로컬에서 별도 관리 · 커밋 전 `git status` 로 큰 파일 섞임 확인

---

## 7. 신규 참여자 온보딩

| 순서 | 할 일 |
| --- | --- |
| 1 | 저장소 collaborator 초대 수락 |
| 2 | 이 문서 + 프로젝트 README 정독 (30분) |
| 3 | 환경 설정 (가상환경 + requirements 설치) |
| 4 | 자기 브랜치(이미 있음)에서 `experiments/` 폴더로 작은 실험 (§3) |
| 5 | 매주 회의에서 진행 공유 |

---

## 8. 용어 정리 (입문자용)

| 용어 | 뜻 |
| --- | --- |
| 브랜치 | 작업을 분리하는 평행 줄기. main 을 건드리지 않고 작업하는 공간 |
| 커밋 | 변경 사항 한 묶음을 기록하는 단위 |
| push | 로컬 커밋을 GitHub 에 올리는 것 |
| pull | 원격의 최신 변경을 로컬로 받는 것 |
| PR (Pull Request) | "이 브랜치를 main 에 합쳐 주세요" 요청. 리뷰가 이뤄지는 곳 |
| merge | 리뷰 통과한 브랜치를 합치는 것 |
| 재구현 | 실험을 그대로 합치지 않고, main 구조에 맞게 새로 깔끔히 작성하는 것 |

---

## 9. 한 장 요약

| 단계 | 무엇 | 결과 |
| --- | --- | --- |
| 1. 개인 브랜치 | 1인1브랜치 + `experiments/` 폴더 실험 · 자기 것만 push | 탐색 |
| 2. 주간 회의 | 데모 → "main 후보" 체크 | 후보 선정 |
| 3. 회의 후 | 담당자가 feature 브랜치로 재구현 → PR | main 반영 |
| 4. 학기말 | main = 정제본 / 개인 브랜치 = 탐색 기록 | 마무리 |

> **기억할 핵심 3가지**
> 1. 실험 = 개인 브랜치 / main 반영 = feature 브랜치 (재구현)
> 2. main 폴더 구조 절대 안 건드림 (정리 ≠ 폴더 이동)
> 3. main 은 merge 가 아니라 재구현으로만 갱신

---

# 부록. 향후 반영 가능한 개선 (선택)

> **현재 규칙이 아닙니다.** huggingface/transformers, cal.diy 등 잘 운영되는
> 저장소에서 참고한 항목으로, 필요해지면 팀 합의 후 도입할 수 있는 후보입니다.
> 지금 적용할 필요 없음 (연구팀은 규칙이 적을수록 잘 굴러갑니다).

### A. 강력 추천

| 항목 | 내용 | 참고 |
| --- | --- | --- |
| A-1. pre-commit | 커밋 시 Black+Ruff 자동 → 스타일 자동 통일. `.pre-commit-config.yaml` 1개 | huggingface, cal.diy |
| A-2. 재구현 후보 → Issue + 라벨 | 회의록 체크박스 → Issue + `main-candidate` 라벨. feature PR 에서 `Closes #12` 추적 | huggingface, cal.diy |
| A-3. `good first issue` 라벨 | 신입용 작은 작업에 라벨 → "뭐부터" 막막함 해결 | cal.diy |

### B. 중간 (있으면 좋음)

| 항목 | 내용 | 참고 |
| --- | --- | --- |
| B-1. PR 템플릿 | "무엇/원본 출처/테스트법" 양식 강제 | huggingface, cal.diy |
| B-2. 실험 재현 규율 | 개인 브랜치 README 에 "환경+실행법+결과" 1줄 | huggingface |
| B-3. CI lint | feature→main PR 에 Black/Ruff 자동 검사 | huggingface, cal.diy |

### C. 도입하지 않음 (우리 규모엔 과함)

| 항목 | 이유 |
| --- | --- |
| Fork 모델 | 외부 기여자 수천 명용. 우리는 신뢰 팀 → 공유 브랜치 |
| Turborepo / Kodiak / 릴리스 자동화 | JS SaaS 전용. 연구팀 과함 |
| E2E 테스트 / 시맨틱 버저닝 | 작품 연구라 불필요 |

### 도입 우선순위 (필요해질 때)

| 순위 | 항목 |
| --- | --- |
| 1 | pre-commit (스타일 자동 통일) |
| 2 | Issue + `main-candidate` / `good first issue` 라벨 |
| 3 | PR 템플릿 |
| 4 | 개인 브랜치 README 재현 규율 |
| 5 | CI lint (feature PR) |

> 참고: huggingface 는 "순수 AI 작성 PR 금지, 사람이 책임진다" 명시.
> 우리 워크플로의 "담당자가 LLM 과 함께 재구현" 과 같은 철학 — 우리 방식이
> 대형 ML 프로젝트 표준과 동일 방향임을 확인.

---

*이 문서 변경이 필요하면 `docs/github-guide` 브랜치로 PR 을 올려 팀 합의 후 갱신합니다.*
