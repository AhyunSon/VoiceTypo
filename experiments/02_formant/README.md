# 02 포먼트 기반 7모음 인식

작성자: jaewon
브랜치: jaewon
기간:  2026-04-27 ~ 진행중

## 무엇 / 왜

포먼트(F1/F2/F3)만으로 한국어 7모음(아/에/이/오/우/으/어)을 실시간 인식.
미디어아트 설치 특성상 학습된 표현(DNN) 대신 **연속적인 포먼트 값**이 필요(실시간 모핑용)해서,
포먼트 접근만으로 어디까지 갈 수 있는지를 끝까지 파고든 실험 라인.

핵심 질문: **포먼트 단독으로 다화자 환경 90% 인식이 가능한가?**

## 한눈에 — 핵심 결과

| 단계 | 방식 | 정확도 | 데이터 |
| --- | --- | --- | --- |
| 이전 시스템 | 앙상블+Kalman+wav2vec K-NN+F0 scale | 22.9% | 1인 오프라인 |
| cleanup 후 baseline | Praat Burg 단독 + Bark Mahalanobis | 54.3% (19/35) | 1인 오프라인 |
| Phase A (cal-free) | + VTLN 정규화 | 천장 54.3% | 1인 / 합성 다화자 |
| 캘리브레이션 라인 | cal + vote + GMM + bandwidth | 85.7% | 1인, take-split |
| Phase B (학습) | Lobanov 정규화 + LDA + 시간동역학 | 92.9% | 합성 다화자 / CV |
| 라이브 다화자 | cal + GMM 실사용 | 미흡 | 실제 다인 마이크 |

→ **결론:** cal-free 휴리스틱은 54.3%가 천장. 학습 기반(Phase B)은 합성 데이터에서 92.9%까지 도달 —
포먼트 단독 90%의 가능성은 보였으나, **실제 다화자 라이브 검증은 아직 미통과**.

## 실험 로그 (최신 → 과거, 실패도 기록)

### Phase B — 학습 기반 정규화 (2026-05~)
- Lobanov z-score 정규화 + LDA 학습 분류기 + 시간동역학(formant slope) → **92.9%**
  (본인 within-speaker 5-fold CV / 합성 다화자 LOSO 기준)
- 휴리스틱 거리매칭(Mahalanobis) 대신 데이터에서 결정경계를 학습하니 천장이 크게 오름
- 한계: 합성 다화자 데이터라 **실제 다화자 검증 필요** (상세: `evaluation/phase_b_v1_lobanov_lda.py`)

### 라이브 다화자 테스트 (2026-05-06)
- 실제 여러 사람 마이크 테스트 → cal + GMM 조합도 미흡
- 포먼트 추출 자체의 화자간 변동 + 어/오/우 후설군 중첩이 실사용에서 그대로 드러남

### Phase A — cal-free 정규화 (2026-04-29 ~)
- A1 VTLN: 가상 남성 48.6% → 54.3% 회복(+5.7%p). 본인(canonical) 화자엔 no-op
- A2 multi-prototype: 효과 미미 → **폐기**
- A3 F3-ratio(F1/F3, F2/F3): 효과 부정/미미 → **폐기**
- A 통합: VTLN speaker만 채택. **cal-free 천장 54.3% 확정** → 90%엔 cal 또는 학습 필요

### 캘리브레이션 라인 v1 ~ v5 (2026-04 ~ 2026-05)
- v1 단일 발화 cal: −3.6%p → **실패**
- v2 cal + 다중청크 vote (Mahalanobis): 71.4%
- v3 + GMM (모음별 k=1/2 BIC 자동, 어/오/우 bimodal 처리): 78.6%
- v4 + 확장 feature (bandwidth B1/B2/B3, F3 가중치): 85.7% (14-wav 평가)
- v5 5-fold leave-one-take-out CV로 검증 (작은 표본 노이즈 보정)

### 기반 정리 — cleanup 3단계 (2026-04-28)
- 죽은 코드 사슬 제거: `mfcc_svm`, `calibration_dialog`, `*_ORIG` 백업
- 복잡한 우회 로직 제거: 앙상블 / Kalman / wav2vec K-NN / F0 scale
- 16 → 9 파일, ~3000 → ~1700 줄. **22.9% → 54.3%** (복잡도가 오히려 정확도를 깎고 있었음)

### 진단 (2026-04-27)
- 시스템 진단: 죽은 코드, _REFS vs 학계 수치 불일치 확인
- bug_hunt: 포먼트 추출값이 학계 대비 6/7 모음에서 ≥100Hz 어긋남
  → "포먼트 추출 정확도 자체가 낮다"는 근본 한계 확인

## 시도한 기법 전체

| 분류 | 기법 | 결과 |
| --- | --- | --- |
| 정규화 | VTLN | 채택 (가상 화자 회복) |
| 정규화 | Lobanov z-score | 채택 (Phase B, 92.9%) |
| 정규화 | multi-prototype, F3-ratio | 폐기 |
| 분류기 | Bark Mahalanobis | baseline |
| 분류기 | GMM (bimodal 모음 대응) | 채택 (cal 라인 +6.4%p) |
| 분류기 | LDA 학습 분류기 | 채택 (Phase B) |
| feature | F3, bandwidth, 시간동역학 | 채택 |
| 추출 | F0-aware F1 (harmonic 오인 방지) | 검증 (`phase1_f0_aware.py`) |
| 보정 | self-reference 캘리브레이션 | 부분 채택 (단일발화 X, 다중발화+vote O) |

## 폴더 안내

상세 기록은 `evaluation/` 에 있음:
- `diagnostic_report.md` — 시스템 진단 (죽은 코드, _REFS 비교)
- `cleanup_log.md` — cleanup 3단계 실행 로그
- `formant_19_methods_analysis.md` — 포먼트 90% 가능성, 19개 기법 분석
- `speaker_independence_plan.md` — 다화자 검증 계획
- `refs_validation.md` — 학계 _REFS 출처 검증
- `bug_hunt.md` — 포먼트 추출 정확도 진단
- `phase_a_*.py` / `phase_b_*.py` / `calibration_v*.py` — 각 실험 스크립트

## 상태

진행중 — Phase B(92.9%)는 합성 데이터 결과. **다음: 실제 다화자 라이브 검증**으로
포먼트 단독 90%가 실사용에서도 성립하는지 확인.
