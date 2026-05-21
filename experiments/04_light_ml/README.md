# 04 경량 ML 모음 인식 (MFCC+CNN / Whisper-tiny)

작성자: jaewon
브랜치: jaewon
기간:  2026-04 ~ (v1~v4 시도)

## 무엇 / 왜

한국어 7모음(아/에/이/오/우/으/어)을 **가벼운 모델(1~5MB)**로 인식.
3가지 모음 인식 접근의 비교 연구에서 **방법 3** 에 해당:

| 방법 | 접근 | 폴더 |
| --- | --- | --- |
| 1 | 포먼트 (F1/F2) | 02_formant |
| 2 | Whisper-base 인코더 + MLP | 03_whisper_ssl |
| 3 | **MFCC + 소형 CNN (이 폴더)** | 04_light_ml |

핵심 질문: **작은 모델로도 Whisper-base(방법 2) 수준 정확도가 나오는가?**
(임베디드·저연산 환경 적용 가능성 확인)

## 왜 이 접근인가 (설계 의도)

| 결정 | 이유 |
| --- | --- |
| 1~5MB 소형 모델 | 방법 2(Whisper-base 290MB)는 무거움. 저연산·임베디드 가능성 탐색 |
| 방법 2 와 **동일 manifest·split 재사용** | 같은 화자·같은 데이터로 평가 → 방법 2 와 사과 대 사과 비교 |
| MFCC + delta/delta-delta (3채널) | 전통적 음성 특징. 작은 CNN 입력으로 적합 |
| 평가에 CPU 추론 속도·모델 용량 포함 | "경량"이 목표라 정확도뿐 아니라 속도·크기도 측정 |

## 사용 데이터

방법 2(`03_whisper_ssl`)의 manifest 를 **그대로 재사용** — 같은 데이터로 비교하기 위함.

- 원천: Zeroth-Korean 모음 28000 샘플 (manifest 공유)
- split: speaker-disjoint, val 10% / test 15% / seed 17 — 방법 2 와 동일하게 맞춤
- 특징: MFCC 40차 (n_fft 400, hop 160 = 10ms step / 25ms window), 64 mel
- 모음 구간: 32 프레임(~320ms)으로 pad/truncate (원천 모음 중앙값 96ms, 최대 348ms)

## 버전별 시도

CNN 으로 시작해서, 막히자 경량 SSL 프로브로 전환한 흐름:

| 버전 | 모델 | 입력 | 핵심 변경 |
| --- | --- | --- | --- |
| v1 | SmallVowelCNN (32채널, 4 conv + GAP + MLP) | MFCC | 베이스라인 |
| v2 | SmallVowelCNN (64채널) | MFCC + 파형 증강 | 채널↑ + 증강으로 일반화 강화 |
| v3 | DeepVowelCNN (stem + residual 5블록) | log-mel | 더 깊은 모델 + 입력 변경 |
| v4 | frozen Whisper-tiny 인코더 + 2층 MLP probe | 파형 | **CNN 포기 → 경량 SSL 프로브로 전환** |

- **증강(v2~)**: pitch ±4반음, stretch 0.9~1.1, gain ±10dB, noise SNR 5~30dB.
  `aug_passes=2` → 학습 세트 3배 (원본 + 증강 2벌). val/test 는 깨끗하게 유지.
- **v4**: 방법 2 와 같은 SSL 접근이되 인코더를 Whisper-**tiny**(소형)로 — 경량 목표 유지.
  인코더는 frozen, MLP probe 만 학습.

## 평가 방식

- test **Top-1 정확도** + 모음별 precision/recall/F1 + 7×7 혼동 행렬
- **CPU 추론 속도** (ms/sample, 단일 스레드) — 경량 목표라 속도도 핵심 지표
- **모델 용량** (디스크 / 메모리)
- v4 는 추론 속도를 probe-only / end-to-end(인코더 포함) 로 분리 측정

## 결과

> **이 폴더에는 결과 파일이 없습니다.** 결과는 실행 시 `results/` 에 생성되는데,
> `results/` 는 `.gitignore` 대상이라 커밋되지 않았습니다.

알려진 단서 — 02_formant 의 방법 비교 메모(`recognition_methods_comparison.md`,
2026-04-29) 에 **"방법 3 CNN, v3 학습 중 약 65% 천장"** 으로 기록됨.

→ v1/v2/v4 의 정확한 수치는 **재실행 또는 작성자 확인 필요**
   (`python -m evaluate` / `evaluate_v4` 로 재생성 가능).

## 한계

1. **결과 미보존** — 버전별 정확도·속도·용량 수치가 repo 에 없음. 재현 필요.
2. **CNN 천장 의심** — 비교 메모 기준 v3 가 ~65%. CNN 표현력 또는 데이터셋 한계로 추정.
3. v4 로 전환한 것 자체가 "CNN 만으로는 부족했다"의 방증일 가능성.

## 앞으로 할 일

1. **v1~v4 재실행 → 결과 표 작성** — 정확도·CPU 속도·모델 용량을 한 표에 정리.
2. **방법 2(Whisper-base)와 직접 비교** — 같은 split 이므로 바로 비교 가능.
   "경량 모델이 얼마나 따라잡는가"가 이 실험의 결론.
3. v4(Whisper-tiny)가 CNN 들보다 나은지 확인 → 경량 접근의 최종 방향 결정.

## 코드 구성

| 파일 | 역할 |
| --- | --- |
| `__init__.py` | 설정값 (경로, split, MFCC/log-mel 파라미터) |
| `features.py` | MFCC 특징 추출 (delta/delta-delta 채널 포함) |
| `model.py` | CNN 모델 — SmallVowelCNN(v1/v2), DeepVowelCNN(v3) |
| `dataset.py` | manifest 로딩 + speaker-disjoint split (방법 2 와 동일 로직) |
| `augment.py` | 파형 증강 (pitch/stretch/gain/noise) — v2~ |
| `train.py` | v1~v3 학습 (MFCC/log-mel + CNN) |
| `train_v4.py` | v4 학습 (Whisper-tiny 인코더 + MLP probe) |
| `evaluate.py` | v1~v3 평가 (정확도·F1·혼동행렬·속도·용량) |
| `evaluate_v4.py` | v4 평가 |
| `live_eval.py` | 라이브 테스트 헬퍼 — v1~v4 통합 분류기 |

## 상태

v1~v4 코드 완성 — 단, **버전별 결과 수치는 미보존**. 현재 **중단된 상태**.

**중단 이유 (정황상 — 정확한 기록은 없음):**
이 프로젝트는 3가지 접근 비교 연구의 *방법 3*(탐색·비교용)이었음.
프로젝트의 실제 방향은 **포먼트 접근(02_formant)** — 실시간 모핑을 위해
연속적인 F1/F2 값이 필요하고(연구실 요구: DNN 학습 표현 비사용),
MFCC+CNN·Whisper-tiny 같은 분류 모델은 모음 라벨만 줄 뿐
모핑용 연속 포먼트를 주지 못함. 그래서 #3·#4 는 비교용 탐색으로 진행되다
본 방향(포먼트)에 우선순위가 밀려 중단된 것으로 보임.

다시 이으려면: v1~v4 재실행 → 결과 표 작성 → 방법 2 와 비교.
