# 03 Whisper SSL + MLP 모음 인식

작성자: jaewon
브랜치: jaewon
기간:  ~2026-04-29 (베이스라인 한 사이클 완주)

## 무엇 / 왜

Whisper SSL 인코더 + 경량 MLP 프로브로 한국어 7모음(아/에/이/오/우/으/어)을
**화자 독립적으로** 인식. 포먼트 접근(02번)과 정반대 전략 —
직접 만든 규칙 대신, 대규모로 사전학습된 표현(SSL)을 그대로 활용.

핵심 질문: **처음 보는 화자(미관찰 화자)에게도 동작하는가?**

## 왜 이 접근인가 (설계 의도)

| 결정 | 이유 |
| --- | --- |
| frozen Whisper-base 인코더 + MLP | 수십만 시간 사전학습된 SSL 특징이 화자/채널/잡음 변이를 걸러냄. 인코더는 학습 안 하고 분류 헤드(MLP)만 학습 |
| layer-8 은닉벡터 사용 (CTC 아님) | Whisper CTC 자모확률은 고립 모음에서 "아" 편향이 심함. 중간 layer 은닉벡터가 음소를 직접 인코딩 |
| Whisper-base(290MB) 선택 | Wav2Vec2-XLSR-Korean(1.2GB)보다 작고, 다양한 웹 오디오 학습으로 잡음에 강함 |
| **speaker-disjoint split** | train/val/test 가 화자를 절대 공유하지 않음 — "안 망하는" 핵심 결정 |
| Pansori-TEDxKR = 평가 전용 | 학습에 한 번도 안 쓴 미관찰 화자로 진짜 일반화 측정 |
| 증강 (MUSAN/RIR/pitch±4반음/tempo/SpecAugment) | 잡음·화자 다양성 폭 확대 |

## 한눈에 — 핵심 결과 (2026-04-29 베이스라인)

| 측정 | 정확도 |
| --- | --- |
| val (held-out) | 0.684 |
| in-corpus 미관찰 화자 | 0.662 |
| Pansori 미관찰 화자 (top-1) | 0.660 |
| 목표 | 0.90 (→ 0.24 갭) |

→ **결론:** 화자 일반화는 성공(in-corpus ≈ unseen, 도메인 갭 거의 0).
그러나 정확도가 **~0.66 에서 막힘** — 목표 0.90 까지 크게 부족.

## 핵심 발견

1. **화자 일반화 우수.** in-corpus 0.662 ≈ Pansori 미관찰 0.660 — 도메인 갭 사실상 0.
   처음 보는 화자에게도 그대로 동작. (1순위 목표였고 달성)
2. **모델 한계가 ~0.66.** train_loss 는 계속 내려가는데 val_acc 는 정체 —
   Whisper-base 임베딩이 한국어 7모음을 충분히 분리하지 못함.
3. **약점 모음 우/으/오.** 후설·원순 모음이 임베딩 공간에서 가깝게 뭉침.
   강점은 아/에/이(전설·저모음). — 포먼트 접근(02)의 어/오/우 문제와 같은 양상.

## 모음별 성능 (Pansori 미관찰 화자, F1)

| 모음 | F1 | precision | recall |
| --- | --- | --- | --- |
| 아 (a) | 0.834 | 0.879 | 0.793 |
| 에 (e) | 0.690 | 0.622 | 0.774 |
| 이 (i) | 0.680 | 0.844 | 0.570 |
| 오 (o) | 0.615 | 0.621 | 0.610 |
| 어 (eo) | 0.577 | 0.560 | 0.595 |
| 으 (eu) | 0.410 | 0.335 | 0.527 |
| 우 (u) | 0.351 | 0.339 | 0.363 |

macro F1 = 0.594. 아(0.834)와 우(0.351)의 격차가 큼.

## 정확도 측정 기준

- **speaker-disjoint:** train/val/test 가 화자를 공유하지 않음.
- **평가:** 학습에 안 쓴 미관찰 화자셋 — Pansori-TEDxKR (41 화자, 14700 세그먼트).
- **데이터:** Zeroth-Korean 28000 샘플 / 104 화자 / 클래스당 4000개로 균형.
  split = train 21719(78화자) / val 1927(14화자) / test 4354(12화자).
- **성공 기준:** Pansori top-1 ≥ 0.90, 모음별 F1 ≥ 0.85.

## 실험 로그 (최신 → 과거)

### 베이스라인 한 사이클 완주 (2026-04-29)
- 데이터 prep → Whisper-base frozen + MLP 학습 → 미관찰 화자 평가, **전 과정 완주**
- 구조: Whisper-base 인코더 frozen → mean+std pooling → MLP(hidden 256, dropout 0.3)
- 학습: AdamW lr 1e-3, batch 256, cosine schedule, class-weighted CE, 증강 2 패스
- 결과: best val_acc 0.6839 (epoch 8), early stop epoch 14 / in-corpus 0.662 / Pansori 0.660

### 버그 수정 — Pansori 로더
- `iter_pansori` 가 디렉토리 구조를 오인식 → 평가 세그먼트 0개 매칭
- 실제 구조에 맞춰 `trans.txt` 단위 순회로 수정 → 14700 세그먼트 정상 평가

### 환경 구축
- GPU torch(cu126) 설치, RTX 3070 인식 — 정렬 속도 ~10 utt/s (CPU 대비 시간→분 단위)

## 한계

1. **정확도 천장 ~0.66.** Whisper-base 임베딩 표현력 한계로 7모음 분리가 부족.
   분류 헤드(MLP)를 키워도 인코더 표현이 천장이면 한계.
2. **후설·원순 모음(우/으/오) 임베딩 중첩.** 포먼트 접근의 어/오/우 문제와 동일한 벽.
3. **아동 약점.** 한국어 공개 코퍼스가 성인 위주 — 피치 증강으로 일부 커버하나 불완전.
   설치 현장에서 아동 음성 실측 필요.
4. **mean+std pooling 이 시간 정보를 버림** — 모음 내 조음 변화를 활용 못 함.

## 앞으로 할 일 (정확도 0.66 → 0.90, 예상 수확 큰 순)

1. **더 큰 Whisper 인코더** — whisper-small(244M) / medium(769M). `config.yaml` 한 줄.
   표현력 한계가 원인이라면 가장 직접적인 해결.
2. **Whisper 중간 레이어 사용** — layer −1 → −3/−5. 음성학 정보가 마지막보다
   중간 레이어에 많다는 probing 문헌 다수.
3. **데이터 풀 확장** — Common Voice ko + FLEURS-ko 추가. 화자 다양성 확대.
4. **모음 코어 윈도우 좁히기** — 현재 30~90% → 50~80%. 조음 안정 구간만 남김.
5. **MLP 확장 / 어텐션 풀링** — mean+std 의 시간 정보 손실 보완.

> 원칙: **변수는 한 번에 하나만** 바꿔서 원인 추적 가능하게.

## 코드 구성

**핵심 모듈** (`voicetypo/`)
| 파일 | 역할 |
| --- | --- |
| `audio_io.py` | 마이크 / VAD / WAV 입출력 |
| `augment.py` | 증강 (noise / RIR / pitch / tempo / SpecAugment) |
| `features.py` | Whisper 인코더 특징 추출, mean+std pooling |
| `model.py` | MLP 프로브 |
| `train.py` | 학습 루프 |
| `evaluate.py` | 미관찰 화자 평가 |
| `infer_realtime.py` | 실시간 마이크 추론 |
| `data/sources.py` | 코퍼스 다운로더 (Zeroth / Pansori 등) |
| `data/align.py` | Wav2Vec2 CTC 정렬 |
| `data/extract_vowels.py` | 자모 분해 + 모음 구간 추출 |
| `data/dataset.py` | PyTorch Dataset, speaker-disjoint split |

**실행 스크립트** (`scripts/`)
| 스크립트 | 역할 |
| --- | --- |
| `01_prepare_data.py` | 데이터 다운로드 + 정렬 + 모음 추출 |
| `02_train.py` | 학습 |
| `02b_evaluate.py` | in-corpus + Pansori 미관찰 화자 평가 |
| `03_run_realtime.py` | 실시간 마이크 데모 |
| `04_live_test.py` / `05_calibrate.py` / `06_evaluate_wav_folder.py` | 라이브 테스트 · 캘리브레이션 · wav 폴더 평가 |

설정은 `config.yaml` 한 곳에서 관리 (인코더 모델·레이어·모음 윈도우 등).
자세한 진행 기록과 다음 세션 절차는 `HANDOFF.md` 참고.

## 상태

베이스라인 완주 — **화자 일반화는 검증됨**, 정확도 0.66.
다음 우선순위: 정확도를 0.90 으로 끌어올리기 (인코더 교체부터).
