# voicetypo_new 핸드오프

마지막 업데이트: 2026-04-29 (베이스라인 학습 + 평가 완료)

## 한 줄 요약

전체 파이프라인(데이터 prep → Whisper 인코더 + MLP 프로브 → 미관찰 화자 평가)
한 사이클 완주. 화자 일반화는 잘 되는데(in-corpus 0.662 ≈ Pansori 0.660),
**전체 정확도가 0.66 근처에서 막힘 — 목표 0.90까지 0.24 갭.**
다음 세션의 핵심 작업은 정확도 끌어올리기.

## 베이스라인 결과 (2026-04-29)

### 데이터
- Zeroth-Korean train만 사용 (CV / FLEURS / Pansori는 학습에서 제외)
- 28000 샘플 / 104 화자 / 클래스당 4000개로 균형 — `data/processed/vowels/manifest.jsonl`
- 화자 분리 split: train=21719 (78 화자) / val=1927 (14 화자) / test=4354 (12 화자)

### 학습
- Whisper-base 인코더 frozen → mean+std pool → MLP(hidden=256, dropout=0.3)
- AdamW lr=1e-3, batch=256, cosine schedule, class-weighted CE, 증강 2 패스
- best **val_acc = 0.6839 (epoch 8)** — early stop epoch 14 (patience 6)
- 체크포인트: `data/checkpoints/probe.pt`

### In-corpus held-out 화자 (4354 샘플)
| 모음 | F1 | precision | recall | support |
|------|----:|----:|----:|----:|
| a   | **0.815** | 0.829 | 0.801 | 634 |
| e   | 0.766 | 0.732 | 0.803 | 691 |
| i   | 0.748 | 0.765 | 0.731 | 606 |
| eo  | 0.674 | 0.651 | 0.699 | 631 |
| o   | 0.575 | 0.646 | 0.519 | 594 |
| eu  | 0.554 | 0.503 | 0.617 | 611 |
| u   | **0.458** | 0.498 | 0.424 | 587 |

**accuracy = 0.662**, macro F1 = 0.656

### Pansori unseen-speaker (14700 세그먼트, 41 화자)
| 모음 | F1 | precision | recall | support |
|------|----:|----:|----:|----:|
| a   | **0.834** | 0.879 | 0.793 | 4676 |
| e   | 0.690 | 0.622 | 0.774 | 1854 |
| i   | 0.680 | 0.844 | 0.570 | 2567 |
| o   | 0.615 | 0.621 | 0.610 | 1378 |
| eo  | 0.577 | 0.560 | 0.595 | 2284 |
| eu  | 0.410 | 0.335 | 0.527 | 1178 |
| u   | **0.351** | 0.339 | 0.363 | 763 |

**top-1 = 0.660**, macro F1 = 0.594

### 핵심 관찰
1. **화자 일반화 우수**: in-corpus 0.662 ≈ Pansori 0.660. 도메인 갭 사실상 없음.
   미디어 아트에서 안 망하는 1순위 목표 달성.
2. **모델 한계가 0.66 근처**: train_loss는 계속 감소했지만 val_acc는 정체 →
   인코더 표현이 한국어 모음을 다 분리 못 함.
3. **약점 모음**: `u`(우) → `o`/`eu`와 혼동, `eu`(으) → `e`/`o`와 혼동,
   `o`(오) → `eo`/`u`와 혼동. **후설·원순 모음 클러스터가 임베딩에서 가깝게 잡힘.**
4. **강점 모음**: `a`, `e`, `i` — 전설/저모음은 안정적.

## 결정 (확정 — 흔들지 마라)

(이전 세션과 동일, 변동 없음)

- **인코더**: 사전훈련 Whisper-base 인코더(290 MB) frozen, 분류 헤드만 학습.
- **데이터**: Zeroth-Korean(105 train + 10 test 화자) + Common Voice ko + FLEURS-ko.
- **분할**: speaker-disjoint. **이게 미디어 아트에서 안 망하는 핵심.**
- **미관찰 화자 평가**: Pansori-TEDxKR (eval 전용).
- **증강**: MUSAN, RIR, 피치 ±4반음, 템포 0.9–1.1, SpecAugment.
- **성공 기준**: Pansori unseen top-1 ≥ 0.90, 7개 모음 F1 ≥ 0.85.

## 환경 (이번 세션에서 변경됨)

- **GPU torch 설치 완료**: cu126 휠. RTX 3070 인식 OK.
  ```
  torch 2.11.0+cu126
  cuda? True
  device count: 1
  name: NVIDIA GeForce RTX 3070
  ```
- 정렬 속도 ~10 utt/s (CPU 대비 시간 단위 → 분 단위).
- Pansori 다운로드 + 압축 해제 완료 (`data/raw/pansori/`).

## 이번 세션에서 발견·수정한 버그

1. **`iter_pansori` 디렉토리 구조 오인식** (`voicetypo/data/sources.py`).
   - 실제 구조: `pansori-tedxkr-corpus-1.0/<speaker>/<video_id>/<utt_id>.flac`
     + 같은 폴더에 `<speaker>-<video_id>.trans.txt` (Zeroth와 동일 포맷).
   - 초기 코드는 `<flac>.with_suffix(".txt")`를 찾아서 0개 매칭.
   - 수정: trans.txt 단위로 순회, 화자 ID = `trans_file.parent.parent.name`.
   - 첫 평가에서 `[pansori] no segments evaluated` → 수정 후 14700 세그먼트 정상.

## 다음 세션 액션 (우선순위 순)

### 1순위 — 정확도 끌어올리기
val_acc가 0.68에서 막힌다는 건 Whisper-base 임베딩으로는 한국어 모음 7개를
선형/얕은 MLP로 분리하기 어렵다는 뜻. 시도할 후보들 (예상 수확 큰 순):

1. **더 큰 Whisper 인코더** (`whisper-small` = 244M, `whisper-medium` = 769M).
   `config.yaml` `encoder.model_id` 한 줄만 바꾸면 됨. 사전계산은 GPU에서 ~3~10배
   느려지지만 한 번만 돌리면 됨. 표현력 한계가 원인이라면 가장 직접적인 해결.
2. **Whisper 중간 레이어 사용** (`encoder.layer: -1` → `-3` 또는 `-5`).
   ASR/probing 문헌에서 phonetic 정보는 마지막 layer보다 중간 layer에 더 많다는
   결과 다수. `voicetypo/features.py`의 layer 인덱싱 확인 필요.
3. **데이터 풀 확장** — Common Voice ko + FLEURS-ko 추가.
   `huggingface-cli login` 후 `--skip-cv` 빼고 prepare 재실행.
   화자 다양성 폭 넓혀서 후설 모음 분리에 도움될 수 있음.
4. **모음 코어 윈도우 좁히기** — 현재 30~90% (60% 폭). 50~80% (30% 폭)로 좁히면
   조음 안정 구간만 남아 분류기 수월. `config.yaml` `extraction.segment_lo/hi`.
5. **MLP 키우거나 소형 어텐션 풀링**으로 교체. 기존 mean+std는 시간 정보 버림.

### 2순위 — 화자 무관성 강화
지금도 in-corpus≈unseen이라 불필요할 수도. 하지만 멀쩡한 정확도(0.85+) 도달
시점에 다시 점검:
- 화자 정규화 (CMVN per utterance)
- 증강 강도 상향

### 3순위 — 라이브 데모
정확도 0.85+ 도달하면 `scripts/03_run_realtime.py`로 마이크 데모 검증.

## 다음 세션 시작 절차

### 0. 환경 활성화
```bash
cd /c/Users/admin/voicetypo_new
source .venv/Scripts/activate
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
# → True 가 나와야 정상
```

### 1. 가장 빠른 첫 실험: Whisper-small로 교체
```bash
# 1) config.yaml 한 줄만 수정
#    encoder.model_id: "openai/whisper-base"  →  "openai/whisper-small"
# 2) 사전계산 캐시 무효화 (사이즈 다르므로 필수)
rm -rf data/processed/features/
# 3) 재학습 (manifest는 그대로 — Whisper는 미세조정 안 하니 데이터 prep 재실행 불필요)
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u scripts/02_train.py
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u scripts/02b_evaluate.py
```

`whisper-small`은 ~967 MB, 첫 실행시 HF 캐시에 다운로드 (~2분).
사전계산은 base 대비 ~3배 느림 — GPU에서 5분 → 15분 정도 예상.

### 2. (대안) Whisper 중간 레이어 실험
`voicetypo/features.py` 보고 `encoder.layer = -1`이 어떻게 풀리는지 먼저 확인.
hidden_states 인덱스가 -1=마지막, 0=임베딩이라면 -3, -5 시도.
**먼저 features.py를 읽고 layer 옵션이 실제로 동작하는지 확인 필수.**

### 3. (병렬 가능) HF 토큰 등록 + Common Voice 추가
```bash
huggingface-cli login                  # 토큰: huggingface.co/settings/tokens
huggingface-cli whoami                 # 확인
# CV ko 약관 수락도 필요: huggingface.co/datasets/mozilla-foundation/common_voice_17_0
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u scripts/01_prepare_data.py --skip-pansori
```
주의: CV 추가하면 manifest 화자 수가 크게 늘어 학습 시간도 늘어남.
인코더 교체 실험과 동시에 하면 변수가 두 개라 원인 추적 어려움. **순서대로 한 번에 한 변수만.**

## 절대 잊지 마라 (반복)

- **화자 독립성 = 1순위.** 평가는 반드시 unseen-speaker 셋(Pansori).
  이번 세션에서 in-corpus와 unseen이 거의 같았으므로 검증됨 — 유지.
- **아동 약점**: 한국어 공개 코퍼스 성인 위주. 설치 현장 실측 필요.
- **변수 한 번에 하나만 바꾼다.** 인코더, 레이어, 데이터, 윈도우 동시 변경 금지.

## 파일 위치 빠른 참조

| 무엇 | 어디 |
|------|------|
| 설정 | `config.yaml` |
| 정렬 코드 | `voicetypo/data/align.py` |
| Zeroth 로더 | `voicetypo/data/sources.py:iter_zeroth` |
| Pansori 로더 (이번 세션 수정) | `voicetypo/data/sources.py:iter_pansori` |
| 추출 슬라이서 | `voicetypo/data/extract_vowels.py` |
| Whisper 특징 추출 | `voicetypo/features.py` |
| MLP 프로브 | `voicetypo/model.py` |
| 학습 진입점 | `scripts/02_train.py` |
| 평가 진입점 | `scripts/02b_evaluate.py` |
| 실시간 데모 | `scripts/03_run_realtime.py` |
| 베이스라인 manifest | `data/processed/vowels/manifest.jsonl` (28000 샘플) |
| 베이스라인 체크포인트 | `data/checkpoints/probe.pt` (val_acc=0.6839) |
| 베이스라인 학습 로그 | `data/train.log` |
| 베이스라인 평가 로그 | `data/eval2.log` |
