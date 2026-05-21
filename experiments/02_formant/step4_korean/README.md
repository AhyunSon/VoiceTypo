# Step 4 — Zeroth + MFA 다화자 LDA 학습

## 파이프라인

```
Zeroth (105 화자, 51h, CC BY 4.0)
   ↓ prep_zeroth.py             (MFA corpus 형식 변환)
zeroth_mfa/
   ↓ MFA align (korean_mfa 모델)  (HMM forced alignment)
zeroth_mfa_aligned/  (TextGrid: 음소별 시간)
   ↓ extract_vowel_features.py   (모음 시점 F1/F2/F3, 9D)
vowel_features.npz
   ↓ train_lda.py                (화자별 Lobanov + LDA + speaker-out CV)
lda_korean_multispeaker.pkl
   ↓ ui_window.py 자동 로드
실시간 분류 (cal-free)
```

## 사용 도구

| 도구 | 역할 | 라이선스 |
|---|---|---|
| Zeroth | 한국어 음성 코퍼스 (105 화자) | CC BY 4.0 |
| MFA 3.3.9 | Forced alignment | MIT |
| Korean MFA models | acoustic + dictionary + g2p | MIT |
| parselmouth | F1/F2/F3 추출 (Praat) | GPL-3 |
| scikit-learn | LDA | BSD-3 |
| python-mecab-ko | 한국어 tokenizer | Apache 2.0 |

## 실행 순서

```bash
# 1. Zeroth → MFA 형식 변환
python step4_korean/prep_zeroth.py

# 2. MFA align (conda env "mfa")
conda run -n mfa mfa align step4_korean/zeroth_mfa korean_mfa korean_mfa \
    step4_korean/zeroth_mfa_aligned --clean -j 4

# 3. 모음 feature 추출
python step4_korean/extract_vowel_features.py \
    --aligned_dir step4_korean/zeroth_mfa_aligned \
    --corpus_dir step4_korean/zeroth_mfa \
    --out step4_korean/vowel_features.npz

# 4. LDA 학습 + 검증
python step4_korean/train_lda.py
# → lda_korean_multispeaker.pkl 생성
# → ui_window 시작 시 자동 로드

# 5. 본인 라이브 검증
python main.py
```

## 핵심 학습 알고리즘

**훈련 시:**
- 각 화자의 raw F1/F2/F3 으로 화자별 (mean, std) 계산
- 각 화자 본인 stats 로 z-score 정규화 (Lobanov)
- 정규화된 9D feature 에 LDA 학습

**추론 시 (cal-free):**
- 모든 학습 화자의 (mean, std) 의 평균 = grand_mean / grand_std
- 입력 9D feature 에 grand stats 로 정규화
- LDA predict + predict_proba

## IPA → 한글 매핑

MFA korean_mfa 모델 출력:
| IPA | 한글 |
|---|---|
| ɐ | ㅏ |
| e, eː, ɛː | ㅔ |
| i, iː | ㅣ |
| o, oː | ㅗ |
| u, uː | ㅜ |
| ɨ, ɨː | ㅡ |
| ʌ, ʌː | ㅓ |

## 정확도

| 데이터 | 화자 | speaker-out CV |
|---|---|---|
| 10 화자 검증 | 10 | **74.8%** |
| 전체 Zeroth | 105 | (학습 후 측정) |

학계 _REFS Mahalanobis 베이스라인: ~50% (cal-free).

## 되돌리기

cal-free 동작 유지 — 시스템 영향 없음.
ui_window 의 LDA 자동 로드 = `lda_korean_multispeaker.pkl` 있을 때만 활성.
파일 삭제 → 학계 _REFS 휴리스틱 자동 복귀.

## 라이선스 경고 ⚠

- **parselmouth (GPL-3)**: 작품 배포 시 소스 공개 의무
- **mecab-ko**: 학계 free, 상업도 OK
- 작품 발표 형태 (학계 / 갤러리 / 판매) 따라 재점검 필요
