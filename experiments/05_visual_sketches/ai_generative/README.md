# ai_generative — Memo Akten "생성형 글자" 레이어

## 무엇 / 왜
스터디로그 3번(Akten): "목소리가 획을 생성한다 / AI가 목소리를 어떻게 해석하는가."
원작 핵심은 **학습형 생성기**(Graves 2013 RMDN / LSTM 생성형 손글씨).

## 현재: 개념 프로토타입 (CPU, 학습 없음)
`asemic_strokes.py` — 같은 아이디어를 **절차적 규칙**으로 흉내냄.
말하면 펜이 흐르며 손글씨 같은 '의미 없는(asemic)' 획을 생성. 모음=손버릇, F1=높이, F2=곡률, Pitch=굵기.

```bash
python ai_generative/asemic_strokes.py   # (상위 폴더에서 실행, C 비움 · ESC 종료)
```

## 진짜 학습형으로 가려면 (연구실, GPU)
1. 데이터: (모음열/포먼트 시퀀스) → (펜 좌표 시퀀스) 쌍. 또는 손글씨 코퍼스(IAM-OnDB)로 RMDN 사전학습.
2. 모델: Graves 2013 식 **RMDN/LSTM**(Mixture Density Network) — PyTorch 로 신규 구현 권장.
   (Akten 의 ofxMSATensorFlow 예제는 TF r1.1 구버전 → 개념·구조만 참고, 그대로 빌드 X)
3. 입력 조건: 03_whisper_ssl 의 Whisper 임베딩을 조건으로 주면 "목소리 스타일→획"으로 확장 가능.
4. 추론만 집에서(경량), 학습은 연구실 GPU.

## 상태
개념 프로토타입 작성·검증 완료. 학습형은 미착수(GPU 필요).
