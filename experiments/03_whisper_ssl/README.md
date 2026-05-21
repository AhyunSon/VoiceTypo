# 03 Whisper SSL + MLP 접근

작성자: jaewon
브랜치: jaewon
날짜:  2026-05-21

## 무엇 / 왜

Frozen Whisper-base 인코더 + 경량 MLP 프로브로 7모음을 화자 독립적으로 인식.
SSL 특징이 화자/채널/잡음 변이를 걸러내므로 처음 보는 화자에도 강함.
포먼트 대비 정확도 10~15p 우위 (아래 원본 문서 참고).

## 실험 로그   (위가 최신, 계속 누적 — 실패도 기록)

- 5/21  개인 브랜치로 정리 업로드 (initial upload)
- (세부 실험 내역은 아래 원본 문서 / HANDOFF.md 참고, 수치는 추후 추가)

## 상태

(작성자가 현재 상태를 적어주세요)

---

아래는 원본 프로젝트 문서입니다.

# voicetypo_new

Korean monophthong (7-vowel) classifier for a media-art installation.
Designed for speaker independence: anyone walks up to the mic, pronounces one
of 아 / 에 / 이 / 오 / 우 / 으 / 어, the system classifies in real time.

## Why this design

- **Speaker independence is the success criterion.** Not "works for me" —
  works for unseen speakers (any age, gender, accent), under gallery noise.
- **Frozen Whisper-base encoder + small MLP probe.** Self-supervised models
  pretrained on hundreds of thousands of hours produce features that strip
  away speaker/channel/noise variation. Whisper-base specifically (290 MB)
  was chosen over Wav2Vec2-XLSR-Korean (1.2 GB) for its noise robustness
  (Whisper trained on 680k h of varied web audio) and for being small enough
  to run real-time on CPU if needed.
- **Formants ruled out as the primary path.** Lobanov-style normalization
  needs multiple vowels per speaker; with one-shot strangers it falls back
  to weaker schemes and accuracy drops 10–15 points vs SSL probes. Kept only
  as auxiliary diagnostic logging.
- **Data pool: Zeroth-Korean (115 spk) + Common Voice Korean (CC0, has age /
  gender / accent metadata) + FLEURS-ko.** All commercial-safe.
  KsponSpeech and AI-Hub are gated for non-Korean residents; KSS is single-
  speaker. Pansori-TEDxKR is held out for unseen-speaker evaluation only
  (its NC license blocks shipping but not internal eval).
- **Speaker-disjoint splits.** Train / val / test never share a speaker.
  This is the load-bearing decision for a public installation.
- **Augmentation: MUSAN noise, RIR reverb, pitch ±4 st (aggressive — covers
  children), time-stretch 0.9–1.1, SpecAugment.**
- **Success bar:** top-1 ≥ 0.90 on the Pansori unseen-speaker test set,
  per-vowel F1 ≥ 0.85.

### Known weakness
Korean public corpora are adult-skewed. Pitch augmentation partially covers
children but not perfectly. Plan: collect ~30 children's vowels at install
site for a final fine-tune pass.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# CPU-only PyTorch (always works):
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
# OR GPU PyTorch — needs MS Visual C++ Redistributable 2015-2022 (x64) installed
# system-wide; otherwise you'll get "caffe2_nvrtc.dll" load errors:
#   pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio

pip install -r requirements.txt

# verify everything wires up
python tests/test_logic.py
python tests/test_pipeline.py        # downloads Whisper-base (~290 MB)

python scripts/01_prepare_data.py    # ~10 GB download + alignment + extraction
python scripts/02_train.py           # CPU: hours / RTX 3070: minutes
python scripts/02b_evaluate.py       # in-corpus + Pansori unseen-speaker
python scripts/03_run_realtime.py    # live mic demo
```

If 03_run_realtime.py picks the wrong mic, list devices:
`python scripts/03_run_realtime.py --list` then `--device <idx>`.

The first run downloads:
- Zeroth-Korean (~10 GB) into `data/raw/zeroth/`
- Common Voice Korean via HuggingFace (a few hundred MB)
- FLEURS-ko via HuggingFace (~1 GB)
- Pansori-TEDxKR (~1 GB; eval only, optional)
- Wav2Vec2-XLSR-Korean (alignment, ~1.2 GB; first use only)
- Whisper-base (~290 MB; first use only)

MUSAN noise (for augmentation) is optional; if absent the loader generates
synthetic noise instead.

## Project layout

```
voicetypo_new/
  config.yaml
  voicetypo/
    audio_io.py          # mic, VAD, WAV I/O
    augment.py           # noise / RIR / pitch / stretch / SpecAugment
    features.py          # Whisper encoder, mean+std pooling
    model.py             # MLP probe
    train.py             # training loop
    evaluate.py          # held-out unseen-speaker metrics
    infer_realtime.py    # live mic CLI
    data/
      sources.py         # downloaders for each corpus
      align.py           # Wav2Vec2 CTC alignment
      extract_vowels.py  # jamo decomposition + segment extraction
      dataset.py         # PyTorch Dataset, speaker-disjoint splits
  scripts/
    01_prepare_data.py
    02_train.py
    03_run_realtime.py
  data/                  # gitignored
    raw/  processed/  checkpoints/
```
