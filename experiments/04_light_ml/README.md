# voicetypo_light — Method 3 (MFCC + small CNN)

Korean monophthong recognition (아/에/이/오/우/으/어). One of three parallel approaches:

| Method | Approach | Size | Status |
|---|---|---:|---|
| 1 | Formant (F1/F2) classifier | tiny | hit a wall at 54.3% offline / 37.1% live |
| 2 | Whisper-base encoder + MLP probe | 290 MB | training in `voicetypo_new` |
| 3 | **MFCC + small CNN (this project)** | **~1 MB** | this repo |

Method 3 tests whether a much smaller model can match method 2 on the same
speaker-disjoint split — i.e. is method 2 over-spec for a 7-class job.

## Data

Shared with method 2 — `voicetypo_new/data/processed/vowels/manifest.jsonl`
(read directly, not copied). 28 000 vowel segments, 104 speakers (Zeroth-Korean),
balanced 4 000 per class, median 96 ms / max 348 ms.

The speaker-disjoint split (`val_frac=0.10`, `test_frac=0.15`, `seed=17`) is the
exact same logic as `voicetypo_new/voicetypo/data/dataset.py`, so train / val /
test speakers match between methods 2 and 3.

## Layout

```
voicetypo_light/
├── voicetypo_light/
│   ├── __init__.py        # paths, classes, MFCC params, split params
│   ├── features.py        # MFCC + delta + delta-delta extractor
│   ├── dataset.py         # manifest loader + speaker-disjoint split + cached dataset
│   ├── model.py           # SmallVowelCNN (3-channel, 4 conv blocks, GAP, MLP head)
│   ├── train.py           # training pipeline
│   └── evaluate.py        # test-set metrics + CPU latency benchmark
├── scripts/
│   ├── 00_check_split.py  # sanity-check split (no training)
│   ├── 01_train.py        # train
│   └── 02_evaluate.py     # evaluate trained checkpoint
├── checkpoints/           # saved models
├── data/features/         # cached MFCC tensors per split
└── results/               # eval.json, confusion_matrix.csv, train.log
```

## Run

```bash
# Activate the existing venv from method 2 (CUDA 12.6, torch 2.11)
source /c/Users/admin/voicetypo_new/.venv/Scripts/activate
cd /c/Users/admin/voicetypo_light

PYTHONIOENCODING=utf-8 python scripts/00_check_split.py    # verify split

# v1 baseline (32 base channels, no waveform aug, 32 frames)
PYTHONIOENCODING=utf-8 python scripts/01_train.py
PYTHONIOENCODING=utf-8 python scripts/02_evaluate.py       # writes results/v1/eval.json

# v2 (64 base channels, waveform aug ×2, 24 frames)
PYTHONIOENCODING=utf-8 python scripts/01b_train_v2.py
PYTHONIOENCODING=utf-8 python scripts/02b_evaluate_v2.py   # writes results/v2/eval.json

# side-by-side
PYTHONIOENCODING=utf-8 python scripts/04_compare_v1_v2.py
```

## Model

`SmallVowelCNN` — input `(B, 3, 40, 32)`:
- Channels: MFCC, delta, delta-delta (40 cepstral coeffs each)
- Time: 32 frames @ 10 ms hop ≈ 320 ms (covers max-length segments)
- Body: 4× Conv2d(3×3) + BatchNorm + ReLU, 3 of them with MaxPool(2,2)
- Head: GAP → Linear(128→64) → Linear(64→7)
- ~250k parameters, ~1 MB on disk
