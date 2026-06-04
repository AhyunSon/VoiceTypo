"""voicetypo_light — MFCC + small CNN baseline for Korean monophthong recognition.

Method 3 in a 3-way comparison study:
  1) Formant (F1/F2) classifier   (Desktop/realtime_formant)
  2) Whisper-base encoder + MLP   (voicetypo_new)
  3) MFCC + small CNN             (this project)

Goal: same speaker-disjoint split / metrics as method 2, with a 1-5MB model.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT_MANIFEST = Path(r"C:\Users\admin\voicetypo_new\data\processed\vowels\manifest.jsonl")

CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
FEATURE_CACHE_DIR = DATA_DIR / "features"

# Match method 2 (voicetypo_new/config.yaml) so the comparison is apples-to-apples.
VOWEL_CLASSES = ["a", "e", "i", "o", "u", "eu", "eo"]
DISPLAY = {"a": "아", "e": "에", "i": "이", "o": "오", "u": "우", "eu": "으", "eo": "어"}

SAMPLE_RATE = 16000
SPLIT_VAL_FRAC = 0.10
SPLIT_TEST_FRAC = 0.15
SPLIT_SEED = 17

# Feature params — keep frame-step at 10 ms (160 samples @16k). 25 ms window.
MFCC_N_MFCC = 40
MFCC_N_FFT = 400
MFCC_HOP = 160
MFCC_N_MELS = 64

# v3 log-mel filterbank input
LOGMEL_N_MELS = 64
LOGMEL_N_FFT = 400
LOGMEL_HOP = 160

# Pad/truncate vowel segment to this many frames (~320 ms covers 99% of segments
# whose median length is 96 ms and max is 348 ms in the source manifest).
TARGET_FRAMES = 32
