"""Entry-point: train v3 with log-mel input + DeepVowelCNN (residual).

Differences from v2:
  - feature       MFCC          -> log-mel filterbank (n_mels=64, dB-scaled)
  - model         SmallVowelCNN -> DeepVowelCNN (1 stem + 5 residual blocks)
  - base_channels 64            -> 48 (kept ~5 MB given the deeper stack)

Same as v2:
  - target_frames=24, aug_passes=2 (pitch / stretch / gain / noise),
    speaker-disjoint split (seed=17, val=0.10, test=0.15)

Usage:
    python scripts/01c_train_v3.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light.train import main

if __name__ == "__main__":
    main(
        version="v3",
        feature_kind="logmel",
        model_kind="deep",
        base_channels=48,
        target_frames=24,
        aug_passes=2,
        epochs=40,
        early_stop_patience=10,
    )
