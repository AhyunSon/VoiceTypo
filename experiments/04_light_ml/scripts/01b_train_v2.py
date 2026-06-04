"""Entry-point: train v2 with bigger CNN + waveform-level augmentation.

Differences from v1:
  - base_channels  32 -> 64       (~1M params, ~4 MB on disk)
  - target_frames  32 -> 24       (less zero-padding for the median 96 ms vowel)
  - aug_passes      0 -> 2        (each train clip yields 3 MFCCs:
                                   original + 2 augmented copies)
  - waveform aug: pitch ±4 st (p=0.7), stretch 0.9–1.1 (p=0.5),
                  gain ±10 dB (always), Gaussian noise SNR 5–30 dB (p=0.7)

Usage:
    python scripts/01b_train_v2.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light.train import main

if __name__ == "__main__":
    main(
        version="v2",
        base_channels=64,
        target_frames=24,
        aug_passes=2,
        epochs=40,
        early_stop_patience=10,
    )
