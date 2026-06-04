"""Entry-point: train v4 with frozen Whisper-tiny encoder + 2-layer MLP.

Usage:
    python scripts/01d_train_v4.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light.train_v4 import main

if __name__ == "__main__":
    main(version="v4", encoder_id="openai/whisper-tiny",
         hidden_dim=256, epochs=40, aug_passes=2, early_stop_patience=8)
