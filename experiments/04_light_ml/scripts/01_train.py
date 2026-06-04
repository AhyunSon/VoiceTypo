"""Entry-point: train the small CNN.

Usage:
    python scripts/01_train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light.train import main

if __name__ == "__main__":
    main()
