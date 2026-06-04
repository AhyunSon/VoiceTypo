"""Entry-point: evaluate the trained small CNN on the held-out test split.

Usage:
    python scripts/02_evaluate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light.evaluate import main

if __name__ == "__main__":
    main()
