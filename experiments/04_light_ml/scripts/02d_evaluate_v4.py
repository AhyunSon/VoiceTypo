"""Entry-point: evaluate v4 (Whisper-tiny + MLP)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light.evaluate_v4 import main

if __name__ == "__main__":
    main(version="v4")
