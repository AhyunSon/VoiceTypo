"""Run evaluation on the in-corpus held-out test set and on Pansori (unseen)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetypo.evaluate import main

if __name__ == "__main__":
    main()
