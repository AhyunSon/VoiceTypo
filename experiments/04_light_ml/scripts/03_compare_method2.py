"""Compare method 3 results to method 2 — best-effort.

Loads voicetypo_light/results/eval.json and, if present, voicetypo_new's
matching results. Prints a side-by-side summary. If method 2 has not produced
its eval yet, only this method's numbers are shown.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light import RESULTS_DIR, VOWEL_CLASSES, DISPLAY


M2_CANDIDATES = [
    Path(r"C:\Users\admin\voicetypo_new\results\eval.json"),
    Path(r"C:\Users\admin\voicetypo_new\data\eval.json"),
    Path(r"C:\Users\admin\voicetypo_new\eval.json"),
]


def load_or_none(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    own = load_or_none(RESULTS_DIR / "eval.json")
    if own is None:
        print("voicetypo_light/results/eval.json missing — run scripts/02_evaluate.py first")
        return
    m2 = None
    for p in M2_CANDIDATES:
        m2 = load_or_none(p)
        if m2 is not None:
            print(f"[m2] using {p}")
            break

    print("\n=== voicetypo_light (Method 3 — MFCC + small CNN) ===")
    print(f"top-1: {own['top1']:.4f}   macro-F1: {own['macro_f1']:.4f}")
    print(f"size:  {own['model']['on_disk_mb']:.3f} MB   params: {own['model']['params']:,}")
    print(f"CPU latency (b=1): {own['latency_cpu_ms_batch1']['median']:.2f} ms median")
    print(f"per-class F1:")
    for c in VOWEL_CLASSES:
        f1 = own["per_class"][c]["f1"]
        print(f"   {c:>3s} {DISPLAY[c]} : {f1:.3f}")

    if m2 is None:
        print("\n[m2] no method-2 eval.json found — comparison skipped")
        return

    print("\n=== voicetypo_new (Method 2 — Whisper-base + MLP) ===")
    print(f"top-1: {m2.get('top1', float('nan')):.4f}   "
          f"macro-F1: {m2.get('macro_f1', float('nan')):.4f}")


if __name__ == "__main__":
    main()
