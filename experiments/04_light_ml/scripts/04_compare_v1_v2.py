"""Side-by-side comparison: v1 vs v2 eval results.

Reads results/v1/eval.json and results/v2/eval.json, prints overall + per-class
F1 deltas with focus on the back vowels (오/우/으) where v1 was weakest.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light import RESULTS_DIR, VOWEL_CLASSES, DISPLAY


def load_or_die(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    v1 = load_or_die(RESULTS_DIR / "v1" / "eval.json")
    v2 = load_or_die(RESULTS_DIR / "v2" / "eval.json")

    print("=== v1 vs v2 (test set, speaker-disjoint) ===")
    print(f"{'metric':<22} {'v1':>10} {'v2':>10} {'Δ':>10}")
    rows = [
        ("top-1 accuracy", v1["top1"], v2["top1"]),
        ("macro F1",       v1["macro_f1"], v2["macro_f1"]),
        ("model size (MB)", v1["model"]["on_disk_mb"], v2["model"]["on_disk_mb"]),
        ("params",          v1["model"]["params"], v2["model"]["params"]),
        ("CPU latency (ms)", v1["latency_cpu_ms_batch1"]["median"],
                              v2["latency_cpu_ms_batch1"]["median"]),
    ]
    for name, a, b in rows:
        if isinstance(a, float):
            print(f"{name:<22} {a:>10.4f} {b:>10.4f} {b-a:>+10.4f}")
        else:
            print(f"{name:<22} {a:>10,} {b:>10,} {b-a:>+10,}")

    print("\nper-class F1:")
    print(f"{'vowel':<6} {'v1 F1':>8} {'v2 F1':>8} {'Δ':>8}   support(v1/v2)")
    for c in VOWEL_CLASSES:
        f1a = v1["per_class"][c]["f1"]
        f1b = v2["per_class"][c]["f1"]
        sa = v1["per_class"][c]["support"]
        sb = v2["per_class"][c]["support"]
        marker = " <-- back vowel" if c in ("o", "u", "eu") else ""
        print(f"  {c:>2s} {DISPLAY[c]} {f1a:>8.3f} {f1b:>8.3f} {f1b-f1a:>+8.3f}   {sa}/{sb}{marker}")


if __name__ == "__main__":
    main()
