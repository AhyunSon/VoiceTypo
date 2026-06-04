"""Three-way comparison: v1 (MFCC + SmallCNN) vs v2 (MFCC + bigger SmallCNN + aug)
vs v3 (log-mel + DeepCNN + aug).

Reads results/v1/eval.json, results/v2/eval.json, results/v3/eval.json and
prints overall + per-class F1 with the back-vowel page (오/우/으) flagged.
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
    v3 = load_or_die(RESULTS_DIR / "v3" / "eval.json")

    print("=== v1 / v2 / v3 (test set, speaker-disjoint, 16 unseen speakers) ===")
    print(f"{'metric':<22} {'v1':>10} {'v2':>10} {'v3':>10}   {'v3-v1':>8}")
    rows = [
        ("top-1 accuracy", v1["top1"], v2["top1"], v3["top1"]),
        ("macro F1",       v1["macro_f1"], v2["macro_f1"], v3["macro_f1"]),
        ("model size (MB)", v1["model"]["on_disk_mb"], v2["model"]["on_disk_mb"],
                            v3["model"]["on_disk_mb"]),
        ("params",          v1["model"]["params"], v2["model"]["params"],
                            v3["model"]["params"]),
        ("CPU latency (ms)", v1["latency_cpu_ms_batch1"]["median"],
                              v2["latency_cpu_ms_batch1"]["median"],
                              v3["latency_cpu_ms_batch1"]["median"]),
    ]
    for name, a, b, c in rows:
        if isinstance(a, float):
            print(f"{name:<22} {a:>10.4f} {b:>10.4f} {c:>10.4f}   {c-a:>+8.4f}")
        else:
            print(f"{name:<22} {a:>10,} {b:>10,} {c:>10,}   {c-a:>+8,}")

    print("\nper-class F1:")
    print(f"{'vowel':<6} {'v1':>8} {'v2':>8} {'v3':>8}   {'v3-v1':>8}   note")
    for c in VOWEL_CLASSES:
        f1a = v1["per_class"][c]["f1"]
        f1b = v2["per_class"][c]["f1"]
        f1c = v3["per_class"][c]["f1"]
        note = "<- back vowel" if c in ("o", "u", "eu") else ""
        print(f"  {c:>2s} {DISPLAY[c]} {f1a:>8.3f} {f1b:>8.3f} {f1c:>8.3f}   {f1c-f1a:>+8.3f}   {note}")

    # back-vowel mean
    back = ("o", "u", "eu")
    def mean_f1(d, keys):
        return sum(d["per_class"][k]["f1"] for k in keys) / len(keys)
    print(f"\nback-vowel mean F1:   v1={mean_f1(v1, back):.3f}  "
          f"v2={mean_f1(v2, back):.3f}  v3={mean_f1(v3, back):.3f}")


if __name__ == "__main__":
    main()
