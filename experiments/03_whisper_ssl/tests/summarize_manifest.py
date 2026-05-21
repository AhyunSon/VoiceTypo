"""Summarize the extracted vowel manifest: counts, durations, speaker spread."""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo import PROCESSED_DIR, load_config


def main():
    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    display = cfg["vowels"]["display"]
    path = PROCESSED_DIR / "vowels" / "manifest.jsonl"

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"manifest: {path}")
    print(f"total samples: {len(rows)}")

    by_label_dur = defaultdict(list)
    by_label_spk = defaultdict(set)
    speakers = set()
    for r in rows:
        by_label_dur[r["label"]].append(r["duration_ms"])
        by_label_spk[r["label"]].add(r["speaker_id"])
        speakers.add(r["speaker_id"])

    print(f"unique speakers: {len(speakers)}")
    print()
    print(f"{'label':<6} {'glyph':<6} {'count':>6} {'speakers':>9} "
          f"{'min':>5} {'p25':>5} {'med':>5} {'p75':>5} {'max':>5}  (ms)")
    for c in classes:
        durs = by_label_dur.get(c, [])
        if not durs:
            print(f"{c:<6} {display.get(c,c):<6} {0:>6} {0:>9}  (no samples)")
            continue
        durs_sorted = sorted(durs)
        n = len(durs_sorted)
        p25 = durs_sorted[n // 4]
        med = durs_sorted[n // 2]
        p75 = durs_sorted[3 * n // 4]
        print(f"{c:<6} {display.get(c,c):<6} {n:>6} {len(by_label_spk[c]):>9} "
              f"{min(durs):>5} {p25:>5} {med:>5} {p75:>5} {max(durs):>5}")

    print()
    all_durs = [r["duration_ms"] for r in rows]
    print(f"all durations: min={min(all_durs)}ms  median={statistics.median(all_durs):.0f}ms  "
          f"mean={statistics.mean(all_durs):.0f}ms  max={max(all_durs)}ms")

    print(f"\nfirst 5 manifest rows:")
    for r in rows[:5]:
        # truncate path for readability
        ap = Path(r["audio"])
        print(f"  label={r['label']:<3}  spk={r['speaker_id']:<22}  dur={r['duration_ms']:>3}ms  "
              f"file=...{ap.parent.name}/{ap.name}")


if __name__ == "__main__":
    main()
