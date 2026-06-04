"""Sanity-check the manifest + speaker-disjoint split before training.

Verifies:
  - manifest loads
  - train/val/test speaker sets are disjoint
  - per-class counts in each split look reasonable
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light import SPLIT_SEED, SPLIT_TEST_FRAC, SPLIT_VAL_FRAC, VOWEL_CLASSES
from voicetypo_light.dataset import load_manifest, speaker_disjoint_split


def main():
    samples = load_manifest()
    print(f"manifest: {len(samples)} samples / "
          f"{len({s.speaker_id for s in samples})} speakers")
    train_s, val_s, test_s = speaker_disjoint_split(
        samples, SPLIT_VAL_FRAC, SPLIT_TEST_FRAC, SPLIT_SEED
    )
    train_spk = {s.speaker_id for s in train_s}
    val_spk = {s.speaker_id for s in val_s}
    test_spk = {s.speaker_id for s in test_s}
    print(f"train: {len(train_s):>5d} samples / {len(train_spk):>3d} speakers")
    print(f"val:   {len(val_s):>5d} samples / {len(val_spk):>3d} speakers")
    print(f"test:  {len(test_s):>5d} samples / {len(test_spk):>3d} speakers")
    assert not (train_spk & val_spk),  f"train∩val:  {train_spk & val_spk}"
    assert not (train_spk & test_spk), f"train∩test: {train_spk & test_spk}"
    assert not (val_spk & test_spk),   f"val∩test:   {val_spk & test_spk}"
    print("OK - speaker sets are disjoint")
    for name, split in [("train", train_s), ("val", val_s), ("test", test_s)]:
        c = Counter(s.label for s in split)
        print(f"  {name:>5s} per-class:", {k: c[k] for k in VOWEL_CLASSES})


if __name__ == "__main__":
    main()
