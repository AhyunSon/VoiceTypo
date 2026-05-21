"""Lightweight logic tests that run without torch/transformers.
Run from the repo root: python tests/test_logic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np


def test_syllable_medial_jamo():
    from voicetypo.data.extract_vowels import syllable_medial_jamo

    cases = {
        "아": "ㅏ", "에": "ㅔ", "이": "ㅣ", "오": "ㅗ",
        "우": "ㅜ", "으": "ㅡ", "어": "ㅓ",
        "안": "ㅏ", "녕": "ㅕ", "한": "ㅏ",
        "글": "ㅡ", "을": "ㅡ", "발": "ㅏ",
        "강": "ㅏ", "곰": "ㅗ", "물": "ㅜ",
        "애": "ㅐ",  # ㅐ — extractor folds to "e"
    }
    for syll, want in cases.items():
        got = syllable_medial_jamo(syll)
        assert got == want, f"{syll}: got {got!r}, want {want!r}"
    # non-syllable inputs
    for x in ["", "a", " ", "abc", "한글"]:
        assert syllable_medial_jamo(x) is None, f"{x!r} should be None"
    print("  [ok] syllable_medial_jamo")


def test_jamo_target_filter():
    from voicetypo import load_config
    from voicetypo.data.extract_vowels import syllable_medial_jamo
    cfg = load_config()
    targets = cfg["vowels"]["jamo_map"]
    expected_seven = {"a", "e", "i", "o", "u", "eu", "eo"}
    assert set(targets.values()) == expected_seven
    # ㅑㅕㅛㅠ are diphthongs / glides — must NOT map
    for jamo in ["ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅘ", "ㅝ", "ㅢ"]:
        assert jamo not in targets, f"{jamo} should NOT be a target"
    # ㅐ folds to e
    assert targets["ㅐ"] == "e"
    print("  [ok] jamo_target_filter")


def test_vad_synthetic():
    from voicetypo.audio_io import EnergyVAD, VADConfig
    cfg = VADConfig(sample_rate=16000, frame_ms=20, threshold_db=-40,
                    hangover_ms=200, min_segment_ms=120, max_segment_ms=800)
    vad = EnergyVAD(cfg)

    # 200ms silence -> 300ms tone -> 400ms silence
    sr = 16000
    sil = np.zeros(int(sr * 0.20), dtype=np.float32)
    t = np.arange(int(sr * 0.30)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    tail = np.zeros(int(sr * 0.40), dtype=np.float32)
    audio = np.concatenate([sil, tone, tail])

    # feed in 20ms chunks
    flen = sr * 20 // 1000
    segs = []
    for i in range(0, len(audio) - flen + 1, flen):
        out = vad.feed(audio[i:i + flen])
        if out is not None:
            segs.append(out)
    assert len(segs) == 1, f"expected 1 segment, got {len(segs)}"
    seg_ms = len(segs[0]) * 1000 // sr
    assert 200 <= seg_ms <= 600, f"segment length {seg_ms} ms out of expected range"
    print(f"  [ok] vad_synthetic (segment {seg_ms} ms)")


def test_speaker_disjoint_split():
    from voicetypo.data.dataset import Sample, speaker_disjoint_split

    samples = []
    for spk in range(20):
        for k in range(5):
            samples.append(Sample(
                audio_path=Path(f"/tmp/{spk}_{k}.wav"),
                label=["a", "e", "i", "o", "u", "eu", "eo"][k % 7],
                speaker_id=f"spk_{spk}",
                source="synthetic",
                duration_ms=200,
            ))
    train, val, test = speaker_disjoint_split(samples, val_frac=0.10, test_frac=0.20, seed=42)
    train_spk = {s.speaker_id for s in train}
    val_spk = {s.speaker_id for s in val}
    test_spk = {s.speaker_id for s in test}
    # speaker disjointness — the load-bearing property
    assert train_spk.isdisjoint(val_spk)
    assert train_spk.isdisjoint(test_spk)
    assert val_spk.isdisjoint(test_spk)
    # all speakers covered
    assert (train_spk | val_spk | test_spk) == {f"spk_{i}" for i in range(20)}
    print(f"  [ok] speaker_disjoint_split (train={len(train_spk)} val={len(val_spk)} test={len(test_spk)})")


def test_extractor_slice():
    from voicetypo.data.extract_vowels import VowelExtractor
    ex = VowelExtractor()
    sr = ex.sr
    audio = np.linspace(-1, 1, sr, dtype=np.float32)  # 1s ramp
    # a 200 ms span should yield ~120 ms of core (60% of 200)
    seg = ex.slice_segment(audio, start_s=0.10, end_s=0.30)
    assert seg is not None
    seg_ms = len(seg) * 1000 // sr
    assert 110 <= seg_ms <= 130, f"core ms = {seg_ms}, expected ~120"
    # too-short span -> None
    assert ex.slice_segment(audio, 0.10, 0.115) is None
    print(f"  [ok] extractor_slice ({seg_ms} ms core)")


def test_augmenter_basic():
    from voicetypo import load_config
    from voicetypo.augment import WaveformAugmenter, add_gaussian_noise, apply_gain
    cfg = load_config()
    sr = cfg["audio"]["sample_rate"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal(sr // 4).astype(np.float32) * 0.1  # 250 ms
    # individual ops
    y1 = add_gaussian_noise(x, snr_db=20)
    assert y1.shape == x.shape and not np.array_equal(y1, x)
    y2 = apply_gain(x, gain_db=-6)
    assert np.allclose(y2 / x, 10 ** (-6 / 20), atol=1e-3)
    # full augmenter (no librosa pitch/stretch yet — we'll allow either)
    aug = WaveformAugmenter(cfg["augment"], sample_rate=sr)
    try:
        y3 = aug(x)
        assert y3.dtype == np.float32
        assert abs(len(y3) - len(x)) <= sr * 0.3   # time-stretch can change length
    except ImportError:
        print("  [skip] augmenter skipped (librosa not yet installed)")
        return
    print("  [ok] augmenter_basic")


def main():
    print("== logic tests ==")
    test_syllable_medial_jamo()
    test_jamo_target_filter()
    test_speaker_disjoint_split()
    test_extractor_slice()
    test_vad_synthetic()
    test_augmenter_basic()
    print("\nall good.")


if __name__ == "__main__":
    main()
