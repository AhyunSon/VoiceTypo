"""Inspect what the Wav2Vec2 aligner actually emits on a real Zeroth utterance."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo.audio_io import read_wav
from voicetypo.data.align import CTCAligner
from voicetypo.data.extract_vowels import syllable_medial_jamo
from voicetypo.data.sources import iter_zeroth


def main():
    print("[debug] grabbing 3 utterances from Zeroth train...")
    utts = []
    for i, u in enumerate(iter_zeroth("train")):
        if i >= 3:
            break
        utts.append(u)
    if not utts:
        print("[debug] no Zeroth utterances found — extraction layout may be off")
        # show a deeper fs view
        from voicetypo import RAW_DIR
        zr = RAW_DIR / "zeroth"
        for p in zr.rglob("*.trans.txt"):
            print(f"   transcript file: {p}")
            break
        return

    print(f"[debug] loading aligner...")
    aligner = CTCAligner()
    print(f"[debug] vocab size = {len(aligner.id2tok)}")
    print("[debug] first 80 tokens in vocab:")
    for tid in list(aligner.id2tok.keys())[:80]:
        print(f"   id={tid}  tok={aligner.id2tok[tid]!r}")

    for u in utts:
        print(f"\n=== {u.audio_path.name} ===")
        print(f"   transcript: {u.text!r}")
        audio = read_wav(u.audio_path, target_sr=16000)
        print(f"   audio: {len(audio)} samples = {len(audio)/16000:.2f} s")
        spans = aligner.align(audio, sr=16000)
        print(f"   spans: {len(spans)}")
        for sp in spans[:30]:
            jamo = syllable_medial_jamo(sp.char) if len(sp.char) == 1 else None
            print(f"     [{sp.start_s:.2f}-{sp.end_s:.2f}] char={sp.char!r}  len={len(sp.char)}  medial_jamo={jamo}")


if __name__ == "__main__":
    main()
