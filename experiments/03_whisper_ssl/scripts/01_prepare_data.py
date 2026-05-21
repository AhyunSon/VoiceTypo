"""Download corpora, run CTC alignment, extract monophthong cores, write manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running this script directly with `python scripts/01_prepare_data.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetypo import load_config
from voicetypo.data.align import CTCAligner
from voicetypo.data.extract_vowels import run_extraction
from voicetypo.data.sources import (
    download_pansori,
    download_zeroth,
    iter_common_voice_ko,
    iter_fleurs_ko,
    iter_zeroth,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-utterances", type=int, default=None,
                        help="cap total utterances processed (smoke testing)")
    parser.add_argument("--skip-cv", action="store_true",
                        help="skip Common Voice (avoid HF auth prompt)")
    parser.add_argument("--skip-fleurs", action="store_true")
    parser.add_argument("--skip-pansori", action="store_true",
                        help="skip Pansori download (eval only)")
    args = parser.parse_args()

    cfg = load_config()

    print("[prepare] step 1/4 — Zeroth-Korean")
    download_zeroth()

    if not args.skip_pansori:
        print("[prepare] step 2/4 — Pansori (held-out eval)")
        try:
            download_pansori()
        except Exception as e:
            print(f"[prepare] pansori download failed: {e}; continuing without it")

    print("[prepare] step 3/4 — alignment + extraction")

    def utterance_iter():
        yield from iter_zeroth("train")
        if not args.skip_fleurs:
            yield from iter_fleurs_ko()
        if not args.skip_cv:
            yield from iter_common_voice_ko()

    def make_aligner():
        return CTCAligner(cfg["aligner"]["model_id"])

    samples = run_extraction(utterance_iter(), make_aligner, limit=args.limit_utterances)
    print(f"[prepare] step 4/4 — done. {len(samples)} vowel segments stored.")


if __name__ == "__main__":
    main()
