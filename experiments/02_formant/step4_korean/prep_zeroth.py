"""
prep_zeroth.py — Zeroth 데이터를 MFA 형식으로 변환

입력: evaluation/external_data/zeroth/train_data_01/{group}/{speaker}/
       ├─ {speaker}_{group}.trans.txt   (combined transcript)
       └─ {speaker}_{group}_{utt}.flac

출력: step4_korean/zeroth_mfa/
       └─ speaker_{spk_id}/
          ├─ {utt_id}.flac
          └─ {utt_id}.lab   (한 줄: 텍스트)

실행:
  python prep_zeroth.py --limit_speakers 1    # 1 화자만
  python prep_zeroth.py                       # 전체 105 화자
"""

import argparse
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
ZEROTH = ROOT / "evaluation" / "external_data" / "zeroth" / "train_data_01"
DEST = Path(__file__).resolve().parent / "zeroth_mfa"


def prep(limit_speakers: int = 0, limit_utts: int = 0) -> None:
    DEST.mkdir(exist_ok=True)
    spk_count = 0
    utt_count = 0

    for group_dir in sorted(ZEROTH.iterdir()):
        if not group_dir.is_dir():
            continue
        for spk_dir in sorted(group_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            spk_id = spk_dir.name
            trans_files = list(spk_dir.glob("*.trans.txt"))
            if not trans_files:
                continue

            out_spk = DEST / f"speaker_{spk_id}"
            out_spk.mkdir(exist_ok=True)

            for tf in trans_files:
                with open(tf, encoding="utf-8") as f:
                    lines = [l.strip().split(" ", 1)
                             for l in f if l.strip()]
                for i, parts in enumerate(lines):
                    if len(parts) != 2:
                        continue
                    utt_id, text = parts
                    flac = spk_dir / f"{utt_id}.flac"
                    if not flac.exists():
                        continue
                    shutil.copy(flac, out_spk / f"{utt_id}.flac")
                    (out_spk / f"{utt_id}.lab").write_text(
                        text, encoding="utf-8"
                    )
                    utt_count += 1
                    if limit_utts and utt_count >= limit_utts:
                        break
                if limit_utts and utt_count >= limit_utts:
                    break

            spk_count += 1
            print(f"  speaker {spk_id}: {len(list(out_spk.glob('*.flac')))} utt")
            if limit_speakers and spk_count >= limit_speakers:
                print(f"\n제한 도달 — {spk_count} 화자")
                print(f"총 {utt_count} 발화 → {DEST}")
                return

    print(f"\n완료 — {spk_count} 화자, {utt_count} 발화 → {DEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_speakers", type=int, default=0,
                    help="처리할 최대 화자 수 (0=전체)")
    ap.add_argument("--limit_utts", type=int, default=0,
                    help="처리할 최대 발화 수 (0=전체)")
    args = ap.parse_args()
    prep(args.limit_speakers, args.limit_utts)
