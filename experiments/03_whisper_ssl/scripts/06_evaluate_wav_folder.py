"""Re-evaluate the saved livetest/ wavs against any checkpoint.

Use this to compare model versions (or ship the same recordings to a different
method/Claude) without re-recording. Filename must encode the target as
<label>_<num>[_extra].wav, where <label> is one of the 7 latin vowel codes
(a, e, i, o, u, eu, eo).

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/06_evaluate_wav_folder.py \
        --ckpt data/checkpoints/probe.pt \
        --output results/livetest_eval_method2.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetypo import CKPT_DIR, PROJECT_ROOT, load_config
from voicetypo.audio_io import read_wav
from voicetypo.features import WhisperFeatureExtractor
from voicetypo.live_eval import classify, load_calibration, vowel_core
from voicetypo.model import load_checkpoint


LIVETEST_DIR = PROJECT_ROOT / "livetest"
RESULTS_DIR = PROJECT_ROOT / "results"

# accept  a_001.wav, eu_017.wav, a_001_takeB.wav, ...
_FNAME_RE = re.compile(r"^([a-z]+)_(\d+)(?:_.*)?\.wav$", re.IGNORECASE)


def parse_target_from_filename(path: Path, classes: list[str]) -> str | None:
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    label = m.group(1).lower()
    return label if label in classes else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=str, default=str(LIVETEST_DIR),
                   help="folder of <label>_<num>.wav files")
    p.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "probe.pt"),
                   help="probe checkpoint (any compatible model)")
    p.add_argument("--calibration", type=str, default=None,
                   help="optional calibration npz from scripts/05_calibrate.py "
                        "(replaces the MLP probe with user-fit logistic regression)")
    p.add_argument("--output", type=str,
                   default=str(RESULTS_DIR / "livetest_eval_method2.csv"),
                   help="output CSV path")
    args = p.parse_args()

    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    display = cfg["vowels"]["display"]
    sr = cfg["audio"]["sample_rate"]

    folder = Path(args.folder)
    if not folder.exists():
        print(f"[eval-wav] folder not found: {folder}")
        return
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        print(f"[eval-wav] no wavs in {folder}")
        return

    print(f"[eval-wav] folder={folder} ({len(wavs)} wavs)")
    print(f"[eval-wav] ckpt={args.ckpt}")
    print(f"[eval-wav] output={args.output}")

    print("[eval-wav] loading Whisper encoder + probe...")
    extractor = WhisperFeatureExtractor(cfg["encoder"]["model_id"])
    model, ckpt_classes, extra = load_checkpoint(args.ckpt)
    if ckpt_classes != classes:
        print(f"[eval-wav] WARNING: ckpt classes {ckpt_classes} != config {classes}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[eval-wav] device={device}, val_acc(at_save)={extra.get('val_acc', '?')}")

    calibration = None
    if args.calibration:
        calibration = load_calibration(args.calibration)
        if calibration["encoder_id"] != cfg["encoder"]["model_id"]:
            print(f"[eval-wav] WARNING: calibration was fit with "
                  f"{calibration['encoder_id']} but config uses "
                  f"{cfg['encoder']['model_id']} — predictions will be wrong")
        print(f"[eval-wav] calibration: {args.calibration}  "
              f"(replaces MLP probe with user LR)")

    rows: list[dict] = []
    skipped: list[str] = []
    for wav_path in wavs:
        target = parse_target_from_filename(wav_path, classes)
        if target is None:
            skipped.append(wav_path.name)
            continue
        wav = read_wav(wav_path, target_sr=sr)
        seg = vowel_core(wav, sr, cfg)
        pred_idx, probs = classify(model, extractor, seg, sr, classes, device,
                                   calibration=calibration)
        pred_label = classes[pred_idx]
        top1 = float(probs[pred_idx])
        ok = (pred_label == target)
        mark = "✓" if ok else "✗"
        print(f"  {wav_path.name}: {display[target]} → {display[pred_label]} "
              f"{mark}  ({top1*100:.1f}%)")
        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ckpt": args.ckpt,
            "target_label": target,
            "target_hangul": display[target],
            "predicted_label": pred_label,
            "predicted_hangul": display[pred_label],
            "correct": int(ok),
            "top1_prob": round(top1, 4),
            "probs": json.dumps(
                {c: round(float(probs[i]), 4) for i, c in enumerate(classes)},
                ensure_ascii=False,
            ),
            "wav_path": str(wav_path.relative_to(PROJECT_ROOT)),
        })

    if skipped:
        print(f"\n[eval-wav] skipped {len(skipped)} files (filename did not match "
              f"<label>_<num>.wav with label in {classes}):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    if not rows:
        print("[eval-wav] nothing evaluated.")
        return

    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    print(f"\n========== summary ({n} wavs) ==========")
    print(f"overall accuracy: {correct}/{n} = {correct/n*100:.1f}%\n")

    per_target: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        d = per_target[r["target_label"]]
        d["n"] += 1
        d["correct"] += r["correct"]
    print("per-vowel:")
    for c in classes:
        d = per_target.get(c, {"n": 0, "correct": 0})
        if d["n"] == 0:
            print(f"  {display[c]} ({c}): no wavs")
        else:
            print(f"  {display[c]} ({c}): "
                  f"{d['correct']}/{d['n']} = {d['correct']/d['n']*100:.0f}%")

    confs = Counter((r["target_label"], r["predicted_label"]) for r in rows
                    if not r["correct"])
    if confs:
        print("\nconfusions (target → predicted, count):")
        for (t, p), c in confs.most_common():
            print(f"  {display[t]} → {display[p]}: {c}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "ckpt",
              "target_label", "target_hangul",
              "predicted_label", "predicted_hangul",
              "correct", "top1_prob", "probs", "wav_path"]
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[eval-wav] csv: {out}")


if __name__ == "__main__":
    main()
