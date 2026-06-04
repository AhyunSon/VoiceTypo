"""Re-evaluate saved livetest/ wavs against ANY voicetypo_light checkpoint.

Drop the same wav folder (filename pattern <label>_<num>.wav) into v1/v2/v3/v4
to compare them on identical recordings — including wavs recorded by method 2's
04_live_test.py, since the file format and naming is the same.

Run:
    cd C:\\Users\\admin\\voicetypo_light
    PYTHONIOENCODING=utf-8 \\
        C:\\Users\\admin\\voicetypo_new\\.venv\\Scripts\\python.exe \\
        scripts/06_evaluate_wav_folder_v3.py \\
        --ckpt checkpoints/small_cnn_v3.pt \\
        --output results/livetest_eval_v3.csv

To compare all 4 versions on the same wavs:
    for v in v1 v2 v3 v4; do
        case $v in
            v1) ckpt=checkpoints/small_cnn_v1.pt ;;
            v2) ckpt=checkpoints/small_cnn_v2.pt ;;
            v3) ckpt=checkpoints/small_cnn_v3.pt ;;
            v4) ckpt=checkpoints/v4.pt ;;
        esac
        python scripts/06_evaluate_wav_folder_v3.py \\
            --ckpt $ckpt --output results/livetest_eval_$v.csv
    done
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voicetypo_light import (
    CKPT_DIR,
    DISPLAY,
    RESULTS_DIR,
    SAMPLE_RATE,
    VOWEL_CLASSES,
)
from voicetypo_light.live_eval import (
    UnifiedClassifier,
    parse_target_from_filename,
    vowel_core,
)


LIVETEST_DIR = ROOT / "livetest"


def read_wav_mono(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    wav = data[:, 0]
    if sr != SAMPLE_RATE:
        # resample using torchaudio for consistency with training
        import torch
        t = torch.from_numpy(wav).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        wav = t.squeeze(0).numpy()
    return wav.astype(np.float32, copy=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=str, default=str(LIVETEST_DIR),
                   help="folder of <label>_<num>.wav files (default ./livetest)")
    p.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "small_cnn_v3.pt"),
                   help="checkpoint path (default v3)")
    p.add_argument("--output", type=str, default=None,
                   help="output csv path (default results/livetest_eval_<version>.csv)")
    args = p.parse_args()

    classes = VOWEL_CLASSES
    display = DISPLAY

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
    print("[eval-wav] loading classifier ...")
    clf = UnifiedClassifier(args.ckpt)
    if clf.classes != classes:
        print(f"[eval-wav] WARNING: ckpt classes {clf.classes} != expected {classes}")

    out_path = Path(args.output) if args.output else \
        RESULTS_DIR / f"livetest_eval_{clf.version}.csv"
    print(f"[eval-wav] output={out_path}")
    print(f"[eval-wav] device={clf.device}  version={clf.version}  "
          f"kind={clf.kind}  feature={clf.feature_kind}  "
          f"val_acc(at_save)={clf.extra.get('val_acc', '?')}")

    rows: list[dict] = []
    skipped: list[str] = []
    for wav_path in wavs:
        target = parse_target_from_filename(wav_path, classes)
        if target is None:
            skipped.append(wav_path.name)
            continue
        wav = read_wav_mono(wav_path)
        seg = vowel_core(wav, sr=SAMPLE_RATE)
        pred_idx, probs = clf.classify(seg)
        pred_label = classes[pred_idx]
        top1 = float(probs[pred_idx])
        ok = (pred_label == target)
        mark = "OK" if ok else "X"
        print(f"  {wav_path.name}: {display[target]} -> {display[pred_label]} "
              f"{mark}  ({top1*100:.1f}%)")
        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ckpt": str(args.ckpt),
            "version": clf.version,
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
            "wav_path": str(wav_path.relative_to(ROOT) if wav_path.is_relative_to(ROOT)
                            else wav_path),
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
        print("\nconfusions (target -> predicted, count):")
        for (t, p), c in confs.most_common():
            print(f"  {display[t]} -> {display[p]}: {c}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "ckpt", "version",
              "target_label", "target_hangul",
              "predicted_label", "predicted_hangul",
              "correct", "top1_prob", "probs", "wav_path"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[eval-wav] csv: {out_path}")


if __name__ == "__main__":
    main()
