"""Live mic test: prompt a target vowel, record N seconds, classify, mark ✓/✗.

Each session is tagged with --stage (e.g. baseline, A, C). Recordings are
persisted to livetest/<label>_<NNN>.wav so the SAME audio can be re-evaluated
later by scripts/06_evaluate_wav_folder.py against any other checkpoint or
even shipped to a different method/model entirely.

Default mic device is 1 (USB PnP Sound Device, mono) per HANDOFF.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/04_live_test.py \
        --stage baseline
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
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetypo import CKPT_DIR, PROJECT_ROOT, load_config
from voicetypo.audio_io import write_wav
from voicetypo.features import WhisperFeatureExtractor
from voicetypo.live_eval import classify, load_calibration, vowel_core
from voicetypo.model import load_checkpoint


LIVETEST_DIR = PROJECT_ROOT / "livetest"
RESULTS_DIR = PROJECT_ROOT / "results"


def hangul_to_label(cfg: dict) -> dict[str, str]:
    return {h: l for l, h in cfg["vowels"]["display"].items()}


def parse_target(text: str, h2l: dict[str, str], classes: list[str]) -> str | None:
    s = text.strip()
    if not s:
        return None
    if s in h2l:
        return h2l[s]
    if s.lower() in classes:
        return s.lower()
    return None


def next_filename(label: str) -> Path:
    LIVETEST_DIR.mkdir(parents=True, exist_ok=True)
    n = len(list(LIVETEST_DIR.glob(f"{label}_*.wav"))) + 1
    return LIVETEST_DIR / f"{label}_{n:03d}.wav"


def record(sr: int, duration_s: float, device: int | None) -> np.ndarray:
    audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    return audio[:, 0].copy()


def print_summary(rows: list[dict], classes: list[str], display: dict[str, str]):
    if not rows:
        print("[live] no trials recorded.")
        return
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    print(f"\n========== summary ({n} trials) ==========")
    print(f"overall accuracy: {correct}/{n} = {correct/n*100:.1f}%\n")

    per_target: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        d = per_target[r["target_label"]]
        d["n"] += 1
        d["correct"] += int(r["correct"])
    print("per-vowel:")
    for c in classes:
        d = per_target.get(c, {"n": 0, "correct": 0})
        if d["n"] == 0:
            print(f"  {display[c]} ({c}): no trials")
        else:
            pct = d["correct"] / d["n"] * 100
            print(f"  {display[c]} ({c}): {d['correct']}/{d['n']} = {pct:.0f}%")

    confs = Counter((r["target_label"], r["predicted_label"]) for r in rows
                    if not r["correct"])
    print("\nconfusions (target → predicted, count):")
    if not confs:
        print("  (none)")
    else:
        for (t, p), c in confs.most_common():
            print(f"  {display[t]} → {display[p]}: {c}")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "stage", "trial",
              "target_label", "target_hangul",
              "predicted_label", "predicted_hangul",
              "correct", "top1_prob", "probs", "wav_path"]
    # utf-8-sig so Excel renders Hangul correctly
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[live] csv: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   help="session label (e.g. baseline, A, C). Goes into the CSV name.")
    p.add_argument("--device", type=int, default=1,
                   help="sounddevice input index (default 1 = USB PnP)")
    p.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "probe.pt"))
    p.add_argument("--calibration", type=str, default=None,
                   help="optional calibration npz from scripts/05_calibrate.py "
                        "(replaces the MLP probe with user-fit logistic regression)")
    p.add_argument("--duration", type=float, default=1.5,
                   help="recording length per trial in seconds (default 1.5)")
    p.add_argument("--target-trials", type=int, default=35,
                   help="auto-finalize after N trials (default 35 = 7×5)")
    args = p.parse_args()

    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    display = cfg["vowels"]["display"]
    h2l = hangul_to_label(cfg)
    sr = cfg["audio"]["sample_rate"]

    print(f"[live] stage={args.stage}  device={args.device}  ckpt={args.ckpt}")
    print("[live] loading Whisper encoder + probe...")
    extractor = WhisperFeatureExtractor(cfg["encoder"]["model_id"])
    model, ckpt_classes, extra = load_checkpoint(args.ckpt)
    if ckpt_classes != classes:
        print(f"[live] WARNING: ckpt classes {ckpt_classes} != config {classes}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[live] device={device}, val_acc(at_save)={extra.get('val_acc', '?')}")

    calibration = None
    if args.calibration:
        calibration = load_calibration(args.calibration)
        if calibration["encoder_id"] != cfg["encoder"]["model_id"]:
            print(f"[live] WARNING: calibration was fit with "
                  f"{calibration['encoder_id']} but config uses "
                  f"{cfg['encoder']['model_id']} — predictions will be wrong")
        print(f"[live] calibration: {args.calibration}  "
              f"(replaces MLP probe with user LR)")

    print(f"[live] vowels: {' '.join(display[c] for c in classes)}  "
          f"(latin: {' '.join(classes)})")
    print("[live] enter target vowel, then sustain it for ~"
          f"{args.duration:.1f}s after the prompt. 'q' quits early.\n")

    rows: list[dict] = []
    trial = 0
    try:
        while trial < args.target_trials:
            user = input(f"[{trial+1:02d}/{args.target_trials}] 발음할 모음 (예: 아) 또는 q: ")
            if user.strip().lower() in ("q", "quit", "exit"):
                break
            label = parse_target(user, h2l, classes)
            if label is None:
                print(f"  ! 알 수 없는 모음 '{user}'. 다시 입력.")
                continue

            print(f"  녹음 {args.duration:.1f}s ... ", end="", flush=True)
            wav = record(sr=sr, duration_s=args.duration, device=args.device)
            print("끝")

            wav_path = next_filename(label)
            write_wav(wav_path, wav, sr)

            seg = vowel_core(wav, sr, cfg)
            pred_idx, probs = classify(model, extractor, seg, sr, classes, device,
                                       calibration=calibration)
            pred_label = classes[pred_idx]
            top1 = float(probs[pred_idx])
            ok = (pred_label == label)
            mark = "✓" if ok else "✗"
            print(f"  결과: {display[label]} → {display[pred_label]} {mark}  "
                  f"({top1*100:.1f}%, seg {len(seg)/sr*1000:.0f}ms, "
                  f"{wav_path.name})")

            rows.append({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": args.stage,
                "trial": trial + 1,
                "target_label": label,
                "target_hangul": display[label],
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
            trial += 1
    except KeyboardInterrupt:
        print("\n[live] interrupted.")

    print_summary(rows, classes, display)
    if rows:
        write_csv(RESULTS_DIR / f"livetest_method2_{args.stage}.csv", rows)


if __name__ == "__main__":
    main()
