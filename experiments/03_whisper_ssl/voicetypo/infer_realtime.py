"""Real-time mic CLI: VAD-segmented vowel classification with a probability bar."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

from voicetypo import CKPT_DIR, load_config
from voicetypo.audio_io import EnergyVAD, MicStream, VADConfig, list_input_devices
from voicetypo.features import WhisperFeatureExtractor, pool_mean_std
from voicetypo.model import load_checkpoint


BAR_WIDTH = 28


def render_bar(probs: np.ndarray, classes: list[str], display_map: dict[str, str]) -> str:
    pred = int(np.argmax(probs))
    lines = []
    for i, c in enumerate(classes):
        glyph = display_map.get(c, c)
        bar_len = int(round(probs[i] * BAR_WIDTH))
        bar = "█" * bar_len + "·" * (BAR_WIDTH - bar_len)
        marker = "◀" if i == pred else " "
        lines.append(f"  {glyph}  |{bar}|  {probs[i]*100:5.1f}%  {marker}")
    return "\n".join(lines)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=None,
                        help="sounddevice input index (use --list to enumerate)")
    parser.add_argument("--list", action="store_true", help="list audio input devices and exit")
    parser.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "probe.pt"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="override VAD threshold in dBFS (default from config)")
    args = parser.parse_args()

    if args.list:
        for d in list_input_devices():
            print(f"  [{d['index']}] {d['name']}  (in_ch={d['channels']})")
        return

    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    display = cfg["vowels"]["display"]
    sr = cfg["audio"]["sample_rate"]

    print("[infer] loading Whisper encoder + probe...")
    extractor = WhisperFeatureExtractor(cfg["encoder"]["model_id"])
    model, ckpt_classes, extra = load_checkpoint(args.ckpt)
    if ckpt_classes != classes:
        print(f"[infer] WARNING: checkpoint classes {ckpt_classes} != config {classes}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    print(f"[infer] device={dev} | val_acc(at_save)={extra.get('val_acc', '?')}")

    vad_cfg = VADConfig(
        sample_rate=sr,
        threshold_db=args.threshold if args.threshold is not None else cfg["inference"]["vad_threshold_db"],
        hangover_ms=cfg["inference"]["vad_hangover_ms"],
        min_segment_ms=cfg["inference"]["segment_min_ms"],
        max_segment_ms=cfg["inference"]["segment_max_ms"],
    )
    vad = EnergyVAD(vad_cfg)

    print("[infer] listening — say a single vowel (Ctrl-C to quit)\n")
    print("\n".join([f"  {display[c]}  |{'·'*BAR_WIDTH}|   0.0%   " for c in classes]))
    print()

    last_msg_lines = len(classes) + 2

    try:
        with MicStream(sample_rate=sr, frame_ms=vad_cfg.frame_ms, device=args.device) as mic:
            for frame in mic.frames():
                seg = vad.feed(frame)
                if seg is None:
                    continue
                t0 = time.perf_counter()
                emb = extractor.encode(seg, sr=sr)
                v = pool_mean_std(emb).unsqueeze(0).to(dev)
                with torch.no_grad():
                    logits = model(v)[0].cpu().numpy()
                probs = softmax_np(logits)
                dt = (time.perf_counter() - t0) * 1000

                # erase last block, then redraw
                sys.stdout.write(f"\033[{last_msg_lines}A")
                sys.stdout.write("\033[J")
                sys.stdout.write(render_bar(probs, classes, display) + "\n")
                pred_label = classes[int(np.argmax(probs))]
                sys.stdout.write(f"  -> {display[pred_label]}  ({len(seg)/sr*1000:.0f} ms seg, {dt:.0f} ms infer)\n\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[infer] bye.")


if __name__ == "__main__":
    main()
