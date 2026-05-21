"""Evaluate a trained small CNN.

Reports:
  - test top-1 accuracy
  - per-class precision / recall / F1
  - 7x7 confusion matrix
  - CPU inference latency (ms / sample, single-threaded)
  - model size on disk vs in-memory

Output is printed and also written to results/<version>/eval.json (+ confusion_matrix.csv).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from voicetypo_light import (
    CKPT_DIR,
    DATA_DIR,
    DISPLAY,
    RESULTS_DIR,
    VOWEL_CLASSES,
)
from voicetypo_light.dataset import CachedMFCCDataset
from voicetypo_light.model import (
    count_parameters,
    load_checkpoint,
    model_size_bytes,
)


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    prec = np.zeros(n_classes, dtype=np.float64)
    rec = np.zeros(n_classes, dtype=np.float64)
    f1 = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec[c], rec[c], f1[c] = p, r, f
    return prec, rec, f1


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def measure_cpu_latency(model: torch.nn.Module, target_frames: int,
                        n_freq: int = 40, n_iters: int = 200) -> tuple[float, float]:
    torch.set_num_threads(1)
    model = model.to("cpu").eval()
    x = torch.randn(1, 3, n_freq, target_frames)
    with torch.no_grad():
        for _ in range(20):
            model(x)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(np.median(arr))


def main(version: str = "v1"):
    # checkpoint name matches train.py's convention
    ckpt_path = CKPT_DIR / f"small_cnn_{version}.pt"
    if not ckpt_path.exists():
        # legacy: v1 was first written as small_cnn.pt before versioning
        legacy = CKPT_DIR / "small_cnn.pt"
        if version == "v1" and legacy.exists():
            ckpt_path = legacy
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, classes, extra = load_checkpoint(ckpt_path)
    n_classes = len(classes)
    target_frames = int(extra.get("target_frames", 32))
    print(f"[eval] version={version}  ckpt={ckpt_path.name}  classes={classes}")
    print(f"[eval] extra={extra}")
    print(f"[eval] target_frames={target_frames}")

    feat_dir = DATA_DIR / f"features_{version}"
    test_npz = feat_dir / "test.npz"
    if not test_npz.exists():
        # v1 legacy lived in data/features/
        legacy_test = DATA_DIR / "features" / "test.npz"
        if version == "v1" and legacy_test.exists():
            test_npz = legacy_test
        else:
            raise FileNotFoundError(
                f"Test cache missing: {test_npz}. Run train for {version} first."
            )
    print(f"[eval] test cache: {test_npz}")
    test_ds = CachedMFCCDataset(test_npz)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    y_true_all, y_pred_all, spk_all = [], [], []
    with torch.no_grad():
        for X, y, spk in loader:
            X = X.to(device)
            logits = model(X)
            pred = logits.argmax(dim=-1).cpu().numpy()
            y_true_all.append(y.numpy())
            y_pred_all.append(pred)
            spk_all.extend(list(spk))
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    spks = np.array(spk_all, dtype=object)
    n_test_spk = len(set(spks.tolist()))

    top1 = float((y_true == y_pred).mean())
    prec, rec, f1 = per_class_metrics(y_true, y_pred, n_classes)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    macro_f1 = float(f1.mean())

    print(f"\n=== {version} test ({len(y_true)} samples / {n_test_spk} speakers) ===")
    print(f"top-1 accuracy: {top1:.4f}")
    print(f"macro F1:       {macro_f1:.4f}")
    print(f"\nper-class (label   precision  recall  F1   support):")
    for c, name in enumerate(classes):
        sup = int((y_true == c).sum())
        print(
            f"  {name:>3s} {DISPLAY[name]}    "
            f"{prec[c]:.3f}      {rec[c]:.3f}    {f1[c]:.3f}   {sup}"
        )

    print("\nconfusion matrix (rows=true, cols=pred):")
    header = "       " + "  ".join(f"{DISPLAY[c]:>4s}" for c in classes)
    print(header)
    for i, name in enumerate(classes):
        row = "  ".join(f"{int(v):>4d}" for v in cm[i])
        print(f"  {DISPLAY[name]:>3s}  {row}")

    cpu_model, _, _ = load_checkpoint(ckpt_path)
    n_freq = int(extra.get("n_freq", 40))
    mean_ms, median_ms = measure_cpu_latency(cpu_model, target_frames, n_freq=n_freq)
    print(f"\nCPU inference latency (batch=1): mean={mean_ms:.2f} ms  median={median_ms:.2f} ms")

    in_mem_mb = model_size_bytes(cpu_model) / 1e6
    on_disk_mb = ckpt_path.stat().st_size / 1e6
    n_params = count_parameters(cpu_model)
    print(f"model size: params={n_params:,}  in-memory={in_mem_mb:.3f} MB  "
          f"on-disk={on_disk_mb:.3f} MB")

    out_dir = RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "version": version,
        "checkpoint": str(ckpt_path),
        "target_frames": target_frames,
        "n_test_samples": int(len(y_true)),
        "n_test_speakers": int(n_test_spk),
        "top1": top1,
        "macro_f1": macro_f1,
        "per_class": {
            classes[c]: {
                "precision": float(prec[c]),
                "recall": float(rec[c]),
                "f1": float(f1[c]),
                "support": int((y_true == c).sum()),
            }
            for c in range(n_classes)
        },
        "model": {
            "params": n_params,
            "in_memory_mb": in_mem_mb,
            "on_disk_mb": on_disk_mb,
        },
        "latency_cpu_ms_batch1": {"mean": mean_ms, "median": median_ms},
        "extra": extra,
    }
    with open(out_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(out_dir / "confusion_matrix.csv", "w", encoding="utf-8") as f:
        f.write("," + ",".join(classes) + "\n")
        for i, name in enumerate(classes):
            f.write(name + "," + ",".join(str(int(v)) for v in cm[i]) + "\n")
    print(f"\nresults written to {out_dir}/eval.json + confusion_matrix.csv")


if __name__ == "__main__":
    main()
