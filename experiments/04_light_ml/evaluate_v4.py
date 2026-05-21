"""Evaluate v4 (Whisper-tiny encoder + MLP probe).

Reports:
  - test top-1, macro F1, per-class precision/recall/F1
  - 7x7 confusion matrix
  - CPU inference latency split into:
      probe-only (vector -> logits)
      end-to-end (raw waveform -> logits, including encoder forward)
  - probe size + encoder size on disk
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
    SAMPLE_RATE,
    VOWEL_CLASSES,
)
from voicetypo_light.evaluate import (
    confusion_matrix,
    per_class_metrics,
)
from voicetypo_light.features import WhisperTinyExtractor
from voicetypo_light.model import (
    count_parameters,
    load_checkpoint,
    model_size_bytes,
)
from voicetypo_light.train_v4 import CachedVectorDataset


def measure_probe_latency(probe: torch.nn.Module, in_dim: int,
                          n_iters: int = 500) -> tuple[float, float]:
    torch.set_num_threads(1)
    probe = probe.to("cpu").eval()
    x = torch.randn(1, in_dim)
    with torch.no_grad():
        for _ in range(20):
            probe(x)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            probe(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(np.median(arr))


def measure_end_to_end_latency(extractor: WhisperTinyExtractor,
                               probe: torch.nn.Module,
                               n_iters: int = 50,
                               clip_ms: int = 200) -> tuple[float, float]:
    """Encoder forward + probe forward on a small fake clip, on CPU."""
    torch.set_num_threads(1)
    extractor.model = extractor.model.to("cpu").eval()
    extractor.device = "cpu"
    probe = probe.to("cpu").eval()
    n_samples = int(SAMPLE_RATE * clip_ms / 1000.0)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.05
    # warmup
    for _ in range(5):
        v = extractor.from_waveform_np(audio)
        with torch.no_grad():
            probe(torch.from_numpy(v).unsqueeze(0))
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        v = extractor.from_waveform_np(audio)
        with torch.no_grad():
            probe(torch.from_numpy(v).unsqueeze(0))
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(np.median(arr))


def main(version: str = "v4"):
    ckpt_path = CKPT_DIR / f"{version}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    probe, classes, extra = load_checkpoint(ckpt_path)
    n_classes = len(classes)
    encoder_id = extra.get("encoder_id", "openai/whisper-tiny")
    in_dim = int(extra.get("in_dim", 768))
    print(f"[eval] version={version}  ckpt={ckpt_path.name}")
    print(f"[eval] extra={extra}")

    feat_dir = DATA_DIR / f"features_{version}"
    test_npz = feat_dir / "test.npz"
    if not test_npz.exists():
        raise FileNotFoundError(test_npz)
    test_ds = CachedVectorDataset(test_npz)
    loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe.to(device).eval()

    y_true_all, y_pred_all, spk_all = [], [], []
    with torch.no_grad():
        for X, y, spk in loader:
            X = X.to(device)
            logits = probe(X)
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

    # Latency
    cpu_probe, _, _ = load_checkpoint(ckpt_path)
    probe_mean_ms, probe_median_ms = measure_probe_latency(cpu_probe, in_dim)
    print(f"\nprobe-only CPU latency (b=1): mean={probe_mean_ms:.3f} ms  "
          f"median={probe_median_ms:.3f} ms")

    # End-to-end (encoder + probe) on CPU — what live demo will pay
    print("[lat] loading encoder for end-to-end measurement (CPU) ...")
    extractor_cpu = WhisperTinyExtractor(model_id=encoder_id, device="cpu")
    e2e_mean, e2e_median = measure_end_to_end_latency(extractor_cpu, cpu_probe)
    print(f"end-to-end CPU latency (~200 ms clip, b=1): "
          f"mean={e2e_mean:.1f} ms  median={e2e_median:.1f} ms")

    # Sizes
    probe_params = count_parameters(cpu_probe)
    probe_in_mem_mb = model_size_bytes(cpu_probe) / 1e6
    probe_disk_mb = ckpt_path.stat().st_size / 1e6
    encoder_params = sum(p.numel() for p in extractor_cpu.model.parameters())
    encoder_size_mb = sum(
        p.numel() * p.element_size() for p in extractor_cpu.model.parameters()
    ) / 1e6
    total_size_mb = encoder_size_mb + probe_in_mem_mb
    print(
        f"\nprobe:    params={probe_params:,}  size={probe_in_mem_mb:.3f} MB  "
        f"on-disk={probe_disk_mb:.3f} MB"
    )
    print(
        f"encoder:  params={encoder_params:,}  size={encoder_size_mb:.3f} MB  "
        f"(frozen, FP32)"
    )
    print(f"total deployable size: {total_size_mb:.3f} MB")

    out_dir = RESULTS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "version": version,
        "checkpoint": str(ckpt_path),
        "encoder_id": encoder_id,
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
            "probe_params": probe_params,
            "probe_in_memory_mb": probe_in_mem_mb,
            "probe_on_disk_mb": probe_disk_mb,
            "encoder_params": encoder_params,
            "encoder_size_mb": encoder_size_mb,
            "total_deployable_mb": total_size_mb,
            # Keep keys consistent with v1/v2/v3 for the comparison script
            "params": probe_params + encoder_params,
            "on_disk_mb": total_size_mb,
        },
        "latency_cpu_ms_batch1": {
            "probe_mean": probe_mean_ms,
            "probe_median": probe_median_ms,
            "end_to_end_mean": e2e_mean,
            "end_to_end_median": e2e_median,
            # mirror v1/v2/v3 schema (single number) using end-to-end median
            "mean": e2e_mean,
            "median": e2e_median,
        },
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
