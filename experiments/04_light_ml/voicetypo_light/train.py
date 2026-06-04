"""Train the small MFCC + CNN vowel classifier.

Pipeline (per `version`):
  1. Load shared manifest from voicetypo_new.
  2. Speaker-disjoint split (seed/fractions match method 2).
  3. Pre-compute MFCC tensors per split, cache to data/features_<version>/.
     - For v2, training waveforms get `aug_passes` extra augmented copies
       (pitch ±4 st, stretch 0.9–1.1, gain ±10 dB, noise SNR 5–30 dB).
  4. Compute train-only mean/std, normalize all splits.
  5. Train SmallVowelCNN with class-weighted CE + cosine LR + early stopping.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from voicetypo_light import (
    CKPT_DIR,
    DATA_DIR,
    LOGMEL_N_MELS,
    MFCC_N_MFCC,
    SPLIT_SEED,
    SPLIT_TEST_FRAC,
    SPLIT_VAL_FRAC,
    TARGET_FRAMES,
    VOWEL_CLASSES,
)
from voicetypo_light.augment import AugmentConfig, WaveformAugmenter
from voicetypo_light.dataset import (
    CachedMFCCDataset,
    Sample,
    load_manifest,
    speaker_disjoint_split,
    write_cached,
)
from voicetypo_light.features import (
    LogMelExtractor,
    MFCCExtractor,
    read_wav_mono,
)
from voicetypo_light.model import (
    CNNConfig,
    DeepCNNConfig,
    DeepVowelCNN,
    SmallVowelCNN,
    count_parameters,
    model_size_bytes,
    save_checkpoint,
)


def _build_extractor(feature_kind: str, device: str, target_frames: int):
    if feature_kind == "mfcc":
        return MFCCExtractor(device=device, target_frames=target_frames), MFCC_N_MFCC
    if feature_kind == "logmel":
        return LogMelExtractor(device=device, target_frames=target_frames), LOGMEL_N_MELS
    raise ValueError(f"unknown feature_kind: {feature_kind}")


def precompute_split(
    samples: list[Sample],
    cache_path: Path,
    device: str,
    desc: str,
    target_frames: int,
    feature_kind: str = "mfcc",
    aug_passes: int = 0,
    augmenter: WaveformAugmenter | None = None,
) -> None:
    if cache_path.exists():
        print(f"[feat] {desc}: cache exists, skip ({cache_path.name})")
        return
    ext, n_freq = _build_extractor(feature_kind, device, target_frames)
    cls_to_idx = {c: i for i, c in enumerate(VOWEL_CLASSES)}
    n_total = len(samples) * (1 + aug_passes)
    feats = np.zeros((n_total, 3, n_freq, target_frames), dtype=np.float32)
    ys = np.zeros(n_total, dtype=np.int64)
    spks = np.empty(n_total, dtype=object)
    idx = 0
    for s in tqdm(samples, desc=f"feat:{desc}"):
        wav = read_wav_mono(s.audio_path)         # (1, T) tensor
        wav_np = wav.squeeze(0).cpu().numpy()
        # Pass 0: clean
        feats[idx] = ext.from_waveform(wav).cpu().numpy()
        ys[idx] = cls_to_idx[s.label]
        spks[idx] = s.speaker_id
        idx += 1
        # Augmented passes
        for _ in range(aug_passes):
            assert augmenter is not None, "augmenter required when aug_passes > 0"
            aug_np = augmenter(wav_np)
            aug = torch.from_numpy(aug_np).unsqueeze(0)
            feats[idx] = ext.from_waveform(aug).cpu().numpy()
            ys[idx] = cls_to_idx[s.label]
            spks[idx] = s.speaker_id
            idx += 1
    feats = feats[:idx]
    ys = ys[:idx]
    spks = spks[:idx]
    write_cached(cache_path, feats, ys, spks)
    print(f"[feat] {desc}: wrote {idx} samples (incl. {aug_passes} aug passes)"
          f" -> {cache_path}")


def normalize_inplace(npz_path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]
    Xn = (X - mean) / (std + 1e-6)
    np.savez(npz_path, X=Xn.astype(np.float32), y=d["y"], spk=d["spk"])


def compute_train_stats(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]
    mean = X.mean(axis=(0, 3), keepdims=True).squeeze(0)
    std = X.std(axis=(0, 3), keepdims=True).squeeze(0)
    return mean.astype(np.float32), std.astype(np.float32)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)


def main(
    version: str = "v1",
    feature_kind: str = "mfcc",        # "mfcc" or "logmel"
    model_kind: str = "small",         # "small" or "deep"
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    base_channels: int = 32,
    dropout: float = 0.3,
    early_stop_patience: int = 8,
    time_mask: int = 6,
    freq_mask: int = 6,
    target_frames: int = TARGET_FRAMES,
    aug_passes: int = 0,
    feat_device: str = "cuda",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if feat_device == "cuda" and not torch.cuda.is_available():
        feat_device = "cpu"
    print(f"[init] version={version}  feature={feature_kind}  model={model_kind}  "
          f"device={device}  feat_device={feat_device}")
    print(f"[init] base_channels={base_channels}  target_frames={target_frames}  "
          f"aug_passes={aug_passes}")

    samples = load_manifest()
    print(
        f"[manifest] {len(samples)} samples / "
        f"{len({s.speaker_id for s in samples})} speakers / "
        f"per-class={dict(Counter(s.label for s in samples))}"
    )

    train_s, val_s, test_s = speaker_disjoint_split(
        samples,
        val_frac=SPLIT_VAL_FRAC,
        test_frac=SPLIT_TEST_FRAC,
        seed=SPLIT_SEED,
    )
    train_spk = {s.speaker_id for s in train_s}
    val_spk = {s.speaker_id for s in val_s}
    test_spk = {s.speaker_id for s in test_s}
    assert not (train_spk & val_spk), "train/val speakers overlap"
    assert not (train_spk & test_spk), "train/test speakers overlap"
    assert not (val_spk & test_spk), "val/test speakers overlap"
    print(
        f"[split] train={len(train_s)} ({len(train_spk)} spk) / "
        f"val={len(val_s)} ({len(val_spk)} spk) / "
        f"test={len(test_s)} ({len(test_spk)} spk)"
    )

    feat_dir = DATA_DIR / f"features_{version}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    train_npz = feat_dir / "train.npz"
    val_npz = feat_dir / "val.npz"
    test_npz = feat_dir / "test.npz"
    stats_npz = feat_dir / "norm_stats.npz"

    augmenter = WaveformAugmenter(AugmentConfig()) if aug_passes > 0 else None

    precompute_split(
        train_s, train_npz, feat_device, "train",
        target_frames=target_frames,
        feature_kind=feature_kind,
        aug_passes=aug_passes, augmenter=augmenter,
    )
    precompute_split(val_s, val_npz, feat_device, "val",
                     target_frames=target_frames, feature_kind=feature_kind)
    precompute_split(test_s, test_npz, feat_device, "test",
                     target_frames=target_frames, feature_kind=feature_kind)

    if not stats_npz.exists():
        print("[norm] computing train mean/std and normalizing all splits ...")
        mean, std = compute_train_stats(train_npz)
        np.savez(stats_npz, mean=mean, std=std)
        for p in (train_npz, val_npz, test_npz):
            normalize_inplace(p, mean, std)
        print(f"[norm] stats saved to {stats_npz}")
    else:
        print("[norm] stats exist, assuming caches already normalized")

    train_ds = CachedMFCCDataset(
        train_npz,
        time_mask=time_mask,
        freq_mask=freq_mask,
    )
    val_ds = CachedMFCCDataset(val_npz)
    test_ds = CachedMFCCDataset(test_npz)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0)

    if model_kind == "small":
        model: nn.Module = SmallVowelCNN(
            CNNConfig(
                n_classes=len(VOWEL_CLASSES),
                in_channels=3,
                base_channels=base_channels,
                dropout=dropout,
            )
        ).to(device)
    elif model_kind == "deep":
        model = DeepVowelCNN(
            DeepCNNConfig(
                n_classes=len(VOWEL_CLASSES),
                in_channels=3,
                base_channels=base_channels,
                dropout=dropout,
            )
        ).to(device)
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")
    print(
        f"[model] params={count_parameters(model):,}  "
        f"size={model_size_bytes(model)/1e6:.3f} MB"
    )

    cw = class_weights(train_ds.y.numpy(), len(VOWEL_CLASSES)).to(device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"small_cnn_{version}.pt"
    best_val = 0.0
    bad = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.numel()
            n += int(y.numel())
        sched.step()
        val_acc = evaluate(model, val_loader, device)
        train_loss = running / max(1, n)
        print(
            f"[ep {ep+1:3d}] loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={sched.get_last_lr()[0]:.2e}"
        )
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            bad = 0
            save_checkpoint(
                ckpt, model, VOWEL_CLASSES,
                extra={
                    "val_acc": best_val,
                    "epoch": ep + 1,
                    "version": version,
                    "feature_kind": feature_kind,
                    "model_kind": model_kind,
                    "target_frames": target_frames,
                    "aug_passes": aug_passes,
                    "base_channels": base_channels,
                    "n_freq": LOGMEL_N_MELS if feature_kind == "logmel" else MFCC_N_MFCC,
                },
            )
            print(f"[ep {ep+1:3d}]  ↑ saved {ckpt.name} (val_acc={best_val:.4f})")
        else:
            bad += 1
            if bad >= early_stop_patience:
                print(f"[stop] early stop at epoch {ep+1}")
                break

    elapsed = time.time() - t0
    test_acc = evaluate(model, test_loader, device)
    print(f"[done] best_val={best_val:.4f}  test_acc(last-model)={test_acc:.4f}  "
          f"elapsed={elapsed:.1f}s")
    print(f"[done] checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
