"""Train v4 = frozen Whisper-tiny encoder + 2-layer MLP probe.

Pipeline:
  1. Load shared manifest from voicetypo_new.
  2. Speaker-disjoint split (seed/fractions match v1/v2/v3).
  3. Whisper-tiny encode each clip -> mean+std pool (768-d vector).
     - Train clips: original + `aug_passes` waveform-augmented copies.
     - Cache vectors in .npz so probe training is fast.
  4. Train MLP head with class-weighted CE + cosine LR + early stop.

The encoder is frozen the whole time — only the MLP probe learns.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from voicetypo_light import (
    CKPT_DIR,
    DATA_DIR,
    SPLIT_SEED,
    SPLIT_TEST_FRAC,
    SPLIT_VAL_FRAC,
    VOWEL_CLASSES,
)
from voicetypo_light.augment import AugmentConfig, WaveformAugmenter
from voicetypo_light.dataset import (
    Sample,
    load_manifest,
    speaker_disjoint_split,
)
from voicetypo_light.features import WhisperTinyExtractor, read_wav_mono
from voicetypo_light.model import (
    MLPHeadConfig,
    VowelMLPHead,
    count_parameters,
    model_size_bytes,
    save_checkpoint,
)


class CachedVectorDataset(Dataset):
    """In-memory pooled-vector dataset. X has shape (N, D)."""

    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=True)
        self.X = torch.from_numpy(d["X"]).float()      # (N, D)
        self.y = torch.from_numpy(d["y"]).long()
        self.spk = d["spk"]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], int(self.y[i]), str(self.spk[i])


def _write_cache(npz_path: Path, X: np.ndarray, y: np.ndarray, spk: np.ndarray):
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        spk=spk.astype(object),
    )


def precompute_split(
    samples: list[Sample],
    cache_path: Path,
    extractor: WhisperTinyExtractor,
    desc: str,
    aug_passes: int = 0,
    augmenter: WaveformAugmenter | None = None,
) -> None:
    if cache_path.exists():
        print(f"[feat] {desc}: cache exists, skip ({cache_path.name})")
        return
    cls_to_idx = {c: i for i, c in enumerate(VOWEL_CLASSES)}
    n_total = len(samples) * (1 + aug_passes)
    D = extractor.embed_dim * 2
    feats = np.zeros((n_total, D), dtype=np.float32)
    ys = np.zeros(n_total, dtype=np.int64)
    spks = np.empty(n_total, dtype=object)
    idx = 0
    for s in tqdm(samples, desc=f"feat:{desc}"):
        wav = read_wav_mono(s.audio_path).squeeze(0).cpu().numpy()
        feats[idx] = extractor.from_waveform_np(wav)
        ys[idx] = cls_to_idx[s.label]
        spks[idx] = s.speaker_id
        idx += 1
        for _ in range(aug_passes):
            assert augmenter is not None
            aug_np = augmenter(wav)
            feats[idx] = extractor.from_waveform_np(aug_np)
            ys[idx] = cls_to_idx[s.label]
            spks[idx] = s.speaker_id
            idx += 1
    feats = feats[:idx]
    ys = ys[:idx]
    spks = spks[:idx]
    _write_cache(cache_path, feats, ys, spks)
    print(f"[feat] {desc}: wrote {idx} samples (incl. {aug_passes} aug passes)"
          f" -> {cache_path}")


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
    version: str = "v4",
    encoder_id: str = "openai/whisper-tiny",
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    early_stop_patience: int = 8,
    aug_passes: int = 2,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] version={version}  encoder={encoder_id}  device={device}")

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
    assert not (train_spk & val_spk)
    assert not (train_spk & test_spk)
    assert not (val_spk & test_spk)
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

    extractor = WhisperTinyExtractor(model_id=encoder_id, device=device)
    print(f"[encoder] embed_dim={extractor.embed_dim}  pooled={extractor.embed_dim*2}")

    augmenter = WaveformAugmenter(AugmentConfig()) if aug_passes > 0 else None

    precompute_split(train_s, train_npz, extractor, "train",
                     aug_passes=aug_passes, augmenter=augmenter)
    precompute_split(val_s, val_npz, extractor, "val")
    precompute_split(test_s, test_npz, extractor, "test")

    train_ds = CachedVectorDataset(train_npz)
    val_ds = CachedVectorDataset(val_npz)
    in_dim = train_ds.X.shape[1]
    print(f"[probe] input dim={in_dim}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = VowelMLPHead(MLPHeadConfig(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        n_classes=len(VOWEL_CLASSES),
        dropout=dropout,
    )).to(device)
    print(f"[probe] params={count_parameters(model):,}  "
          f"size={model_size_bytes(model)/1e6:.3f} MB")

    cw = class_weights(train_ds.y.numpy(), len(VOWEL_CLASSES)).to(device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{version}.pt"
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
        print(f"[ep {ep+1:3d}] loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={sched.get_last_lr()[0]:.2e}")
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            bad = 0
            save_checkpoint(
                ckpt, model, VOWEL_CLASSES,
                extra={
                    "val_acc": best_val,
                    "epoch": ep + 1,
                    "version": version,
                    "encoder_id": encoder_id,
                    "encoder_embed_dim": extractor.embed_dim,
                    "aug_passes": aug_passes,
                    "in_dim": in_dim,
                },
            )
            print(f"[ep {ep+1:3d}]  ↑ saved {ckpt.name} (val_acc={best_val:.4f})")
        else:
            bad += 1
            if bad >= early_stop_patience:
                print(f"[stop] early stop at epoch {ep+1}")
                break

    elapsed = time.time() - t0
    test_ds = CachedVectorDataset(test_npz)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    test_acc = evaluate(model, test_loader, device)
    print(f"[done] best_val={best_val:.4f}  test_acc(last-model)={test_acc:.4f}  "
          f"elapsed={elapsed:.1f}s")
    print(f"[done] checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
