"""Training entry-point: pre-compute Whisper embeddings, then train MLP probe."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from voicetypo import CKPT_DIR, PROCESSED_DIR, load_config
from voicetypo.augment import WaveformAugmenter, specaugment_embeddings
from voicetypo.data.dataset import (
    CachedFeatureDataset,
    VowelWaveformDataset,
    load_manifest,
    speaker_disjoint_split,
    write_cached_features,
)
from voicetypo.features import WhisperFeatureExtractor, feature_dim, pool_mean_std
from voicetypo.model import ProbeConfig, VowelProbe, save_checkpoint


def precompute_features(
    samples, classes, extractor: WhisperFeatureExtractor, augmenter=None,
    aug_passes: int = 0, desc: str = "feat",
):
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    feats, ys, spks = [], [], []
    from voicetypo.audio_io import read_wav
    for s in tqdm(samples, desc=desc):
        wav = read_wav(s.audio_path, target_sr=extractor.target_sr)
        passes = [wav]
        if augmenter and aug_passes > 0:
            passes += [augmenter(wav) for _ in range(aug_passes)]
        for w in passes:
            emb = extractor.encode(w, sr=extractor.target_sr)
            v = pool_mean_std(emb).numpy().astype(np.float32)
            feats.append(v)
            ys.append(cls_to_idx[s.label])
            spks.append(s.speaker_id)
    X = np.stack(feats, axis=0) if feats else np.zeros((0, feature_dim(extractor)), dtype=np.float32)
    return X, np.array(ys, dtype=np.int64), np.array(spks, dtype=object)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_probe(
    train_npz: Path, val_npz: Path, classes: list[str], cfg: dict,
    ckpt_path: Path,
):
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[train] CUDA not available, falling back to CPU")

    train_ds = CachedFeatureDataset(train_npz, classes)
    val_ds = CachedFeatureDataset(val_npz, classes)

    in_dim = train_ds.X.shape[1]
    probe = VowelProbe(ProbeConfig(
        in_dim=in_dim,
        n_classes=len(classes),
        hidden_dim=cfg["training"]["hidden_dim"],
        dropout=cfg["training"]["dropout"],
    )).to(device)

    cw = class_weights(train_ds.y.numpy(), len(classes)).to(device)
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(probe.parameters(),
                            lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["training"]["epochs"])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"])

    best_val = 0.0
    bad_epochs = 0
    for ep in range(cfg["training"]["epochs"]):
        probe.train()
        running = 0.0
        n = 0
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            logits = probe(X)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.numel()
            n += y.numel()
        sched.step()

        probe.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                pred = probe(X).argmax(dim=-1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        val_acc = correct / max(1, total)
        train_loss = running / max(1, n)
        print(f"[train] epoch {ep+1:3d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val + 1e-4:
            best_val = val_acc
            bad_epochs = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt_path, probe, classes,
                            extra={"val_acc": best_val, "epoch": ep + 1})
            print(f"[train]  ↑ saved {ckpt_path} (val_acc={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["training"]["early_stop_patience"]:
                print(f"[train] early stop at epoch {ep+1}")
                break
    return best_val


def main():
    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    samples = load_manifest()
    print(f"[train] manifest: {len(samples)} samples, "
          f"{len({s.speaker_id for s in samples})} speakers, "
          f"per-class={Counter(s.label for s in samples)}")
    train_s, val_s, test_s = speaker_disjoint_split(
        samples,
        val_frac=cfg["training"]["speaker_split"]["val_frac"],
        test_frac=cfg["training"]["speaker_split"]["test_frac"],
        seed=cfg["training"]["speaker_split"]["seed"],
    )
    print(f"[train] split: train={len(train_s)} / val={len(val_s)} / test={len(test_s)}")

    extractor = WhisperFeatureExtractor(cfg["encoder"]["model_id"])
    augmenter = WaveformAugmenter(cfg["augment"], sample_rate=cfg["audio"]["sample_rate"])

    feat_dir = PROCESSED_DIR / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    train_npz = feat_dir / "train.npz"
    val_npz = feat_dir / "val.npz"
    test_npz = feat_dir / "test.npz"

    if not train_npz.exists():
        Xtr, ytr, str_ = precompute_features(
            train_s, classes, extractor, augmenter=augmenter, aug_passes=2, desc="feat:train")
        write_cached_features(train_npz, Xtr, ytr, str_)
    if not val_npz.exists():
        Xv, yv, sv = precompute_features(val_s, classes, extractor, desc="feat:val")
        write_cached_features(val_npz, Xv, yv, sv)
    if not test_npz.exists():
        Xt, yt, st = precompute_features(test_s, classes, extractor, desc="feat:test")
        write_cached_features(test_npz, Xt, yt, st)

    ckpt = CKPT_DIR / "probe.pt"
    best = train_probe(train_npz, val_npz, classes, cfg, ckpt)
    print(f"[train] done. best val_acc={best:.4f}. checkpoint at {ckpt}")
    print("[train] Run scripts/02b_evaluate.py (or evaluate.py directly) for "
          "the speaker-disjoint test set and Pansori unseen-speaker eval.")


if __name__ == "__main__":
    main()
