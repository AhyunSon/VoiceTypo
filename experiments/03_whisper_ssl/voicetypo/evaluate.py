"""Evaluation: per-vowel F1 + confusion matrix on the speaker-disjoint test set
and on the held-out Pansori corpus (unseen-speaker, license-restricted)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from voicetypo import CKPT_DIR, PROCESSED_DIR, load_config
from voicetypo.audio_io import read_wav
from voicetypo.data.dataset import CachedFeatureDataset, load_manifest
from voicetypo.features import WhisperFeatureExtractor, pool_mean_std
from voicetypo.model import load_checkpoint


def _predict(model, X: torch.Tensor, device: str) -> np.ndarray:
    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(X.to(device))
        return logits.argmax(dim=-1).cpu().numpy()


def evaluate_cached(npz_path: Path, ckpt_path: Path, label: str = "test"):
    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    model, ckpt_classes, extra = load_checkpoint(ckpt_path)
    assert ckpt_classes == classes, "class set mismatch between checkpoint and config"
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    ds = CachedFeatureDataset(npz_path, classes)
    pred = _predict(model, ds.X, device)
    y = ds.y.numpy()
    print(f"\n=== eval: {label} ({len(y)} samples) ===")
    print(classification_report(y, pred, labels=list(range(len(classes))), target_names=classes, digits=3))
    print("confusion (rows=true, cols=pred):")
    print(confusion_matrix(y, pred, labels=list(range(len(classes)))))
    return float((pred == y).mean())


def evaluate_pansori(ckpt_path: Path):
    """Run alignment + extraction + classification on Pansori in-memory.
    Returns top-1 accuracy on the held-out unseen speakers."""
    from voicetypo.data.sources import iter_pansori
    from voicetypo.data.align import CTCAligner
    from voicetypo.data.extract_vowels import VowelExtractor

    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    model, ckpt_classes, _ = load_checkpoint(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    print("[pansori] loading aligner + extractor...")
    aligner = CTCAligner(cfg["aligner"]["model_id"])
    extractor = VowelExtractor()
    enc = WhisperFeatureExtractor(cfg["encoder"]["model_id"])

    pansori_root = PROCESSED_DIR / "pansori_segments"
    pansori_root.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = [], []
    for utt in tqdm(iter_pansori(), desc="pansori"):
        try:
            audio = read_wav(utt.audio_path, target_sr=16000)
            spans = aligner.align(audio, sr=16000)
        except Exception:
            continue
        from voicetypo.data.extract_vowels import syllable_medial_jamo
        for sp in spans:
            if len(sp.char) != 1:
                continue
            jamo = syllable_medial_jamo(sp.char)
            if jamo is None or jamo not in extractor.jamo_to_label:
                continue
            label = extractor.jamo_to_label[jamo]
            seg = extractor.slice_segment(audio, sp.start_s, sp.end_s)
            if seg is None:
                continue
            emb = enc.encode(seg, sr=16000)
            v = pool_mean_std(emb).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = int(model(v).argmax(dim=-1).item())
            y_true.append(cls_to_idx[label])
            y_pred.append(pred_idx)

    if not y_true:
        print("[pansori] no segments evaluated (download / alignment failed?).")
        return 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(f"\n=== eval: pansori unseen-speaker ({len(y_true)} segments) ===")
    print(classification_report(y_true, y_pred,
                                labels=list(range(len(classes))),
                                target_names=classes, digits=3))
    print("confusion (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))))
    acc = float((y_true == y_pred).mean())
    print(f"top-1 = {acc:.4f}")
    return acc


def main():
    ckpt = CKPT_DIR / "probe.pt"
    cfg = load_config()
    feat_dir = PROCESSED_DIR / "features"

    in_corpus_test = feat_dir / "test.npz"
    if in_corpus_test.exists():
        evaluate_cached(in_corpus_test, ckpt, "in-corpus held-out speakers")
    else:
        print("[eval] no cached test features; run training first.")

    print()
    pansori_acc = evaluate_pansori(ckpt)
    print(f"\n[eval] pansori top-1 vs target {cfg['eval']['thresholds']['unseen_top1']}: {pansori_acc:.4f}")


if __name__ == "__main__":
    main()
