"""End-to-end smoke test using synthetic vowel-like audio.
Downloads Whisper-base on first run (~290 MB).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def synth_vowel_like(formants_hz, sr=16000, dur=0.25, f0=140.0):
    """Toy vowel-ish signal: sum of formant resonances driven by a pulse train.
    Not real speech — just enough to confirm feature shapes propagate."""
    t = np.arange(int(sr * dur)) / sr
    # glottal-pulse-ish source
    x = np.zeros_like(t, dtype=np.float32)
    period = max(1, int(sr / f0))
    x[::period] = 1.0
    out = np.zeros_like(x)
    for f in formants_hz:
        carrier = np.sin(2 * np.pi * f * t).astype(np.float32)
        out += 0.3 * carrier * x
    out += 0.05 * np.random.randn(len(out)).astype(np.float32)
    out = (out / (np.max(np.abs(out)) + 1e-9) * 0.7).astype(np.float32)
    return out


def main():
    from voicetypo import load_config
    from voicetypo.features import WhisperFeatureExtractor, feature_dim, pool_mean_std
    from voicetypo.model import ProbeConfig, VowelProbe, save_checkpoint, load_checkpoint

    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    print(f"== pipeline smoke test (classes={classes}) ==")

    print("[1/5] loading Whisper-base encoder (downloads on first run)...")
    enc = WhisperFeatureExtractor(cfg["encoder"]["model_id"])
    print(f"      embed_dim={enc.embed_dim}  device={enc.device}")
    fdim = feature_dim(enc)
    assert fdim == enc.embed_dim * 2

    # rough cardinal-vowel formants (Hz) for a male-ish speaker
    # source: standard phonetics references; values approximate
    formant_table = {
        "a":  [730, 1090, 2440],
        "e":  [530, 1840, 2480],
        "i":  [270, 2290, 3010],
        "o":  [570,  840, 2410],
        "u":  [300,  870, 2240],
        "eu": [360, 1310, 2330],
        "eo": [600, 1170, 2400],
    }

    print("[2/5] synthesizing 5 samples per vowel (35 total)...")
    X_list, y_list = [], []
    for cls_idx, c in enumerate(classes):
        for k in range(5):
            f0 = 110 + 25 * k   # vary pitch
            wav = synth_vowel_like(formant_table[c], dur=0.22, f0=f0)
            emb = enc.encode(wav, sr=16000)
            assert emb.ndim == 2 and emb.shape[1] == enc.embed_dim, f"emb shape: {tuple(emb.shape)}"
            v = pool_mean_std(emb)
            assert v.shape == (fdim,), f"pooled shape: {tuple(v.shape)}"
            X_list.append(v)
            y_list.append(cls_idx)
    X = torch.stack(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    print(f"      X={tuple(X.shape)}  y={tuple(y.shape)}")

    print("[3/5] training probe for 200 epochs on synthetic data (overfit OK -- we only check forward/back)...")
    probe = VowelProbe(ProbeConfig(in_dim=fdim, n_classes=len(classes), hidden_dim=128, dropout=0.0))
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    probe.train()
    last_loss = float("inf")
    for ep in range(200):
        logits = probe(X)
        loss = crit(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        last_loss = float(loss.item())
    probe.eval()
    with torch.no_grad():
        pred = probe(X).argmax(dim=-1)
        acc = float((pred == y).float().mean().item())
    print(f"      final_loss={last_loss:.4f}  train_acc={acc:.3f}")
    assert acc > 0.6, f"probe failed to learn on synthetic data (acc={acc:.3f})"

    print("[4/5] checkpoint round-trip...")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "probe.pt"
        save_checkpoint(ckpt, probe, classes, extra={"smoke": True})
        loaded, loaded_classes, extra = load_checkpoint(ckpt)
        assert loaded_classes == classes
        assert extra.get("smoke") is True
        with torch.no_grad():
            ref = probe(X[:3]).numpy()
            new = loaded(X[:3]).numpy()
        assert np.allclose(ref, new, atol=1e-5), "round-trip output mismatch"
    print("      [ok] save/load round-trip")

    print("[5/5] end-to-end inference path (single 220 ms clip)...")
    wav = synth_vowel_like(formant_table["a"], dur=0.22)
    emb = enc.encode(wav, sr=16000)
    v = pool_mean_std(emb).unsqueeze(0)
    with torch.no_grad():
        logits = probe(v)[0]
        probs = torch.softmax(logits, dim=-1).numpy()
    print(f"      input='a'  pred={classes[int(probs.argmax())]}  probs={dict(zip(classes, [round(float(p), 3) for p in probs]))}")
    print("\nall good. heavy path (Whisper + probe) works end-to-end.")


if __name__ == "__main__":
    main()
