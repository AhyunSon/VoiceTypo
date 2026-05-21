"""Calibrate the vowel probe to a single user from saved livetest wavs.

Trains a multinomial logistic regression on the user's pooled Whisper
embeddings → saves W,b to data/calibration/<name>.npz. Apply with
--calibration <path> in scripts/04_live_test.py and 06_evaluate_wav_folder.py.

Why a separate file (not a new probe checkpoint)? The calibration is
*user-specific* and lives on top of any model checkpoint. By keeping it
disjoint we can switch ON/OFF cleanly per stage (baseline / A / C / B) and
measure model improvements without confounding them with user adaptation.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/05_calibrate.py \
        --name user_default \
        --from-csv results/livetest_method2_baseline.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicetypo import PROJECT_ROOT, load_config
from voicetypo.audio_io import read_wav
from voicetypo.features import WhisperFeatureExtractor, pool_mean_std
from voicetypo.live_eval import vowel_core


LIVETEST_DIR = PROJECT_ROOT / "livetest"
CALIB_DIR = PROJECT_ROOT / "data" / "calibration"

_FNAME_RE = re.compile(r"^([a-z]+)_(\d+)(?:_.*)?\.wav$", re.IGNORECASE)


def gather_wavs(folder: Path, csv_path: Path | None,
                classes: list[str]) -> list[tuple[Path, str]]:
    """Collect (wav_path, label) pairs.

    If --from-csv is given we trust the CSV's target_label column (handles
    the case where the user's livetest/ folder accumulated stray recordings
    from earlier sessions). Otherwise parse the filename.
    """
    items: list[tuple[Path, str]] = []
    if csv_path and csv_path.exists():
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                p = PROJECT_ROOT / r["wav_path"]
                if p.exists() and r["target_label"] in classes:
                    items.append((p, r["target_label"]))
        return items
    for w in sorted(folder.glob("*.wav")):
        m = _FNAME_RE.match(w.name)
        if not m:
            continue
        label = m.group(1).lower()
        if label in classes:
            items.append((w, label))
    return items


def extract_features(items: list[tuple[Path, str]], cfg: dict, sr: int,
                     extractor: WhisperFeatureExtractor
                     ) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    for p, lab in items:
        wav = read_wav(p, target_sr=sr)
        seg = vowel_core(wav, sr, cfg)
        emb = extractor.encode(seg, sr=sr)
        v = pool_mean_std(emb).numpy().astype(np.float32)
        feats.append(v)
        labels.append(lab)
    return np.stack(feats), np.array(labels)


def fit_lr(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        solver="lbfgs", C=C, max_iter=2000, class_weight="balanced",
    )
    clf.fit(X, y)
    return clf


def expand_to_full(coef: np.ndarray, intercept: np.ndarray,
                   model_classes: np.ndarray, n_classes: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """sklearn omits classes that had 0 samples — pad them with -inf-ish bias
    so they're never predicted."""
    if coef.shape[0] == n_classes:
        return coef, intercept
    full_W = np.zeros((n_classes, coef.shape[1]), dtype=np.float32)
    full_b = np.full(n_classes, -1e6, dtype=np.float32)
    for i, c_idx in enumerate(model_classes):
        full_W[int(c_idx)] = coef[i]
        full_b[int(c_idx)] = intercept[i]
    return full_W, full_b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="user_default",
                   help="output saved to data/calibration/<name>.npz")
    p.add_argument("--folder", type=str, default=str(LIVETEST_DIR))
    p.add_argument("--from-csv", type=str, default=None,
                   help="restrict to wavs listed in this CSV (by wav_path column)")
    p.add_argument("--C", type=float, default=1.0,
                   help="LR inverse regularization strength (default 1.0)")
    p.add_argument("--no-loo", action="store_true",
                   help="skip leave-one-out cross-validation (faster)")
    args = p.parse_args()

    cfg = load_config()
    classes = cfg["vowels"]["classes"]
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    sr = cfg["audio"]["sample_rate"]

    items = gather_wavs(Path(args.folder),
                        Path(args.from_csv) if args.from_csv else None,
                        classes)
    if not items:
        print("[calib] no wavs found.")
        return
    counts = Counter(lab for _, lab in items)
    print(f"[calib] {len(items)} samples, per-class: "
          f"{ {c: counts.get(c, 0) for c in classes} }")
    if any(counts.get(c, 0) < 2 for c in classes):
        missing = [c for c in classes if counts.get(c, 0) < 2]
        print(f"[calib] WARNING: classes with <2 samples: {missing}  "
              f"(LOO will be skipped, predictions for those classes will be weak)")

    print(f"[calib] loading Whisper encoder ({cfg['encoder']['model_id']})...")
    extractor = WhisperFeatureExtractor(cfg["encoder"]["model_id"])

    print("[calib] extracting embeddings...")
    X, y_str = extract_features(items, cfg, sr, extractor)
    y = np.array([cls_to_idx[s] for s in y_str], dtype=np.int64)
    print(f"[calib] X shape {X.shape}")

    clf = fit_lr(X, y, args.C)
    train_acc = float((clf.predict(X) == y).mean())
    print(f"[calib] train accuracy (overfit reference, NOT honest): "
          f"{train_acc*100:.1f}%")

    if not args.no_loo and all(counts.get(c, 0) >= 2 for c in classes):
        from sklearn.model_selection import LeaveOneOut
        oof_correct, oof_n = 0, 0
        per_class_correct: dict[int, int] = {i: 0 for i in range(len(classes))}
        per_class_n: dict[int, int] = {i: 0 for i in range(len(classes))}
        for tr, te in LeaveOneOut().split(X):
            cl = fit_lr(X[tr], y[tr], args.C)
            pred = int(cl.predict(X[te])[0])
            true = int(y[te][0])
            oof_correct += int(pred == true)
            oof_n += 1
            per_class_n[true] += 1
            per_class_correct[true] += int(pred == true)
        print(f"[calib] leave-one-out accuracy (honest): "
              f"{oof_correct/oof_n*100:.1f}%  ({oof_correct}/{oof_n})")
        print("[calib] per-class LOO:")
        for i, c in enumerate(classes):
            n = per_class_n[i]
            if n == 0:
                print(f"  {c}: no trials")
            else:
                print(f"  {c}: {per_class_correct[i]}/{n} = "
                      f"{per_class_correct[i]/n*100:.0f}%")

    W, b = expand_to_full(
        clf.coef_.astype(np.float32),
        clf.intercept_.astype(np.float32),
        clf.classes_,
        len(classes),
    )

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    out = CALIB_DIR / f"{args.name}.npz"
    np.savez(
        out,
        W=W, b=b,
        classes=np.array(classes),
        in_dim=np.int32(X.shape[1]),
        encoder_id=cfg["encoder"]["model_id"],
        n_train=np.int32(len(items)),
        train_acc=np.float32(train_acc),
    )
    print(f"[calib] saved: {out}")
    print(f"[calib] use with:  --calibration {out}")


if __name__ == "__main__":
    main()
