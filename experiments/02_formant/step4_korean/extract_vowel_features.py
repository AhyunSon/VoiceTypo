"""
extract_vowel_features.py — TextGrid 파싱 + 모음 시점 F1/F2/F3 추출

입력: step4_korean/zeroth_mfa_aligned/  (MFA 가 생성한 TextGrid)
       └─ speaker_{spk}/{utt}.TextGrid

출력: step4_korean/vowel_features.npz
       X: (N, 9) — F1/F2/F3 at [20%, 50%, 80%]
       y: (N,)   — 모음 라벨 (ㅏ ㅔ ㅣ ㅗ ㅜ ㅡ ㅓ)
       spk: (N,) — 화자 ID
"""

import sys
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from praatio import textgrid

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SAMPLE_RATE, PARAMS

# MFA Korean acoustic model 음소 → 한국어 모음 매핑
# 실제 korean_mfa 모델 출력 (test_corpus 검증):
#   ɐ  → ㅏ      i, iː → ㅣ    o, oː → ㅗ    u, uː → ㅜ
#   e, eː, ɛː → ㅔ    ɨ, ɨː → ㅡ    ʌ, ʌː → ㅓ
PHONE_TO_VOWEL = {
    "ɐ":  "ㅏ",
    "e":  "ㅔ", "eː": "ㅔ", "ɛː": "ㅔ",
    "i":  "ㅣ", "iː": "ㅣ",
    "o":  "ㅗ", "oː": "ㅗ",
    "u":  "ㅜ", "uː": "ㅜ",
    "ɨ":  "ㅡ", "ɨː": "ㅡ",
    "ʌ":  "ㅓ", "ʌː": "ㅓ",
}

SAMPLE_POS = [0.20, 0.50, 0.80]
MIN_VOWEL_DUR = 0.05    # 50ms 미만은 신뢰 X
MAX_VOWEL_DUR = 0.50    # 500ms 초과는 비정상


def extract_9d(wav: np.ndarray, sr: int,
               t_start: float, t_end: float,
               gender: str = "male") -> np.ndarray:
    """주어진 시간 구간에서 F1/F2/F3 추출 (3 시점)."""
    n_start = int(t_start * sr)
    n_end = int(t_end * sr)
    chunk = wav[n_start:n_end].astype(np.float64)
    chunk = chunk - np.mean(chunk)

    snd = parselmouth.Sound(chunk, sampling_frequency=float(sr))
    p = PARAMS[gender]
    try:
        fmt = call(snd, "To Formant (burg)",
                   0.0, p["max_formants"], 5500,
                   p["window_length"], p["pre_emphasis"])
    except Exception:
        return np.full(9, np.nan)

    dur = t_end - t_start
    feat = []
    for pos in SAMPLE_POS:
        t = dur * pos
        for fn in [1, 2, 3]:
            try:
                v = call(fmt, "Get value at time", fn, t, "Hertz", "Linear")
                feat.append(float(v) if not np.isnan(v) else np.nan)
            except Exception:
                feat.append(np.nan)
    return np.array(feat)


def process_corpus(aligned_dir: Path, corpus_dir: Path,
                   out_path: Path,
                   gender: str = "male") -> None:
    X_list, y_list, spk_list = [], [], []
    file_count = 0
    vowel_count = 0
    skip_count = 0

    for spk_dir in sorted(aligned_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = spk_dir.name
        audio_dir = corpus_dir / spk_id

        for tg_path in spk_dir.glob("*.TextGrid"):
            file_count += 1
            utt_id = tg_path.stem
            wav_path = audio_dir / f"{utt_id}.flac"
            if not wav_path.exists():
                # 대체: .wav 시도
                wav_path = audio_dir / f"{utt_id}.wav"
                if not wav_path.exists():
                    continue

            try:
                tg = textgrid.openTextgrid(str(tg_path), False)
                phones_tier = tg.getTier("phones")
                wav, sr = sf.read(str(wav_path))
            except Exception:
                continue

            for interval in phones_tier.entries:
                ph = interval.label.strip()
                if ph not in PHONE_TO_VOWEL:
                    continue
                vowel = PHONE_TO_VOWEL[ph]
                dur = interval.end - interval.start
                if dur < MIN_VOWEL_DUR or dur > MAX_VOWEL_DUR:
                    skip_count += 1
                    continue

                feat = extract_9d(wav, sr,
                                  interval.start, interval.end,
                                  gender=gender)
                if np.any(np.isnan(feat)):
                    skip_count += 1
                    continue

                X_list.append(feat)
                y_list.append(vowel)
                spk_list.append(spk_id)
                vowel_count += 1

        if file_count % 50 == 0:
            print(f"  ... {file_count} TextGrid, {vowel_count} 모음 "
                  f"(skip {skip_count})")

    X = np.array(X_list)
    y = np.array(y_list)
    spk = np.array(spk_list)
    np.savez(str(out_path), X=X, y=y, spk=spk)

    print(f"\n완료:")
    print(f"  TextGrid: {file_count}")
    print(f"  모음 인스턴스: {vowel_count}")
    print(f"  skip: {skip_count}")
    print(f"  화자: {len(set(spk_list))}")
    print(f"  → {out_path}")

    print(f"\n모음별 카운트:")
    for v in ["ㅏ", "ㅔ", "ㅣ", "ㅗ", "ㅜ", "ㅡ", "ㅓ"]:
        print(f"  {v}: {(y == v).sum()}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_dir",
                    default=str(Path(__file__).resolve().parent
                                / "zeroth_mfa_aligned"))
    ap.add_argument("--corpus_dir",
                    default=str(Path(__file__).resolve().parent
                                / "zeroth_mfa"),
                    help="FLAC 위치 (prep_zeroth.py 출력 폴더)")
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent
                                / "vowel_features.npz"))
    ap.add_argument("--gender", default="male",
                    help="Zeroth 화자는 대부분 남성 (Yoon 2015)")
    args = ap.parse_args()
    process_corpus(Path(args.aligned_dir), Path(args.corpus_dir),
                   Path(args.out), gender=args.gender)
