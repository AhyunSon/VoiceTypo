"""2단계 분류기 사전 학습 & 저장.

전체 525개 데이터로 학습:
  - Stage 1: XLSR-53 Layer 16 → SVM (7모음)
  - Stage 2: XLSR-53 Layer 5-7 → SVM (오/우 이진)

사용법:
  python -m vowel_recognition.method_6_embedding.pretrain_models \
    --audio_dir vowel_recognition/dataset
"""

import sys
import os
import argparse
import numpy as np
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}


def syllable_to_vowel(ch):
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    medial = (code % (28 * 21)) // 28
    return _MEDIAL_TO_VOWEL.get(medial)


def parse_vowel_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    first = stem.split('_')[0]
    if first in VOWELS:
        return first
    if len(first) == 1:
        return syllable_to_vowel(first)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', required=True)
    args = parser.parse_args()

    audio_dir = args.audio_dir
    out_dir = os.path.dirname(__file__)

    # 파일 목록
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    all_files = sorted([f for f in os.listdir(audio_dir)
                        if os.path.splitext(f)[1].lower() in audio_exts])

    samples = []
    for f in all_files:
        v = parse_vowel_from_filename(f)
        if v is None:
            continue
        samples.append((f, v))

    filenames = [f for f, _ in samples]
    labels = np.array([v for _, v in samples])

    print(f"데이터: {len(samples)}개")
    for v in VOWELS:
        n = sum(labels == v)
        if n > 0:
            print(f"  {v}: {n}개")

    # Layer cache 로드 (XLSR-53)
    import hashlib
    cache_key = f"allayers_facebook/wav2vec2-large-xlsr-53_{os.path.abspath(audio_dir)}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(out_dir, f"layer_cache_{cache_hash}.npz")

    print(f"\n레이어 캐시 로드: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    cached_files = list(data['filenames'])
    idx_map = {f: i for i, f in enumerate(cached_files)}
    file_indices = [idx_map[f] for f in filenames]

    # ── Stage 1: Layer 16 전체 7모음 SVM ──
    print("\n[Stage 1] Layer 16 전체 7모음 SVM 학습...")
    X_s1 = data['layer_16'][file_indices]

    scaler_s1 = StandardScaler()
    X_s1_scaled = scaler_s1.fit_transform(X_s1)

    clf_s1 = SVC(kernel='rbf', C=10.0, gamma='scale',
                 probability=True, random_state=42)
    clf_s1.fit(X_s1_scaled, labels)

    train_acc = np.mean(clf_s1.predict(X_s1_scaled) == labels) * 100
    print(f"  학습 정확도: {train_acc:.1f}%")
    print(f"  클래스: {list(clf_s1.classes_)}")

    # ── Stage 2: Layer 5-7 오/우 이진 SVM ──
    print("\n[Stage 2] Layer 5-7 오/우 이진 SVM 학습...")
    X_s2 = np.mean([data[f'layer_{l}'][file_indices] for l in [5, 6, 7]], axis=0)

    ou_mask = np.isin(labels, ['오', '우'])
    X_s2_ou = X_s2[ou_mask]
    y_s2_ou = labels[ou_mask]

    scaler_s2 = StandardScaler()
    X_s2_scaled = scaler_s2.fit_transform(X_s2_ou)

    clf_s2 = SVC(kernel='rbf', C=10.0, gamma='scale',
                 probability=True, random_state=42)
    clf_s2.fit(X_s2_scaled, y_s2_ou)

    train_acc2 = np.mean(clf_s2.predict(X_s2_scaled) == y_s2_ou) * 100
    print(f"  학습 정확도: {train_acc2:.1f}%")
    print(f"  클래스: {list(clf_s2.classes_)}")

    # ── 저장 ──
    model_path = os.path.join(out_dir, 'twostage_model.pkl')
    model_data = {
        'stage1': {
            'scaler': scaler_s1,
            'clf': clf_s1,
            'layers': [16],
            'description': 'XLSR-53 Layer 16 -> SVM (7 vowels)',
        },
        'stage2': {
            'scaler': scaler_s2,
            'clf': clf_s2,
            'layers': [5, 6, 7],
            'target_vowels': ['오', '우'],
            'description': 'XLSR-53 Layer 5-7 mean -> SVM (오/우 binary)',
        },
        'model_name': 'facebook/wav2vec2-large-xlsr-53',
        'n_train': len(samples),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n모델 저장: {model_path}")
    print(f"파일 크기: {os.path.getsize(model_path) / 1024:.0f} KB")
    print("완료!")


if __name__ == '__main__':
    main()
