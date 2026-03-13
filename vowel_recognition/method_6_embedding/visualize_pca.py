"""PCA 2D 시각화: 모음별, 화자별, 모음+화자 결합.

사용법:
  python -m vowel_recognition.method_6_embedding.visualize_pca \
    --audio_dir vowel_recognition/dataset
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}

VOWEL_COLORS = {
    '아': '#e6194b', '어': '#f58231', '오': '#ffe119',
    '우': '#3cb44b', '으': '#42d4f4', '이': '#4363d8',
    '에': '#911eb4', '애': '#f032e6',
}

SPEAKER_COLORS = {
    'Anna': '#e6194b',
    '김동규': '#3cb44b',
    '이은서': '#4363d8',
}

SPEAKER_MARKERS = {
    'Anna': 'o',
    '김동규': 's',
    '이은서': '^',
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


def parse_metadata(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    meta = {}
    if len(parts) >= 3: meta['speaker'] = parts[2]
    return meta


def setup_korean_font():
    """한글 폰트 설정."""
    font_candidates = [
        'Malgun Gothic', 'NanumGothic', 'NanumBarunGothic',
        'AppleGothic', 'Gulim', 'Dotum',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in font_candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--cache_dir', default=os.path.dirname(__file__))
    args = parser.parse_args()

    font_name = setup_korean_font()
    print(f"Font: {font_name}")

    audio_dir = args.audio_dir
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    all_files = sorted([f for f in os.listdir(audio_dir)
                        if os.path.splitext(f)[1].lower() in audio_exts])

    samples = []
    for f in all_files:
        vowel = parse_vowel_from_filename(f)
        if vowel is None:
            continue
        meta = parse_metadata(f)
        samples.append((f, vowel, meta.get('speaker', '?')))

    filenames = [f for f, _, _ in samples]
    vowels = np.array([v for _, v, _ in samples])
    speakers = np.array([s for _, _, s in samples])

    # 캐시에서 임베딩 로드
    import glob
    cache_files = glob.glob(os.path.join(args.cache_dir, "embeddings_cache_*.npz"))
    # Layer 4 캐시 찾기
    embeddings = None
    for cp in cache_files:
        data = np.load(cp, allow_pickle=True)
        cached_files = list(data['filenames'])
        if set(filenames) <= set(cached_files) and data['embeddings'].shape[1] == 768:
            idx_map = {f: i for i, f in enumerate(cached_files)}
            embeddings = np.array([data['embeddings'][idx_map[f]] for f in filenames])
            print(f"캐시 로드: {cp}")
            break

    if embeddings is None:
        print("캐시를 찾을 수 없습니다. 먼저 train_and_eval.py를 --layers 4로 실행하세요.")
        sys.exit(1)

    print(f"임베딩: {embeddings.shape}")

    # StandardScaler + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    print(f"PCA 설명 분산: PC1={var_ratio[0]:.3f}, PC2={var_ratio[1]:.3f}, "
          f"합계={sum(var_ratio):.3f}")

    out_dir = os.path.dirname(__file__)

    # ── 그림 1: 모음별 ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for v in VOWELS:
        mask = vowels == v
        if not any(mask):
            continue
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=VOWEL_COLORS[v], label=v, alpha=0.7, s=40, edgecolors='white', linewidth=0.3)
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
    ax.set_title('wav2vec2 Layer4 임베딩 PCA — 모음별')
    ax.legend(fontsize=12, markerscale=1.5)
    ax.grid(True, alpha=0.3)
    path1 = os.path.join(out_dir, 'pca_vowels.png')
    fig.tight_layout()
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"저장: {path1}")

    # ── 그림 2: 화자별 ──
    fig, ax = plt.subplots(figsize=(10, 8))
    for spk in sorted(set(speakers)):
        mask = speakers == spk
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=SPEAKER_COLORS.get(spk, 'gray'),
                   marker=SPEAKER_MARKERS.get(spk, 'o'),
                   label=spk, alpha=0.6, s=40, edgecolors='white', linewidth=0.3)
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
    ax.set_title('wav2vec2 Layer4 임베딩 PCA — 화자별')
    ax.legend(fontsize=12, markerscale=1.5)
    ax.grid(True, alpha=0.3)
    path2 = os.path.join(out_dir, 'pca_speakers.png')
    fig.tight_layout()
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"저장: {path2}")

    # ── 그림 3: 모음+화자 결합 ──
    fig, ax = plt.subplots(figsize=(12, 10))
    for v in VOWELS:
        for spk in sorted(set(speakers)):
            mask = (vowels == v) & (speakers == spk)
            if not any(mask):
                continue
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=VOWEL_COLORS[v],
                       marker=SPEAKER_MARKERS.get(spk, 'o'),
                       alpha=0.65, s=45, edgecolors='white', linewidth=0.3)

    # 범례: 모음 (색) + 화자 (마커)
    vowel_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=VOWEL_COLORS[v], markersize=10, label=v)
                     for v in VOWELS if any(vowels == v)]
    speaker_handles = [plt.Line2D([0], [0], marker=SPEAKER_MARKERS.get(s, 'o'),
                                  color='w', markerfacecolor='gray', markersize=10, label=s)
                       for s in sorted(set(speakers))]

    leg1 = ax.legend(handles=vowel_handles, title='모음', loc='upper left',
                     fontsize=10, title_fontsize=11)
    ax.add_artist(leg1)
    ax.legend(handles=speaker_handles, title='화자', loc='upper right',
              fontsize=10, title_fontsize=11)

    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
    ax.set_title('wav2vec2 Layer4 임베딩 PCA — 모음(색) + 화자(마커)')
    ax.grid(True, alpha=0.3)
    path3 = os.path.join(out_dir, 'pca_vowel_speaker.png')
    fig.tight_layout()
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"저장: {path3}")

    # ── 그림 4: 모음별 서브플롯 (화자 분리) ──
    active_vowels = [v for v in VOWELS if any(vowels == v)]
    n_vowels = len(active_vowels)
    cols = 4
    rows = (n_vowels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for i, v in enumerate(active_vowels):
        ax = axes[i]
        # 배경: 다른 모음 회색
        other_mask = vowels != v
        ax.scatter(X_2d[other_mask, 0], X_2d[other_mask, 1],
                   c='#dddddd', s=10, alpha=0.3)
        # 해당 모음: 화자별 마커
        for spk in sorted(set(speakers)):
            mask = (vowels == v) & (speakers == spk)
            if not any(mask):
                continue
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=SPEAKER_COLORS.get(spk, 'gray'),
                       marker=SPEAKER_MARKERS.get(spk, 'o'),
                       label=spk, s=50, alpha=0.8, edgecolors='white', linewidth=0.3)
        ax.set_title(f'[{v}]', fontsize=14)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # 빈 서브플롯 숨기기
    for j in range(len(active_vowels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('모음별 화자 분포 (회색=다른 모음)', fontsize=14, y=1.01)
    fig.tight_layout()
    path4 = os.path.join(out_dir, 'pca_per_vowel.png')
    fig.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"저장: {path4}")

    print("\n완료!")


if __name__ == '__main__':
    main()
