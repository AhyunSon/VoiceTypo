"""오/우 혼동 심층 분석 (XLSR-53 Layer 16).

분석 항목:
  1. 오/우만 PCA / UMAP 시각화
  2. 화자별 오/우 분포 (겹침 정도)
  3. 어떤 화자에서 우→오 혼동이 심한지 (LOSO)
  4. 중앙 안정구간만 사용 시 개선 여부

사용법:
  python -m vowel_recognition.method_6_embedding.analyze_oh_oo \
    --audio_dir vowel_recognition/dataset
"""

import sys
import os
import argparse
import hashlib
import time
import wave
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import umap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ── 상수 ──
VOWELS_ALL = ["아", "어", "오", "우", "으", "이", "에", "애"]
TARGET_VOWELS = ["오", "우"]

VOWEL_COLORS = {'오': '#ffe119', '우': '#3cb44b'}
SPEAKER_COLORS = {'Anna': '#e6194b', '김동규': '#3cb44b', '이은서': '#4363d8'}
SPEAKER_MARKERS = {'Anna': 'o', '김동규': 's', '이은서': '^'}

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
    if first in VOWELS_ALL:
        return first
    if len(first) == 1:
        return syllable_to_vowel(first)
    return None


def parse_metadata(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    meta = {}
    if len(parts) >= 1: meta['syllable'] = parts[0]
    if len(parts) >= 2: meta['gender'] = parts[1]
    if len(parts) >= 3: meta['speaker'] = parts[2]
    if len(parts) >= 4: meta['number'] = parts[3]
    if len(parts) >= 5: meta['condition'] = parts[4]
    if len(parts) >= 6: meta['pitch'] = parts[5]
    return meta


def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        with wave.open(path, 'r') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            ch = wf.getnchannels()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            audio = audio.reshape(-1, ch)[:, 0]
        return audio, sr
    else:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        seg = seg.set_channels(1)
        sr = seg.frame_rate
        raw = seg.raw_data
        sw = seg.sample_width
        if sw == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        return audio, sr


def setup_korean_font():
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


def get_cache_path(model_name, layers, audio_dir):
    key = f"{model_name}_{layers}_{os.path.abspath(audio_dir)}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    cache_dir = os.path.dirname(__file__)
    return os.path.join(cache_dir, f"embeddings_cache_{h}.npz")


def load_embeddings_from_cache(cache_path, filenames):
    """캐시에서 임베딩 로드."""
    if not os.path.exists(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=True)
    cached_files = list(data['filenames'])
    cached_embeddings = data['embeddings']
    if set(filenames) <= set(cached_files):
        idx_map = {f: i for i, f in enumerate(cached_files)}
        return np.array([cached_embeddings[idx_map[f]] for f in filenames])
    return None


def extract_center_embeddings(audio_dir, filenames, model_name, layers,
                              center_ratio=0.5):
    """중앙 안정구간만 잘라서 임베딩 추출 (전체 대비 center_ratio 비율)."""
    from vowel_recognition.method_6_embedding.features import EmbeddingExtractor

    extractor = EmbeddingExtractor(
        model_name=model_name, layers=layers, pooling='mean')

    embeddings = []
    print(f"\n중앙 {center_ratio*100:.0f}% 구간 임베딩 추출 ({len(filenames)}개)...")
    t_start = time.perf_counter()

    for i, fname in enumerate(filenames):
        filepath = os.path.join(audio_dir, fname)
        audio, sr = load_audio(filepath)

        # 중앙 구간 추출
        total = len(audio)
        margin = int(total * (1 - center_ratio) / 2)
        center_audio = audio[margin:total - margin]

        emb = extractor.extract(center_audio, sr)
        embeddings.append(emb)

        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(filenames) - i - 1)
            print(f"  [{i+1}/{len(filenames)}] ETA: {eta:.0f}s")

    total_time = time.perf_counter() - t_start
    print(f"완료: {total_time:.1f}초")
    return np.array(embeddings, dtype=np.float32)


# ═══════════════════════════════════════════
# 분석 1: PCA / UMAP 시각화
# ═══════════════════════════════════════════
def plot_pca_umap(X, vowels, speakers, out_dir):
    """오/우 PCA + UMAP 2D 시각화."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ── PCA 모음별 ──
    ax = axes[0, 0]
    for v in TARGET_VOWELS:
        mask = vowels == v
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=VOWEL_COLORS[v], label=v, alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
    ax.set_title('PCA: 오 vs 우 (모음별)')
    ax.legend(fontsize=14, markerscale=1.5)
    ax.grid(True, alpha=0.3)

    # ── PCA 화자별 ──
    ax = axes[0, 1]
    for v in TARGET_VOWELS:
        for spk in sorted(set(speakers)):
            mask = (vowels == v) & (speakers == spk)
            if not any(mask):
                continue
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=VOWEL_COLORS[v],
                       marker=SPEAKER_MARKERS.get(spk, 'o'),
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5,
                       label=f'{v}-{spk}')
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
    ax.set_title('PCA: 오 vs 우 (화자별)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── UMAP 모음별 ──
    ax = axes[1, 0]
    for v in TARGET_VOWELS:
        mask = vowels == v
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=VOWEL_COLORS[v], label=v, alpha=0.7, s=60,
                   edgecolors='black', linewidth=0.5)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP: 오 vs 우 (모음별)')
    ax.legend(fontsize=14, markerscale=1.5)
    ax.grid(True, alpha=0.3)

    # ── UMAP 화자별 ──
    ax = axes[1, 1]
    for v in TARGET_VOWELS:
        for spk in sorted(set(speakers)):
            mask = (vowels == v) & (speakers == spk)
            if not any(mask):
                continue
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=VOWEL_COLORS[v],
                       marker=SPEAKER_MARKERS.get(spk, 'o'),
                       alpha=0.7, s=60, edgecolors='black', linewidth=0.5,
                       label=f'{v}-{spk}')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title('UMAP: 오 vs 우 (화자별)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('XLSR-53 Layer16 — 오/우 임베딩 분석', fontsize=16, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'oh_oo_pca_umap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"저장: {path}")

    return X_pca, X_umap


# ═══════════════════════════════════════════
# 분석 2: 화자별 오/우 분포 통계
# ═══════════════════════════════════════════
def analyze_speaker_distribution(X, vowels, speakers, out_dir):
    """화자별 오/우 클러스터 중심 거리, 겹침 정도."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    unique_speakers = sorted(set(speakers))

    print("\n" + "=" * 60)
    print("  화자별 오/우 임베딩 분포 분석")
    print("=" * 60)

    stats = {}
    for spk in unique_speakers:
        oh_mask = (vowels == '오') & (speakers == spk)
        oo_mask = (vowels == '우') & (speakers == spk)

        oh_emb = X_scaled[oh_mask]
        oo_emb = X_scaled[oo_mask]

        if len(oh_emb) == 0 or len(oo_emb) == 0:
            continue

        # 클러스터 중심
        oh_center = oh_emb.mean(axis=0)
        oo_center = oo_emb.mean(axis=0)

        # 중심 간 거리
        center_dist = np.linalg.norm(oh_center - oo_center)

        # 클러스터 내 분산 (평균 거리)
        oh_spread = np.mean(np.linalg.norm(oh_emb - oh_center, axis=1))
        oo_spread = np.mean(np.linalg.norm(oo_emb - oo_center, axis=1))

        # 분리도 = 중심간거리 / (오_분산 + 우_분산)
        separability = center_dist / (oh_spread + oo_spread + 1e-8)

        # 겹침: 오 샘플 중 우 중심이 더 가까운 비율
        oh_to_oh = np.linalg.norm(oh_emb - oh_center, axis=1)
        oh_to_oo = np.linalg.norm(oh_emb - oo_center, axis=1)
        oh_confused = np.sum(oh_to_oo < oh_to_oh) / len(oh_emb) * 100

        oo_to_oh = np.linalg.norm(oo_emb - oh_center, axis=1)
        oo_to_oo = np.linalg.norm(oo_emb - oo_center, axis=1)
        oo_confused = np.sum(oo_to_oh < oo_to_oo) / len(oo_emb) * 100

        stats[spk] = {
            'oh_n': len(oh_emb), 'oo_n': len(oo_emb),
            'center_dist': center_dist,
            'oh_spread': oh_spread, 'oo_spread': oo_spread,
            'separability': separability,
            'oh_confused_pct': oh_confused,
            'oo_confused_pct': oo_confused,
        }

        print(f"\n  [{spk}] 오:{len(oh_emb)}개, 우:{len(oo_emb)}개")
        print(f"    중심 간 거리:    {center_dist:.3f}")
        print(f"    오 클러스터 폭:  {oh_spread:.3f}")
        print(f"    우 클러스터 폭:  {oo_spread:.3f}")
        print(f"    분리도:          {separability:.3f}  (>1이면 잘 분리됨)")
        print(f"    오→우 혼동률:    {oh_confused:.1f}%  (오인데 우 중심이 더 가까운 비율)")
        print(f"    우→오 혼동률:    {oo_confused:.1f}%  (우인데 오 중심이 더 가까운 비율)")

    # 전체 통계
    oh_all = X_scaled[vowels == '오']
    oo_all = X_scaled[vowels == '우']
    oh_c = oh_all.mean(axis=0)
    oo_c = oo_all.mean(axis=0)
    dist_all = np.linalg.norm(oh_c - oo_c)
    spread_oh = np.mean(np.linalg.norm(oh_all - oh_c, axis=1))
    spread_oo = np.mean(np.linalg.norm(oo_all - oo_c, axis=1))
    sep_all = dist_all / (spread_oh + spread_oo + 1e-8)

    print(f"\n  [전체] 오:{len(oh_all)}개, 우:{len(oo_all)}개")
    print(f"    중심 간 거리: {dist_all:.3f}")
    print(f"    분리도:       {sep_all:.3f}")

    return stats


# ═══════════════════════════════════════════
# 분석 3: LOSO에서 화자별 오/우 혼동 상세
# ═══════════════════════════════════════════
def loso_oh_oo_detail(X, vowels, speakers):
    """오/우만으로 LOSO 이진 분류 → 화자별 상세 결과."""
    unique_speakers = sorted(set(speakers))

    print("\n" + "=" * 60)
    print("  LOSO 오/우 이진 분류 상세 (SVM)")
    print("=" * 60)

    all_gt, all_pred = [], []

    for held_out in unique_speakers:
        test_mask = speakers == held_out
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = vowels[train_mask], vowels[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        clf.fit(X_train_s, y_train)

        preds = clf.predict(X_test_s)
        probs = clf.predict_proba(X_test_s)

        acc = np.mean(preds == y_test) * 100
        oh_mask = y_test == '오'
        oo_mask = y_test == '우'

        oh_acc = np.mean(preds[oh_mask] == '오') * 100 if any(oh_mask) else 0
        oo_acc = np.mean(preds[oo_mask] == '우') * 100 if any(oo_mask) else 0

        print(f"\n  Hold-out [{held_out}]: 전체 {acc:.1f}%")
        print(f"    오 정확도: {oh_acc:.1f}% ({sum(oh_mask)}개)")
        print(f"    우 정확도: {oo_acc:.1f}% ({sum(oo_mask)}개)")

        # 오답 상세
        if any(oo_mask):
            wrong_oo = (preds[oo_mask] != '우')
            if any(wrong_oo):
                wrong_indices = np.where(oo_mask)[0][wrong_oo]
                avg_prob = np.mean(probs[wrong_indices], axis=0)
                cls_labels = clf.classes_
                prob_str = ", ".join(f"{l}:{p:.2f}" for l, p in zip(cls_labels, avg_prob))
                print(f"    우→오 오답 {sum(wrong_oo)}개 평균 확률: {prob_str}")

        if any(oh_mask):
            wrong_oh = (preds[oh_mask] != '오')
            if any(wrong_oh):
                wrong_indices = np.where(oh_mask)[0][wrong_oh]
                avg_prob = np.mean(probs[wrong_indices], axis=0)
                cls_labels = clf.classes_
                prob_str = ", ".join(f"{l}:{p:.2f}" for l, p in zip(cls_labels, avg_prob))
                print(f"    오→우 오답 {sum(wrong_oh)}개 평균 확률: {prob_str}")

        all_gt.extend(y_test)
        all_pred.extend(preds)

    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)
    total_acc = np.mean(all_gt == all_pred) * 100
    print(f"\n  전체 LOSO 오/우 이진 정확도: {total_acc:.1f}%")

    cm = confusion_matrix(all_gt, all_pred, labels=['오', '우'])
    print(f"\n  혼동 행렬:")
    print(f"           예측:오  예측:우")
    print(f"    정답:오  {cm[0,0]:4d}    {cm[0,1]:4d}   ({cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%)")
    print(f"    정답:우  {cm[1,0]:4d}    {cm[1,1]:4d}   ({cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%)")

    return total_acc


# ═══════════════════════════════════════════
# 분석 4: 중앙 안정구간 효과
# ═══════════════════════════════════════════
def compare_center_crop(X_full, X_center, vowels, speakers):
    """전체 구간 vs 중앙 구간 LOSO 비교."""
    print("\n" + "=" * 60)
    print("  중앙 안정구간 효과 비교 (LOSO)")
    print("=" * 60)

    results = {}
    for label, X in [("전체 구간", X_full), ("중앙 50%", X_center)]:
        unique_speakers = sorted(set(speakers))
        all_gt, all_pred = [], []

        for held_out in unique_speakers:
            test_mask = speakers == held_out
            train_mask = ~test_mask

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_mask])
            X_test = scaler.transform(X[test_mask])

            clf = SVC(kernel='rbf', C=10, gamma='scale')
            clf.fit(X_train, vowels[train_mask])
            preds = clf.predict(X_test)

            all_gt.extend(vowels[test_mask])
            all_pred.extend(preds)

        all_gt = np.array(all_gt)
        all_pred = np.array(all_pred)

        total_acc = np.mean(all_gt == all_pred) * 100
        oh_mask = all_gt == '오'
        oo_mask = all_gt == '우'
        oh_acc = np.mean(all_pred[oh_mask] == '오') * 100
        oo_acc = np.mean(all_pred[oo_mask] == '우') * 100

        results[label] = (total_acc, oh_acc, oo_acc)
        print(f"\n  [{label}]")
        print(f"    전체: {total_acc:.1f}%")
        print(f"    오:   {oh_acc:.1f}%")
        print(f"    우:   {oo_acc:.1f}%")

    # 비교
    d_total = results["중앙 50%"][0] - results["전체 구간"][0]
    d_oh = results["중앙 50%"][1] - results["전체 구간"][1]
    d_oo = results["중앙 50%"][2] - results["전체 구간"][2]
    print(f"\n  변화량 (중앙 50% - 전체):")
    print(f"    전체: {d_total:+.1f}%")
    print(f"    오:   {d_oh:+.1f}%")
    print(f"    우:   {d_oo:+.1f}%")

    verdict = "개선" if d_total > 0 else "악화" if d_total < 0 else "동일"
    print(f"\n  결론: 중앙 구간 사용 시 → {verdict}")

    return results


# ═══════════════════════════════════════════
# 분석 5: 화자별 서브플롯 (PCA)
# ═══════════════════════════════════════════
def plot_per_speaker(X, vowels, speakers, out_dir):
    """화자별 오/우 PCA 서브플롯."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    unique_speakers = sorted(set(speakers))
    n_spk = len(unique_speakers)

    fig, axes = plt.subplots(1, n_spk + 1, figsize=(5 * (n_spk + 1), 5))

    # 전체
    ax = axes[0]
    for v in TARGET_VOWELS:
        mask = vowels == v
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=VOWEL_COLORS[v], label=v, alpha=0.7, s=50,
                   edgecolors='black', linewidth=0.4)
    ax.set_title('전체', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')

    # 화자별
    for i, spk in enumerate(unique_speakers):
        ax = axes[i + 1]
        # 배경: 다른 화자 회색
        other = speakers != spk
        ax.scatter(X_pca[other, 0], X_pca[other, 1],
                   c='#dddddd', s=15, alpha=0.3)
        # 해당 화자
        for v in TARGET_VOWELS:
            mask = (vowels == v) & (speakers == spk)
            if not any(mask):
                continue
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=VOWEL_COLORS[v], label=v, alpha=0.8, s=60,
                       edgecolors='black', linewidth=0.4)
        n_oh = sum((vowels == '오') & (speakers == spk))
        n_oo = sum((vowels == '우') & (speakers == spk))
        ax.set_title(f'{spk}\n(오:{n_oh}, 우:{n_oo})', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle('XLSR-53 L16 — 화자별 오/우 PCA', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'oh_oo_per_speaker.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"저장: {path}")


# ═══════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="오/우 혼동 심층 분석")
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--model', default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--skip_center', action='store_true',
                        help='중앙 구간 분석 건너뛰기 (느림)')
    args = parser.parse_args()

    setup_korean_font()

    audio_dir = args.audio_dir
    model_name = args.model
    layers = (args.layer,)
    out_dir = os.path.dirname(__file__)

    # ── 데이터 로드 ──
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    all_files = sorted([f for f in os.listdir(audio_dir)
                        if os.path.splitext(f)[1].lower() in audio_exts])

    samples = []
    for f in all_files:
        vowel = parse_vowel_from_filename(f)
        if vowel not in TARGET_VOWELS:
            continue
        meta = parse_metadata(f)
        samples.append((f, vowel, meta.get('speaker', '?')))

    filenames = [f for f, _, _ in samples]
    vowels = np.array([v for _, v, _ in samples])
    speakers = np.array([s for _, _, s in samples])

    print(f"오/우 데이터: {len(samples)}개")
    for spk in sorted(set(speakers)):
        n_oh = sum((vowels == '오') & (speakers == spk))
        n_oo = sum((vowels == '우') & (speakers == spk))
        print(f"  {spk}: 오={n_oh}, 우={n_oo}")

    # ── 캐시에서 임베딩 로드 ──
    cache_key = f"{model_name}_mean_f0"
    cache_path = get_cache_path(cache_key, layers, audio_dir)
    print(f"\n캐시 경로: {cache_path}")

    # 전체 파일 목록 (캐시 인덱싱을 위해)
    all_audio = sorted([f for f in os.listdir(audio_dir)
                        if os.path.splitext(f)[1].lower() in audio_exts
                        and parse_vowel_from_filename(f) is not None])

    embeddings_all = load_embeddings_from_cache(cache_path, all_audio)
    if embeddings_all is not None:
        # 오/우만 필터
        idx_map = {f: i for i, f in enumerate(all_audio)}
        X = np.array([embeddings_all[idx_map[f]] for f in filenames])
        print(f"캐시에서 로드 완료: shape={X.shape}")
    else:
        print("캐시 없음. 직접 추출합니다...")
        from vowel_recognition.method_6_embedding.features import EmbeddingExtractor
        extractor = EmbeddingExtractor(model_name=model_name, layers=layers, pooling='mean')
        X = []
        for fname in filenames:
            audio, sr = load_audio(os.path.join(audio_dir, fname))
            X.append(extractor.extract(audio, sr))
        X = np.array(X, dtype=np.float32)

    print(f"\n임베딩 shape: {X.shape}")
    print(f"모델: {model_name}, 레이어: {layers}")

    # ── 분석 1: PCA + UMAP ──
    print("\n" + "#" * 60)
    print("  분석 1: PCA / UMAP 시각화")
    print("#" * 60)
    plot_pca_umap(X, vowels, speakers, out_dir)

    # ── 분석 2: 화자별 분포 ──
    print("\n" + "#" * 60)
    print("  분석 2: 화자별 오/우 분포 통계")
    print("#" * 60)
    dist_stats = analyze_speaker_distribution(X, vowels, speakers, out_dir)

    # ── 분석 3: LOSO 상세 ──
    print("\n" + "#" * 60)
    print("  분석 3: LOSO 오/우 이진 분류 상세")
    print("#" * 60)
    loso_acc = loso_oh_oo_detail(X, vowels, speakers)

    # ── 분석 5: 화자별 서브플롯 ──
    plot_per_speaker(X, vowels, speakers, out_dir)

    # ── 분석 4: 중앙 안정구간 ──
    if not args.skip_center:
        print("\n" + "#" * 60)
        print("  분석 4: 중앙 안정구간 효과")
        print("#" * 60)
        X_center = extract_center_embeddings(
            audio_dir, filenames, model_name, layers, center_ratio=0.5)
        center_results = compare_center_crop(X, X_center, vowels, speakers)
    else:
        print("\n[중앙 구간 분석 건너뜀 (--skip_center)]")

    # ── 최종 요약 ──
    print("\n" + "=" * 60)
    print("  최종 진단 요약")
    print("=" * 60)

    # 가장 분리도 낮은 화자
    if dist_stats:
        worst_spk = min(dist_stats, key=lambda s: dist_stats[s]['separability'])
        best_spk = max(dist_stats, key=lambda s: dist_stats[s]['separability'])
        print(f"\n  분리도 최악 화자: {worst_spk} ({dist_stats[worst_spk]['separability']:.3f})")
        print(f"  분리도 최고 화자: {best_spk} ({dist_stats[best_spk]['separability']:.3f})")

        worst_confused = max(dist_stats, key=lambda s: dist_stats[s]['oo_confused_pct'])
        print(f"  우→오 혼동 최악: {worst_confused} ({dist_stats[worst_confused]['oo_confused_pct']:.1f}%)")

    print(f"\n  LOSO 오/우 이진 정확도: {loso_acc:.1f}%")
    print()


if __name__ == '__main__':
    main()
