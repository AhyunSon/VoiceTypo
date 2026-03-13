"""오/우 임베딩 PCA 시각화 + SVM 결정 경계."""
import sys, io, os, wave
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

BASE = os.path.dirname(__file__)
DATASET = os.path.join(BASE, '..', 'dataset')
LIVE = os.path.join(BASE, 'live_recordings')
TARGET = ['오', '우']
VOWELS = ['아', '어', '오', '우', '으', '이', '에', '애']
_MED = {0: '아', 1: '애', 4: '어', 5: '에', 8: '오', 13: '우', 18: '으', 20: '이'}


def setup_korean_font():
    candidates = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic',
                   'AppleGothic', 'Gulim', 'Dotum']
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return name
    return None


def parse_vowel(fn):
    s = os.path.splitext(fn)[0].split('_')[0]
    if s in VOWELS:
        return s
    if len(s) == 1:
        c = ord(s) - 0xAC00
        if 0 <= c <= 11171:
            return _MED.get((c % (28 * 21)) // 28)
    return None


def parse_speaker(fn):
    p = os.path.splitext(fn)[0].split('_')
    return p[2] if len(p) >= 3 else 'unknown'


def main():
    setup_korean_font()

    # ── 데이터 수집 ──
    files = []

    # 기존 dataset
    for f in sorted(os.listdir(DATASET)):
        full = os.path.join(DATASET, f)
        if os.path.isdir(full):
            continue
        v = parse_vowel(f)
        if v in TARGET:
            files.append((full, v, parse_speaker(f), 'dataset'))

    # live recordings
    live_dirs = [
        ('서울여성', os.path.join(LIVE, 'session_20260310_145230')),
        ('서울여성2', os.path.join(LIVE, 'session_20260310_151524')),
        ('경상도여성', os.path.join(LIVE, 'speaker_F_20s_gyeongsang')),
        ('20대남성', os.path.join(LIVE, 'speaker_M_20s')),
        ('20대남성2', os.path.join(LIVE, 'session_20260310_153047')),
    ]
    for spk, d in live_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            v = f.split('_')[0]
            if v in TARGET:
                files.append((os.path.join(d, f), v, spk, 'live'))

    # ── 캐시에서 임베딩 로드 ──
    cache_path = os.path.join(BASE, 'retrain_stage2_cache.npz')
    cached = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        for i, p in enumerate(data['paths']):
            cached[str(p)] = data['emb2'][i]

    missing = [f for f in files if str(f[0]) not in cached]
    if missing:
        print(f"캐시에 없는 파일 {len(missing)}개. loso_with_live.py를 먼저 실행하세요.")
        return

    X = np.array([cached[str(p)] for p, _, _, _ in files])
    y = np.array([v for _, v, _, _ in files])
    speakers = np.array([s for _, _, s, _ in files])
    sources = np.array([s for _, _, _, s in files])

    print(f"데이터: {len(files)}개 (dataset:{(sources=='dataset').sum()}, live:{(sources=='live').sum()})")

    # ── PCA ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_

    # ── SVM on PCA 2D (결정 경계 시각화용) ──
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    clf.fit(X_pca, y)

    # 그리드 생성
    margin = 2
    x_min, x_max = X_pca[:, 0].min() - margin, X_pca[:, 0].max() + margin
    y_min, y_max = X_pca[:, 1].min() - margin, X_pca[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z_proba = clf.predict_proba(grid)
    # 오 클래스의 확률
    oh_idx = list(clf.classes_).index('오')
    Z = Z_proba[:, oh_idx].reshape(xx.shape)

    # ── 그림 ──
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # --- Plot 1: 모음별 (경계 포함) ---
    ax = axes[0]
    # 배경: 결정 경계 등고선
    ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21),
                cmap='RdYlBu', alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2,
               linestyles='--')

    colors = {'오': '#ffe119', '우': '#3cb44b'}
    for v in TARGET:
        mask = y == v
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors[v], label=v, alpha=0.8, s=80,
                   edgecolors='black', linewidth=0.6, zorder=5)
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=12)
    ax.set_title('오 vs 우 — SVM 결정 경계 (PCA 2D)', fontsize=14)
    ax.legend(fontsize=14, markerscale=1.2)
    ax.grid(True, alpha=0.2)

    # --- Plot 2: 출처별 (dataset vs live) ---
    ax = axes[1]
    ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21),
                cmap='RdYlBu', alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2,
               linestyles='--')

    marker_map = {'dataset': 'o', 'live': '^'}
    for v in TARGET:
        for src in ['dataset', 'live']:
            mask = (y == v) & (sources == src)
            if not mask.any():
                continue
            label = f'{v} ({src})'
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[v], marker=marker_map[src],
                       label=label, alpha=0.8, s=80,
                       edgecolors='black', linewidth=0.6, zorder=5)
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=12)
    ax.set_title('출처별 분포 (○ dataset, △ live)', fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.2)

    # --- Plot 3: 화자별 ---
    ax = axes[2]
    ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 21),
                cmap='RdYlBu', alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2,
               linestyles='--')

    unique_spk = sorted(set(speakers))
    spk_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_spk)))
    spk_markers = {'오': 'o', '우': 's'}
    for i, spk in enumerate(unique_spk):
        for v in TARGET:
            mask = (y == v) & (speakers == spk)
            if not mask.any():
                continue
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[spk_colors[i]], marker=spk_markers[v],
                       label=f'{spk} {v}', alpha=0.8, s=70,
                       edgecolors='black', linewidth=0.4, zorder=5)
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=12)
    ax.set_title('화자별 분포 (○ 오, □ 우)', fontsize=14)
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.2)

    fig.suptitle('XLSR-53 Layer 5-7 — 오/우 임베딩 분석', fontsize=16, y=1.02)
    fig.tight_layout()

    out_path = os.path.join(BASE, 'ou_boundary_pca.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'저장: {out_path}')


if __name__ == '__main__':
    main()
