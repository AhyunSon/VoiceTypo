"""오/우 임베딩 시각화: TTS vs remote vs live.

Stage 1(Layer 16)과 Stage 2(Layer 5-7) 각각의 PCA 2D 투영.
"""
import sys, os, io, wave, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트
for name in ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Gulim']:
    if any(name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = name
        break
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, '..', 'dataset')
LIVE_DIR = os.path.join(BASE, 'live_recordings')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}

REMOTE_DIRS = [
    'vowel-remote-001_kdg0534 (1)',
    'vowel-remote-001_lynn03 (1)',
    'vowel-remote-001_아현 (1)',
]


def syllable_to_vowel(ch):
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    return _MEDIAL_TO_VOWEL.get((code % (28 * 21)) // 28)


def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        with wave.open(path, 'r') as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            ch = wf.getnchannels()
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            a = a.reshape(-1, ch)[:, 0]
        return a, sr
    else:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path).set_channels(1)
        sr = seg.frame_rate
        raw = seg.raw_data
        sw = seg.sample_width
        if sw == 2:
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        return a, sr


def pool(frames):
    e = frames.norm(dim=1)
    k = max(1, len(e) // 2)
    return frames[torch.topk(e, k).indices].mean(dim=0).numpy().astype(np.float32)


def collect_all_ou():
    """모든 오/우 데이터 수집: TTS, remote, live."""
    samples = []  # (path, vowel, speaker, domain)

    # TTS (이은서 제외)
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    for f in sorted(os.listdir(DATASET_DIR)):
        if os.path.splitext(f)[1].lower() not in audio_exts:
            continue
        if os.path.isdir(os.path.join(DATASET_DIR, f)):
            continue
        parts = os.path.splitext(f)[0].split('_')
        first = parts[0]
        speaker = parts[2] if len(parts) >= 3 else 'unknown'
        if speaker == '이은서':
            continue
        if first in VOWELS:
            vowel = first
        elif len(first) == 1:
            vowel = syllable_to_vowel(first)
        else:
            continue
        if vowel not in ['오', '우']:
            continue
        samples.append((os.path.join(DATASET_DIR, f), vowel, f'TTS_{speaker}', 'TTS'))

    # Remote
    for rd in REMOTE_DIRS:
        d = os.path.join(BASE, rd)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            parts = os.path.splitext(f)[0].split('_')
            if len(parts) < 4:
                continue
            speaker = parts[0]
            vowel = parts[2]
            if vowel not in ['오', '우']:
                continue
            samples.append((os.path.join(d, f), vowel, f'R_{speaker}', 'remote'))

    # Live (서울여성=아현 포함)
    live_sessions = {
        'session_20260310_145230': 'L_서울여성(아현)',
        'session_20260310_151524': 'L_경상도여성',
        'session_20260310_153047': 'L_20대남성',
    }
    for session, speaker in live_sessions.items():
        d = os.path.join(LIVE_DIR, session)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            vowel = f.split('_')[0]
            if vowel not in ['오', '우']:
                continue
            samples.append((os.path.join(d, f), vowel, speaker, 'live'))

    return samples


def get_embeddings(samples, emb16_map, emb567_map):
    need = [s[0] for s in samples if s[0] not in emb16_map]
    if not need:
        return
    print(f'  임베딩 추출: {len(need)}개...', flush=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    model.eval()
    for i, p in enumerate(need):
        audio, sr = load_audio(p)
        if sr != 16000:
            ratio = 16000 / sr
            n_out = int(len(audio) * ratio)
            idx = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(audio) - 1)
            audio = audio[idx]
        inputs = fe(audio, sampling_rate=16000, return_tensors='pt', padding=False)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states
        emb16_map[p] = pool(h[16].squeeze(0))
        emb567_map[p] = (pool(h[5].squeeze(0)) + pool(h[6].squeeze(0)) + pool(h[7].squeeze(0))) / 3.0
        if (i + 1) % 20 == 0:
            print(f'    [{i+1}/{len(need)}]', flush=True)
    print('  완료.')


def main():
    print('오/우 임베딩 시각화')

    samples = collect_all_ou()
    print(f'전체 오/우: {len(samples)}개')

    # 캐시 로드
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}

    get_embeddings(samples, emb16_map, emb567_map)

    # 분류기 경계 로드
    with open(os.path.join(BASE, 'twostage_model.pkl'), 'rb') as f:
        model_data = pickle.load(f)

    # 데이터 준비
    emb16_all = np.array([emb16_map[s[0]] for s in samples])
    emb567_all = np.array([emb567_map[s[0]] for s in samples])
    vowels = [s[1] for s in samples]
    speakers = [s[2] for s in samples]
    domains = [s[3] for s in samples]

    # 화자 목록
    unique_speakers = sorted(set(speakers))
    print(f'화자: {len(unique_speakers)}')
    for spk in unique_speakers:
        oh = sum(1 for s in samples if s[2] == spk and s[1] == '오')
        oo = sum(1 for s in samples if s[2] == spk and s[1] == '우')
        print(f'  {spk:20s}: 오={oh:2d} 우={oo:2d}')

    # ═══════════════════════════════════════
    #  Figure 1: Stage 1 (Layer 16) PCA
    # ═══════════════════════════════════════
    s1_scaler = model_data['stage1']['scaler']
    X1_scaled = s1_scaler.transform(emb16_all)
    pca1 = PCA(n_components=2)
    X1_2d = pca1.fit_transform(X1_scaled)

    # ═══════════════════════════════════════
    #  Figure 2: Stage 2 (Layer 5-7) PCA
    # ═══════════════════════════════════════
    s2_scaler = model_data['stage2']['scaler']
    X2_scaled = s2_scaler.transform(emb567_all)
    pca2 = PCA(n_components=2)
    X2_2d = pca2.fit_transform(X2_scaled)

    # ═══════════════════════════════════════
    #  플롯
    # ═══════════════════════════════════════

    # 색상/마커 설정
    domain_style = {
        'TTS': {'marker_oh': 's', 'marker_oo': 's', 'alpha': 0.3, 'size': 30, 'edge': 'none'},
        'remote': {'marker_oh': 'o', 'marker_oo': 'o', 'alpha': 0.7, 'size': 60, 'edge': 'black'},
        'live': {'marker_oh': '^', 'marker_oo': '^', 'alpha': 0.9, 'size': 100, 'edge': 'red'},
    }

    # 화자별 색상
    cmap = plt.cm.get_cmap('tab10')
    speaker_colors = {}
    # TTS는 회색 계열
    tts_spks = [s for s in unique_speakers if s.startswith('TTS')]
    for i, spk in enumerate(tts_spks):
        speaker_colors[spk] = (0.6, 0.6, 0.6, 0.4)

    # remote/live는 고유 색상
    real_spks = [s for s in unique_speakers if not s.startswith('TTS')]
    for i, spk in enumerate(real_spks):
        speaker_colors[spk] = cmap(i / max(len(real_spks) - 1, 1))

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax_idx, (X_2d, pca, title) in enumerate([
        (X1_2d, pca1, 'Stage 1: Layer 16 (7모음 분류)'),
        (X2_2d, pca2, 'Stage 2: Layer 5-7 (오/우 분류)'),
    ]):
        ax = axes[ax_idx]

        # TTS 먼저 (뒤에)
        for spk in unique_speakers:
            if not spk.startswith('TTS'):
                continue
            for vowel_label, vowel_marker, fc in [('오', 's', '#FF6B6B'), ('우', 's', '#4ECDC4')]:
                mask = [(speakers[i] == spk and vowels[i] == vowel_label) for i in range(len(samples))]
                if not any(mask):
                    continue
                idx = [i for i, m in enumerate(mask) if m]
                ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                          c=fc, marker='s', s=25, alpha=0.25,
                          edgecolors='none', zorder=1)

        # Remote
        for spk in unique_speakers:
            if not spk.startswith('R_'):
                continue
            color = speaker_colors[spk]
            for vowel_label, marker in [('오', 'o'), ('우', 'D')]:
                mask = [(speakers[i] == spk and vowels[i] == vowel_label) for i in range(len(samples))]
                if not any(mask):
                    continue
                idx = [i for i, m in enumerate(mask) if m]
                fc = '#FF6B6B' if vowel_label == '오' else '#4ECDC4'
                ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                          c=fc, marker=marker, s=70, alpha=0.7,
                          edgecolors=color, linewidths=1.5, zorder=2,
                          label=f'{spk} {vowel_label}')

        # Live (크게, 강조)
        for spk in unique_speakers:
            if not spk.startswith('L_'):
                continue
            color = speaker_colors[spk]
            for vowel_label, marker in [('오', '^'), ('우', 'v')]:
                mask = [(speakers[i] == spk and vowels[i] == vowel_label) for i in range(len(samples))]
                if not any(mask):
                    continue
                idx = [i for i, m in enumerate(mask) if m]
                fc = '#FF6B6B' if vowel_label == '오' else '#4ECDC4'
                ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                          c=fc, marker=marker, s=120, alpha=0.9,
                          edgecolors=color, linewidths=2, zorder=3,
                          label=f'{spk} {vowel_label}')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.grid(True, alpha=0.2)

    # 범례 (우측)
    handles, labels = axes[1].get_legend_handles_labels()
    # 중복 제거
    by_label = dict(zip(labels, handles))
    # TTS 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF6B6B',
               markersize=8, alpha=0.3, label='TTS 오'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4ECDC4',
               markersize=8, alpha=0.3, label='TTS 우'),
    ]
    for label, handle in by_label.items():
        legend_elements.append(handle)

    fig.legend(handles=legend_elements, loc='center right',
              bbox_to_anchor=(0.99, 0.5), fontsize=9, framealpha=0.9)

    plt.suptitle('오/우 임베딩 분포: TTS(□) vs Remote(○◇) vs Live(▲▽)',
                fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(right=0.82, top=0.92, wspace=0.25)

    out_path = os.path.join(BASE, 'ou_embedding_all.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n저장: {out_path}')

    # ═══════════════════════════════════════
    #  Figure 2: 화자별 상세 (Stage 2만)
    # ═══════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(14, 10))

    # 배경: TTS
    for vowel_label, fc in [('오', '#FF6B6B'), ('우', '#4ECDC4')]:
        mask = [domains[i] == 'TTS' and vowels[i] == vowel_label for i in range(len(samples))]
        idx = [i for i, m in enumerate(mask) if m]
        if idx:
            ax2.scatter(X2_2d[idx, 0], X2_2d[idx, 1],
                       c=fc, marker='s', s=20, alpha=0.15, edgecolors='none', zorder=1)

    # 실제 화자별 (라벨 포함)
    markers_oh = {'R_kdg0534': 'o', 'R_lynn03': 'o', 'R_아현': 'o',
                  'L_서울여성(아현)': '^', 'L_경상도여성': '^', 'L_20대남성': '^'}
    markers_oo = {'R_kdg0534': 'D', 'R_lynn03': 'D', 'R_아현': 'D',
                  'L_서울여성(아현)': 'v', 'L_경상도여성': 'v', 'L_20대남성': 'v'}

    for spk in real_spks:
        color = speaker_colors[spk]
        for vowel_label, markers, fc in [('오', markers_oh, '#FF6B6B'), ('우', markers_oo, '#4ECDC4')]:
            mask = [speakers[i] == spk and vowels[i] == vowel_label for i in range(len(samples))]
            idx = [i for i, m in enumerate(mask) if m]
            if not idx:
                continue
            m = markers.get(spk, 'o')
            sz = 120 if spk.startswith('L_') else 80
            ax2.scatter(X2_2d[idx, 0], X2_2d[idx, 1],
                       c=fc, marker=m, s=sz, alpha=0.85,
                       edgecolors=color, linewidths=2, zorder=3,
                       label=f'{spk} {vowel_label}')

            # 텍스트 라벨 (live만)
            if spk.startswith('L_'):
                for ii in idx:
                    ax2.annotate(f'{spk[2:]}\n{vowel_label}',
                                (X2_2d[ii, 0], X2_2d[ii, 1]),
                                fontsize=6, alpha=0.7,
                                textcoords='offset points', xytext=(5, 5))

    ax2.set_title('Stage 2 (Layer 5-7): 오/우 화자별 분포', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.grid(True, alpha=0.2)

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    legend2 = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF6B6B',
               markersize=7, alpha=0.2, label='TTS 오'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4ECDC4',
               markersize=7, alpha=0.2, label='TTS 우'),
    ] + list(by_label2.values())
    ax2.legend(handles=legend2,
              labels=['TTS 오', 'TTS 우'] + list(by_label2.keys()),
              loc='best', fontsize=8, framealpha=0.9)

    out_path2 = os.path.join(BASE, 'ou_stage2_detail.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f'저장: {out_path2}')

    print('\n완료!')


if __name__ == '__main__':
    main()
