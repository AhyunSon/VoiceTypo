"""오/우 분류: 전 레이어 탐색 + 레이어 범위 조합 LOSO."""
import sys, io, os, wave, time
import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

BASE = os.path.dirname(__file__)
DATASET = os.path.join(BASE, '..', 'dataset')
LIVE = os.path.join(BASE, 'live_recordings')
TARGET = ['오', '우']
VOWELS = ['아', '어', '오', '우', '으', '이', '에', '애']
_MED = {0: '아', 1: '애', 4: '어', 5: '에', 8: '오', 13: '우', 18: '으', 20: '이'}


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


def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        with wave.open(path, 'r') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            ch = wf.getnchannels()
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            a = a.reshape(-1, ch)[:, 0]
        return a, sr
    else:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path).set_channels(1)
        sr = seg.frame_rate
        sw = seg.sample_width
        raw = seg.raw_data
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


def loso(X, y, spk, speakers):
    all_gt, all_pred = [], []
    for held in speakers:
        test_mask = spk == held
        train_mask = ~test_mask
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        clf.fit(X_tr, y[train_mask])
        all_gt.extend(y[test_mask])
        all_pred.extend(clf.predict(X_te))
    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)
    total = np.mean(all_gt == all_pred) * 100
    oh_a = np.mean(all_pred[all_gt == '오'] == '오') * 100
    oo_a = np.mean(all_pred[all_gt == '우'] == '우') * 100
    return total, oh_a, oo_a


def main():
    # ── 데이터 수집 ──
    files = []
    for f in sorted(os.listdir(DATASET)):
        if os.path.isdir(os.path.join(DATASET, f)):
            continue
        v = parse_vowel(f)
        if v in TARGET:
            files.append((os.path.join(DATASET, f), v, parse_speaker(f)))

    live_dirs = [
        ('서울여성', os.path.join(LIVE, 'session_20260310_145230')),
        ('서울여성2', os.path.join(LIVE, 'session_20260310_151524')),
        ('경상도여성', os.path.join(LIVE, 'speaker_F_20s_gyeongsang')),
        ('20대남성', os.path.join(LIVE, 'speaker_M_20s')),
        ('20대남성2', os.path.join(LIVE, 'session_20260310_153047')),
    ]
    for spk_name, d in live_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            v = f.split('_')[0]
            if v in TARGET:
                files.append((os.path.join(d, f), v, spk_name))

    print(f"전체: {len(files)}개")

    # ── 임베딩 추출 (전 레이어, 캐시) ──
    cache_path = os.path.join(BASE, 'all_layers_ou_cache.npz')
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = [str(p) for p in data['paths']]
        all_layers_cached = data['all_layers']
        path_to_idx = {p: i for i, p in enumerate(cached_paths)}
    else:
        path_to_idx = {}
        all_layers_cached = None
        cached_paths = []

    need = [(p, v, s) for p, v, s in files if str(p) not in path_to_idx]

    if need:
        print(f"새로 추출: {len(need)}개 (전 레이어)")
        fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        xlsr = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        xlsr.eval()

        new_layers = []
        new_paths = []
        t0 = time.perf_counter()
        for i, (p, v, s) in enumerate(need):
            a, sr = load_audio(p)
            if sr != 16000:
                ratio = 16000 / sr
                n_out = int(len(a) * ratio)
                idx = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(a) - 1)
                a = a[idx]
            inp = fe(a, sampling_rate=16000, return_tensors='pt', padding=False)
            with torch.no_grad():
                out = xlsr(**inp, output_hidden_states=True)
            h = out.hidden_states
            layer_embs = [pool(h[l].squeeze(0)) for l in range(len(h))]
            new_layers.append(np.stack(layer_embs))
            new_paths.append(str(p))
            if (i + 1) % 30 == 0:
                eta = (time.perf_counter() - t0) / (i + 1) * (len(need) - i - 1)
                print(f"  [{i+1}/{len(need)}] ETA: {eta:.0f}s", flush=True)
        print(f"추출 완료: {time.perf_counter() - t0:.1f}초")

        new_layers = np.array(new_layers)
        if all_layers_cached is not None:
            combined_paths = cached_paths + new_paths
            combined_layers = np.concatenate([all_layers_cached, new_layers], axis=0)
        else:
            combined_paths = new_paths
            combined_layers = new_layers
        np.savez(cache_path,
                 paths=np.array(combined_paths, dtype=object),
                 all_layers=combined_layers)
        path_to_idx = {p: i for i, p in enumerate(combined_paths)}
        all_layers_cached = combined_layers
        print(f"캐시 저장: shape={combined_layers.shape}")

    # ── 인덱싱 ──
    indices = [path_to_idx[str(p)] for p, _, _ in files]
    y = np.array([v for _, v, _ in files])
    spk = np.array([s for _, _, s in files])
    speakers = sorted(set(spk))
    n_layers = all_layers_cached.shape[1]

    print(f"레이어: {n_layers}, dim: {all_layers_cached.shape[2]}")
    print()

    # ── 단일 레이어 LOSO ──
    print("=" * 60)
    print("  단일 레이어 LOSO (오/우 이진)")
    print("=" * 60)

    single_results = []
    for layer in range(n_layers):
        X = all_layers_cached[indices, layer, :]
        total, oh_a, oo_a = loso(X, y, spk, speakers)
        single_results.append((layer, total, oh_a, oo_a))
        marker = " ★" if total >= 90 else ""
        print(f"  Layer {layer:2d}: {total:5.1f}%  오:{oh_a:5.1f}%  우:{oo_a:5.1f}%{marker}")

    best_single = max(single_results, key=lambda x: x[1])

    # ── 레이어 범위 조합 ──
    print()
    print("=" * 60)
    print("  레이어 범위 조합 LOSO")
    print("=" * 60)

    combos = []
    for start in range(n_layers):
        for end in range(start + 1, min(start + 5, n_layers)):
            X = all_layers_cached[indices, start:end+1, :].mean(axis=1)
            total, oh_a, oo_a = loso(X, y, spk, speakers)
            combos.append((f"L{start}-{end}", total, oh_a, oo_a))

    combos.sort(key=lambda x: x[1], reverse=True)
    print("  상위 10:")
    for name, total, oh_a, oo_a in combos[:10]:
        print(f"    {name:8s}: {total:5.1f}%  오:{oh_a:5.1f}%  우:{oo_a:5.1f}%")

    # ── 비교 ──
    print()
    print("=" * 60)
    print("  현재(L5-7) vs 최적")
    print("=" * 60)

    X_567 = all_layers_cached[indices, 5:8, :].mean(axis=1)
    cur_total, cur_oh, cur_oo = loso(X_567, y, spk, speakers)
    print(f"  현재 L5-7:  {cur_total:5.1f}%  오:{cur_oh:5.1f}%  우:{cur_oo:5.1f}%")
    print(f"  최적 단일:  Layer {best_single[0]}  {best_single[1]:5.1f}%  오:{best_single[2]:5.1f}%  우:{best_single[3]:5.1f}%")
    print(f"  최적 범위:  {combos[0][0]}  {combos[0][1]:5.1f}%  오:{combos[0][2]:5.1f}%  우:{combos[0][3]:5.1f}%")


if __name__ == '__main__':
    main()
