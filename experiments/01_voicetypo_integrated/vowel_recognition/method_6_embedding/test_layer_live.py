"""순수 테스트: dataset만 학습 → live만 테스트 (레이어별)."""
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


def extract_all_layers(fe, model, files):
    """파일 목록 → (n_files, n_layers, dim) 임베딩."""
    all_embs = []
    t0 = time.perf_counter()
    for i, (p, _, _) in enumerate(files):
        a, sr = load_audio(p)
        if sr != 16000:
            ratio = 16000 / sr
            n_out = int(len(a) * ratio)
            idx = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(a) - 1)
            a = a[idx]
        inp = fe(a, sampling_rate=16000, return_tensors='pt', padding=False)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = out.hidden_states
        embs = [pool(h[l].squeeze(0)) for l in range(len(h))]
        all_embs.append(np.stack(embs))
        if (i + 1) % 30 == 0:
            eta = (time.perf_counter() - t0) / (i + 1) * (len(files) - i - 1)
            print(f"  [{i+1}/{len(files)}] ETA: {eta:.0f}s", flush=True)
    print(f"  완료: {time.perf_counter() - t0:.1f}초")
    return np.array(all_embs)


def main():
    # ── 데이터 수집 ──
    train_files = []
    for f in sorted(os.listdir(DATASET)):
        if os.path.isdir(os.path.join(DATASET, f)):
            continue
        v = parse_vowel(f)
        if v in TARGET:
            train_files.append((os.path.join(DATASET, f), v, parse_speaker(f)))

    test_files = []
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
                test_files.append((os.path.join(d, f), v, spk))

    print(f"학습: dataset {len(train_files)}개 (3화자)")
    print(f"테스트: live {len(test_files)}개 (5화자)")

    y_train = np.array([v for _, v, _ in train_files])
    y_test = np.array([v for _, v, _ in test_files])
    s_test = np.array([s for _, _, s in test_files])

    # ── 임베딩 추출 (캐시) ──
    train_cache = os.path.join(BASE, 'layers_train_cache.npz')
    test_cache = os.path.join(BASE, 'layers_test_cache.npz')

    fe = None
    xlsr = None

    def ensure_model():
        nonlocal fe, xlsr
        if fe is None:
            print("\nXLSR-53 로드 중...")
            fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
            xlsr = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
            xlsr.eval()
            print("로드 완료")

    if os.path.exists(train_cache):
        data = np.load(train_cache)
        X_train_all = data['layers']
        print(f"학습 캐시 로드: {X_train_all.shape}")
    else:
        ensure_model()
        print("\n학습 데이터 임베딩 추출...")
        X_train_all = extract_all_layers(fe, xlsr, train_files)
        np.savez(train_cache, layers=X_train_all)
        print(f"학습 캐시 저장: {X_train_all.shape}")

    if os.path.exists(test_cache):
        data = np.load(test_cache)
        X_test_all = data['layers']
        print(f"테스트 캐시 로드: {X_test_all.shape}")
    else:
        ensure_model()
        print("\n테스트 데이터 임베딩 추출...")
        X_test_all = extract_all_layers(fe, xlsr, test_files)
        np.savez(test_cache, layers=X_test_all)
        print(f"테스트 캐시 저장: {X_test_all.shape}")

    n_layers = X_train_all.shape[1]

    # ── 단일 레이어 ──
    print()
    print("=" * 60)
    print("  dataset 학습 → live 테스트 (단일 레이어)")
    print("=" * 60)

    results = []
    for layer in range(n_layers):
        X_tr = X_train_all[:, layer, :]
        X_te = X_test_all[:, layer, :]
        scaler = StandardScaler()
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        clf.fit(scaler.fit_transform(X_tr), y_train)
        preds = clf.predict(scaler.transform(X_te))
        total = np.mean(preds == y_test) * 100
        oh_a = np.mean(preds[y_test == '오'] == '오') * 100 if (y_test == '오').any() else 0
        oo_a = np.mean(preds[y_test == '우'] == '우') * 100 if (y_test == '우').any() else 0
        results.append((layer, total, oh_a, oo_a))
        marker = " ★" if total >= 65 else ""
        print(f"  Layer {layer:2d}: {total:5.1f}%  오:{oh_a:5.1f}%  우:{oo_a:5.1f}%{marker}")

    # ── 레이어 범위 ──
    print()
    print("=" * 60)
    print("  dataset 학습 → live 테스트 (레이어 범위)")
    print("=" * 60)

    combos = []
    for start in range(n_layers):
        for end in range(start + 1, min(start + 5, n_layers)):
            X_tr = X_train_all[:, start:end+1, :].mean(axis=1)
            X_te = X_test_all[:, start:end+1, :].mean(axis=1)
            scaler = StandardScaler()
            clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
            clf.fit(scaler.fit_transform(X_tr), y_train)
            preds = clf.predict(scaler.transform(X_te))
            total = np.mean(preds == y_test) * 100
            oh_a = np.mean(preds[y_test == '오'] == '오') * 100 if (y_test == '오').any() else 0
            oo_a = np.mean(preds[y_test == '우'] == '우') * 100 if (y_test == '우').any() else 0
            combos.append((f"L{start}-{end}", total, oh_a, oo_a))

    combos.sort(key=lambda x: x[1], reverse=True)
    print("  상위 10:")
    for name, total, oh_a, oo_a in combos[:10]:
        print(f"    {name:8s}: {total:5.1f}%  오:{oh_a:5.1f}%  우:{oo_a:5.1f}%")

    # ── 비교 ──
    best_single = max(results, key=lambda x: x[1])

    X_tr_567 = X_train_all[:, 5:8, :].mean(axis=1)
    X_te_567 = X_test_all[:, 5:8, :].mean(axis=1)
    scaler = StandardScaler()
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    clf.fit(scaler.fit_transform(X_tr_567), y_train)
    preds_cur = clf.predict(scaler.transform(X_te_567))
    cur = np.mean(preds_cur == y_test) * 100
    cur_oh = np.mean(preds_cur[y_test == '오'] == '오') * 100
    cur_oo = np.mean(preds_cur[y_test == '우'] == '우') * 100

    print()
    print("=" * 60)
    print("  비교")
    print("=" * 60)
    print(f"  현재 L5-7:  {cur:5.1f}%  오:{cur_oh:5.1f}%  우:{cur_oo:5.1f}%")
    print(f"  최적 단일:  Layer {best_single[0]}  {best_single[1]:5.1f}%  오:{best_single[2]:5.1f}%  우:{best_single[3]:5.1f}%")
    print(f"  최적 범위:  {combos[0][0]}  {combos[0][1]:5.1f}%  오:{combos[0][2]:5.1f}%  우:{combos[0][3]:5.1f}%")

    # ── 최적으로 화자별 상세 ──
    best_name = combos[0][0]
    parts = best_name.replace('L', '').split('-')
    bs, be = int(parts[0]), int(parts[1])
    X_tr = X_train_all[:, bs:be+1, :].mean(axis=1)
    X_te = X_test_all[:, bs:be+1, :].mean(axis=1)
    scaler = StandardScaler()
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    clf.fit(scaler.fit_transform(X_tr), y_train)
    preds_best = clf.predict(scaler.transform(X_te))

    print()
    print(f"  {best_name} 화자별:")
    for spk in sorted(set(s_test)):
        mask = s_test == spk
        acc = np.mean(preds_best[mask] == y_test[mask]) * 100
        details = []
        for v in TARGET:
            vm = mask & (y_test == v)
            if vm.any():
                va = np.mean(preds_best[vm] == v) * 100
                details.append(f"{v}:{va:.0f}%({vm.sum()})")
        print(f"    {spk:12s}: {acc:5.1f}%  {' '.join(details)}")

    # 현재 L5-7 화자별도 출력
    print()
    print(f"  현재 L5-7 화자별:")
    for spk in sorted(set(s_test)):
        mask = s_test == spk
        acc = np.mean(preds_cur[mask] == y_test[mask]) * 100
        details = []
        for v in TARGET:
            vm = mask & (y_test == v)
            if vm.any():
                va = np.mean(preds_cur[vm] == v) * 100
                details.append(f"{v}:{va:.0f}%({vm.sum()})")
        print(f"    {spk:12s}: {acc:5.1f}%  {' '.join(details)}")


if __name__ == '__main__':
    main()
