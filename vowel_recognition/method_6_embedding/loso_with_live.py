"""LOSO 평가: 기존 dataset + live recordings, 화자 하나씩 hold-out."""
import sys, io, os, time, wave
import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
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


def main():
    # ── 데이터 수집 ──
    files = []

    # 1) 기존 dataset 오/우
    for f in sorted(os.listdir(DATASET)):
        full = os.path.join(DATASET, f)
        if os.path.isdir(full):
            continue
        v = parse_vowel(f)
        if v in TARGET:
            files.append((full, v, parse_speaker(f)))

    # 2) live_recordings 오/우
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
                files.append((os.path.join(d, f), v, spk))

    print(f'전체 오/우 데이터: {len(files)}개')
    speakers = sorted(set(s for _, _, s in files))
    for s in speakers:
        oh = sum(1 for _, v, sp in files if sp == s and v == '오')
        oo = sum(1 for _, v, sp in files if sp == s and v == '우')
        print(f'  {s:12s}: 오={oh:2d}  우={oo:2d}')

    # ── 임베딩 추출 (캐시 활용) ──
    cache_path = os.path.join(BASE, 'retrain_stage2_cache.npz')
    cached = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        for i, p in enumerate(data['paths']):
            cached[str(p)] = data['emb2'][i]
        print(f'\n캐시에서 {len(cached)}개 로드')

    need = [(p, v, s) for p, v, s in files if str(p) not in cached]
    if need:
        print(f'새로 추출: {len(need)}개')
        fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        model.eval()
        for i, (p, v, s) in enumerate(need):
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
            emb = (pool(h[5].squeeze(0)) + pool(h[6].squeeze(0)) + pool(h[7].squeeze(0))) / 3.0
            cached[str(p)] = emb
            if (i + 1) % 20 == 0:
                print(f'  [{i+1}/{len(need)}]', flush=True)
        print('추출 완료')

        # 캐시 업데이트
        all_paths = list(cached.keys())
        # emb1은 없으므로 dummy
        all_emb2 = np.array([cached[p] for p in all_paths])
        # 기존 캐시에 emb1이 있으면 유지
        if os.path.exists(cache_path):
            old = np.load(cache_path, allow_pickle=True)
            old_paths = list(old['paths'])
            old_emb1 = old['emb1']
            emb1_map = {str(p): old_emb1[i] for i, p in enumerate(old_paths)}
        else:
            emb1_map = {}
        all_emb1 = np.array([emb1_map.get(p, np.zeros(1024, dtype=np.float32)) for p in all_paths])
        np.savez(cache_path, paths=np.array(all_paths, dtype=object),
                 emb1=all_emb1, emb2=all_emb2)

    X = np.array([cached[str(p)] for p, _, _ in files])
    y = np.array([v for _, v, _ in files])
    spk = np.array([s for _, _, s in files])

    # ── LOSO 평가 ──
    print()
    print('=' * 60)
    print('  LOSO: 화자 하나씩 빼고 학습 → 테스트')
    print('=' * 60)

    all_gt, all_pred = [], []
    for held in speakers:
        test_mask = spk == held
        train_mask = ~test_mask

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])

        clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        clf.fit(X_tr, y[train_mask])
        preds = clf.predict(X_te)
        y_te = y[test_mask]

        acc = np.mean(preds == y_te) * 100
        oh_m = y_te == '오'
        oo_m = y_te == '우'
        oh_a = np.mean(preds[oh_m] == '오') * 100 if oh_m.any() else -1
        oo_a = np.mean(preds[oo_m] == '우') * 100 if oo_m.any() else -1

        parts = []
        if oh_m.any():
            parts.append(f'오:{oh_a:.0f}%({oh_m.sum()}개)')
        if oo_m.any():
            parts.append(f'우:{oo_a:.0f}%({oo_m.sum()}개)')
        print(f'  {held:12s}: {acc:5.1f}%  {" ".join(parts)}')

        all_gt.extend(y_te)
        all_pred.extend(preds)

    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)
    total = np.mean(all_gt == all_pred) * 100
    oh_t = np.mean(all_pred[all_gt == '오'] == '오') * 100
    oo_t = np.mean(all_pred[all_gt == '우'] == '우') * 100
    cm = confusion_matrix(all_gt, all_pred, labels=['오', '우'])
    print()
    print(f'  전체 LOSO: {total:.1f}%')
    print(f'  오: {oh_t:.1f}%  우: {oo_t:.1f}%')
    print(f'  혼동행렬:')
    print(f'           예측:오  예측:우')
    print(f'    정답:오  {cm[0,0]:4d}    {cm[0,1]:4d}')
    print(f'    정답:우  {cm[1,0]:4d}    {cm[1,1]:4d}')


if __name__ == '__main__':
    main()
