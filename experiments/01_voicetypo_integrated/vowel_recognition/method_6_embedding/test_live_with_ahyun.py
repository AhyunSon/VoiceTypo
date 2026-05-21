"""아현 데이터 포함 학습 → live(경상도여성, 20대남성) 테스트.

비교:
  현재: TTS(이은서제외) + remote2(kdg0534,lynn03) → live 테스트
  +아현: TTS(이은서제외) + remote3(kdg0534,lynn03,아현) → live 테스트
"""
import sys, os, io, wave, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def collect_tts_no_eunseo():
    samples = []
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    for f in sorted(os.listdir(DATASET_DIR)):
        if os.path.splitext(f)[1].lower() not in audio_exts:
            continue
        if os.path.isdir(os.path.join(DATASET_DIR, f)):
            continue
        parts = os.path.splitext(f)[0].split('_')
        speaker = parts[2] if len(parts) >= 3 else 'unknown'
        if speaker == '이은서':
            continue
        first = parts[0]
        if first in VOWELS:
            vowel = first
        elif len(first) == 1:
            vowel = syllable_to_vowel(first)
        else:
            continue
        if vowel not in VOWELS:
            continue
        samples.append((os.path.join(DATASET_DIR, f), vowel, f'tts_{speaker}'))
    return samples


def collect_remote(dirs):
    samples = []
    for rd in dirs:
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
            if vowel not in VOWELS:
                continue
            samples.append((os.path.join(d, f), vowel, speaker))
    return samples


def collect_live_test():
    """경상도여성 + 20대남성 오/우."""
    session_map = {
        'session_20260310_151524': '경상도여성',
        'session_20260310_153047': '20대남성',
    }
    samples = []
    for session, speaker in session_map.items():
        d = os.path.join(LIVE_DIR, session)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            vowel = f.split('_')[0]
            if vowel in ['오', '우']:
                samples.append((os.path.join(d, f), vowel, speaker, f))
    return samples


def load_embeddings():
    # 메인 캐시
    cache1 = os.path.join(BASE, 'cache_loso_compare.npz')
    d1 = np.load(cache1, allow_pickle=True)
    emb16 = {str(d1['paths'][i]): d1['emb16'][i] for i in range(len(d1['paths']))}
    emb567 = {str(d1['paths'][i]): d1['emb567'][i] for i in range(len(d1['paths']))}

    # live 캐시 (이전 스크립트에서 추출한 것이 cache에 없을 수 있으므로)
    # test_live15.py에서 이미 추출했으니 직접 추출
    return emb16, emb567


def ensure_live_embeddings(test_samples, emb16, emb567):
    need = [s[0] for s in test_samples if s[0] not in emb16]
    if not need:
        return
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    def load_audio(path):
        with wave.open(path, 'r') as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            ch = wf.getnchannels()
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            a = a.reshape(-1, ch)[:, 0]
        return a, sr

    def pool(frames):
        e = frames.norm(dim=1)
        k = max(1, len(e) // 2)
        return frames[torch.topk(e, k).indices].mean(dim=0).numpy().astype(np.float32)

    print(f'  live 임베딩 추출: {len(need)}개...', flush=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    model.eval()
    for p in need:
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
        emb16[p] = pool(h[16].squeeze(0))
        emb567[p] = (pool(h[5].squeeze(0)) + pool(h[6].squeeze(0)) + pool(h[7].squeeze(0))) / 3.0
    print('  완료.')


def train_and_test(label, train_samples, test_samples, emb16, emb567):
    """학습 → live 테스트."""
    X16 = np.array([emb16[s[0]] for s in train_samples])
    X567 = np.array([emb567[s[0]] for s in train_samples])
    y = np.array([s[1] for s in train_samples])

    # Stage 1
    s1_scaler = StandardScaler()
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(s1_scaler.fit_transform(X16), y)

    # Stage 2
    ou_mask = np.isin(y, ['오', '우'])
    s2_scaler = StandardScaler()
    s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s2_clf.fit(s2_scaler.fit_transform(X567[ou_mask]), y[ou_mask])

    # 테스트
    print(f'\n  {label}')
    print(f'  학습: {len(train_samples)}개 (오/우: {ou_mask.sum()}개)')
    print(f'  {"─"*60}')

    results = []
    for path, gt, speaker, fname in test_samples:
        X1 = s1_scaler.transform(emb16[path].reshape(1, -1))
        p1 = s1_clf.predict(X1)[0]
        proba1 = s1_clf.predict_proba(X1)[0]
        classes1 = s1_clf.classes_

        if p1 in ['오', '우']:
            X2 = s2_scaler.transform(emb567[path].reshape(1, -1))
            pred = s2_clf.predict(X2)[0]
            proba2 = s2_clf.predict_proba(X2)[0]
            conf = float(max(proba2))
            s2_str = f'S2: {" ".join(f"{c}={p:.0%}" for c,p in zip(s2_clf.classes_, proba2))}'
        else:
            pred = p1
            conf = float(max(proba1))
            s2_str = '(S2 미진입)'

        mark = '✓' if pred == gt else '✗'
        top3 = sorted(zip(classes1, proba1), key=lambda x: -x[1])[:3]
        s1_str = ' '.join(f'{c}={p:.0%}' for c, p in top3)
        print(f'    {speaker:8s} {fname:10s} {gt}→{pred} {mark} ({conf:.0%})  S1:[{s1_str}]  {s2_str}')
        results.append((speaker, gt, pred))

    # 요약
    total = len(results)
    correct = sum(1 for _, g, p in results if g == p)
    print(f'\n  전체: {correct}/{total} ({correct/total*100:.1f}%)')
    for spk in sorted(set(s[0] for s in results)):
        sr = [(g, p) for s, g, p in results if s == spk]
        sc = sum(1 for g, p in sr if g == p)
        oh = [(g, p) for g, p in sr if g == '오']
        oo = [(g, p) for g, p in sr if g == '우']
        oh_c = sum(1 for g, p in oh if g == p)
        oo_c = sum(1 for g, p in oo if g == p)
        print(f'    {spk:8s}: {sc}/{len(sr)}  오={oh_c}/{len(oh)} 우={oo_c}/{len(oo)}')

    return correct, total


def main():
    print('=' * 60)
    print('  아현 추가 효과: live(경상도여성, 20대남성) 테스트')
    print('=' * 60)

    tts = collect_tts_no_eunseo()
    remote_2 = collect_remote(REMOTE_DIRS[:2])  # kdg0534, lynn03
    remote_ah = collect_remote(REMOTE_DIRS[2:])  # 아현
    test = collect_live_test()

    emb16, emb567 = load_embeddings()
    ensure_live_embeddings(test, emb16, emb567)

    print(f'\n  TTS(이은서제외): {len(tts)}')
    print(f'  Remote 2명: {len(remote_2)}')
    print(f'  아현: {len(remote_ah)}')
    print(f'  테스트(live): {len(test)}')

    # 조건 1: 현재 (아현 없이)
    train_no_ah = tts + remote_2
    c1, t1 = train_and_test(
        '현재: TTS + remote2 (아현 없음)',
        train_no_ah, test, emb16, emb567)

    # 조건 2: 아현 추가
    train_with_ah = tts + remote_2 + remote_ah
    c2, t2 = train_and_test(
        '+아현: TTS + remote3 (아현 포함)',
        train_with_ah, test, emb16, emb567)

    # 비교
    print(f'\n{"="*60}')
    print(f'  비교')
    print(f'{"="*60}')
    print(f'  아현 없음: {c1}/{t1} ({c1/t1*100:.1f}%)')
    print(f'  아현 포함: {c2}/{t2} ({c2/t2*100:.1f}%)')
    diff = c2 - c1
    sign = '+' if diff > 0 else ''
    print(f'  차이: {sign}{diff}개')


if __name__ == '__main__':
    main()
