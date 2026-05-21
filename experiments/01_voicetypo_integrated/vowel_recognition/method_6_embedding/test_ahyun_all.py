"""아현 데이터 종합 테스트: remote(70) + live(20) = 90개.

조건 A(현재), B(+구분용) 비교.
아현은 학습에 미포함 (unseen speaker).
"""
import sys, os, io, wave
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, '..', 'dataset')
OU_EXTRA_DIR = os.path.join(DATASET_DIR, '우 오 구분용')
LIVE_DIR = os.path.join(BASE, 'live_recordings')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}

REMOTE_DIRS = [
    'vowel-remote-001_kdg0534 (1)',
    'vowel-remote-001_lynn03 (1)',
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


def collect_tts_no_eunseo():
    samples = []
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    for f in sorted(os.listdir(DATASET_DIR)):
        fp = os.path.join(DATASET_DIR, f)
        if os.path.isdir(fp):
            continue
        if os.path.splitext(f)[1].lower() not in audio_exts:
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
        samples.append((fp, vowel, f'tts_{speaker}'))
    return samples


def collect_ou_extra():
    samples = []
    if not os.path.isdir(OU_EXTRA_DIR):
        return samples
    for f in sorted(os.listdir(OU_EXTRA_DIR)):
        if not f.endswith('.mp3'):
            continue
        parts = os.path.splitext(f)[0].split('_')
        if len(parts) < 3:
            continue
        vowel = parts[0]
        speaker = parts[2]
        if vowel not in ['오', '우']:
            continue
        samples.append((os.path.join(OU_EXTRA_DIR, f), vowel, f'tts_{speaker}'))
    return samples


def collect_remote2():
    samples = []
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
            if vowel not in VOWELS:
                continue
            samples.append((os.path.join(d, f), vowel, speaker))
    return samples


def collect_ahyun_remote():
    """아현 remote 70개."""
    d = os.path.join(BASE, 'vowel-remote-001_아현 (1)')
    samples = []
    if not os.path.isdir(d):
        return samples
    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        fp = os.path.join(d, f)
        if os.path.isdir(fp):
            continue
        parts = os.path.splitext(f)[0].split('_')
        if len(parts) < 4:
            continue
        vowel = parts[2]
        if vowel not in VOWELS:
            continue
        samples.append((fp, vowel, 'remote', f))
    return samples


def collect_ahyun_live():
    """아현 live 20개."""
    d = os.path.join(LIVE_DIR, 'session_20260310_145230')
    samples = []
    if not os.path.isdir(d):
        return samples
    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        vowel = f.split('_')[0]
        if vowel in VOWELS:
            samples.append((os.path.join(d, f), vowel, 'live', f))
    return samples


def get_embeddings(paths, emb16_map, emb567_map):
    need = [p for p in paths if p not in emb16_map]
    if not need:
        return
    print(f'  임베딩 추출: {len(need)}개...', flush=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    model.eval()
    for i, p in enumerate(need):
        if (i + 1) % 20 == 0:
            print(f'    {i+1}/{len(need)}...', flush=True)
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
    print('  완료.')


def test_condition(name, s1_scaler, s1_clf, s2_scaler, s2_clf, test_samples, emb16_map, emb567_map):
    """테스트 실행 + 상세 출력."""
    results = []
    for item in test_samples:
        path, gt, source = item[0], item[1], item[2]
        fname = item[3] if len(item) > 3 else os.path.basename(path)

        X1 = s1_scaler.transform(emb16_map[path].reshape(1, -1))
        p1 = s1_clf.predict(X1)[0]
        proba1 = s1_clf.predict_proba(X1)[0]
        classes1 = s1_clf.classes_

        if p1 in ['오', '우']:
            X2 = s2_scaler.transform(emb567_map[path].reshape(1, -1))
            pred = s2_clf.predict(X2)[0]
            proba2 = s2_clf.predict_proba(X2)[0]
            conf = float(max(proba2))
            s2_str = f'S2: {" ".join(f"{c}={p:.0%}" for c,p in zip(s2_clf.classes_, proba2))}'
        else:
            pred = p1
            conf = float(max(proba1))
            s2_str = ''

        mark = '✓' if pred == gt else '✗'
        top3 = sorted(zip(classes1, proba1), key=lambda x: -x[1])[:3]
        s1_str = ' '.join(f'{c}={p:.0%}' for c, p in top3)
        print(f'    {source:6s} {fname:20s} {gt}→{pred} {mark} ({conf:.0%})  S1:[{s1_str}]  {s2_str}')
        results.append((source, gt, pred))

    return results


def print_summary(results, label):
    total = len(results)
    correct = sum(1 for _, g, p in results if g == p)
    print(f'\n  [{label}] 전체: {correct}/{total} ({correct/total*100:.1f}%)')

    # 소스별
    for src in ['remote', 'live']:
        sr = [(g, p) for s, g, p in results if s == src]
        if not sr:
            continue
        sc = sum(1 for g, p in sr if g == p)
        print(f'    {src:6s}: {sc}/{len(sr)} ({sc/len(sr)*100:.1f}%)')

    # 모음별
    print(f'    모음별:')
    for v in VOWELS:
        vr = [(g, p) for _, g, p in results if g == v]
        if not vr:
            continue
        vc = sum(1 for g, p in vr if g == p)
        # 오분류 상세
        wrong = [p for g, p in vr if g != p]
        wrong_str = ''
        if wrong:
            from collections import Counter
            wc = Counter(wrong)
            wrong_str = '  오류: ' + ', '.join(f'→{k}({v})' for k, v in wc.most_common())
        print(f'      {v}: {vc}/{len(vr)} ({vc/len(vr)*100:.0f}%){wrong_str}')

    return correct, total


def main():
    print('=' * 65)
    print('  아현 종합 테스트: remote(70) + live(20) = 90개')
    print('  학습에 아현 미포함 (unseen)')
    print('=' * 65)

    # 학습 데이터
    tts = collect_tts_no_eunseo()
    ou_extra = collect_ou_extra()
    remote2 = collect_remote2()

    # 테스트 데이터
    ah_remote = collect_ahyun_remote()
    ah_live = collect_ahyun_live()
    ah_all = [(p, v, src, f) for p, v, src, f in ah_remote] + \
             [(p, v, src, f) for p, v, src, f in ah_live]

    print(f'\n  아현 remote: {len(ah_remote)}개')
    for v in VOWELS:
        n = sum(1 for s in ah_remote if s[1] == v)
        if n > 0:
            print(f'    {v}: {n}')
    print(f'  아현 live: {len(ah_live)}개')
    for v in VOWELS:
        n = sum(1 for s in ah_live if s[1] == v)
        if n > 0:
            print(f'    {v}: {n}')

    # 캐시 로드
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}

    all_paths = [s[0] for s in ou_extra + ah_all]
    get_embeddings(all_paths, emb16_map, emb567_map)

    # Stage 1 (E3)
    print('\n  Stage 1 학습 (E3: TTS이은서제외 + remote2)...')
    s1_all = tts + remote2
    X1 = np.array([emb16_map[s[0]] for s in s1_all])
    y1 = np.array([s[1] for s in s1_all])
    s1_scaler = StandardScaler()
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(s1_scaler.fit_transform(X1), y1)

    tts_ou = [s for s in tts if s[1] in ['오', '우']]
    remote2_ou = [s for s in remote2 if s[1] in ['오', '우']]

    # --- 조건 A: 현재 ---
    print(f'\n{"="*65}')
    print(f'  조건 A: 현재 (TTS+remote2)')
    print(f'{"="*65}')
    s2a_data = tts_ou + remote2_ou
    X2a = np.array([emb567_map[s[0]] for s in s2a_data])
    y2a = np.array([s[1] for s in s2a_data])
    s2a_scaler = StandardScaler()
    s2a_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s2a_clf.fit(s2a_scaler.fit_transform(X2a), y2a)
    res_a = test_condition('A', s1_scaler, s1_clf, s2a_scaler, s2a_clf, ah_all, emb16_map, emb567_map)
    ca, ta = print_summary(res_a, 'A: 현재')

    # --- 조건 B: +구분용 ---
    print(f'\n{"="*65}')
    print(f'  조건 B: +구분용 (TTS+remote2+우오구분용4명)')
    print(f'{"="*65}')
    s2b_data = tts_ou + remote2_ou + ou_extra
    X2b = np.array([emb567_map[s[0]] for s in s2b_data])
    y2b = np.array([s[1] for s in s2b_data])
    s2b_scaler = StandardScaler()
    s2b_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s2b_clf.fit(s2b_scaler.fit_transform(X2b), y2b)
    res_b = test_condition('B', s1_scaler, s1_clf, s2b_scaler, s2b_clf, ah_all, emb16_map, emb567_map)
    cb, tb = print_summary(res_b, 'B: +구분용')

    # --- 비교 ---
    print(f'\n\n{"#"*65}')
    print(f'  종합 비교')
    print(f'{"#"*65}')
    header = '조건'
    print(f'\n  {header:<25s} {"remote(70)":>12s} {"live(20)":>12s} {"전체(90)":>12s}')
    print(f'  {"─"*25} {"─"*12} {"─"*12} {"─"*12}')

    for label, res in [('A: 현재', res_a), ('B: +구분용', res_b)]:
        rem = [(g, p) for s, g, p in res if s == 'remote']
        liv = [(g, p) for s, g, p in res if s == 'live']
        rc = sum(1 for g, p in rem if g == p)
        lc = sum(1 for g, p in liv if g == p)
        tc = rc + lc
        print(f'  {label:<25s} {rc:>3d}/{len(rem):<3d} ({rc/len(rem)*100:4.1f}%)'
              f'  {lc:>3d}/{len(liv):<3d} ({lc/len(liv)*100:4.1f}%)'
              f'  {tc:>3d}/{len(res):<3d} ({tc/len(res)*100:4.1f}%)')

    # 오/우 상세 비교
    print(f'\n  오/우 상세:')
    print(f'  {"조건":<25s} {"오(remote)":>10s} {"우(remote)":>10s} {"오(live)":>10s} {"우(live)":>10s}')
    print(f'  {"─"*25} {"─"*10} {"─"*10} {"─"*10} {"─"*10}')
    for label, res in [('A: 현재', res_a), ('B: +구분용', res_b)]:
        or_ = [(g, p) for s, g, p in res if s == 'remote' and g == '오']
        ur = [(g, p) for s, g, p in res if s == 'remote' and g == '우']
        ol = [(g, p) for s, g, p in res if s == 'live' and g == '오']
        ul = [(g, p) for s, g, p in res if s == 'live' and g == '우']
        orc = sum(1 for g, p in or_ if g == p)
        urc = sum(1 for g, p in ur if g == p)
        olc = sum(1 for g, p in ol if g == p)
        ulc = sum(1 for g, p in ul if g == p)
        print(f'  {label:<25s} {orc:>3d}/{len(or_):<3d}'
              f'      {urc:>3d}/{len(ur):<3d}'
              f'      {olc:>3d}/{len(ol):<3d}'
              f'      {ulc:>3d}/{len(ul):<3d}')


if __name__ == '__main__':
    main()
