"""Stage 2에 '우 오 구분용' TTS 추가 실험.

비교:
  S2-현재: TTS(Anna+김동규) + remote2  오/우
  S2-추가: TTS(Anna+김동규) + remote2 + 우오구분용(4명) 오/우

평가:
  1) Remote 3명 LOSO (아현 unseen)
  2) Live 15개 (경상도여성, 20대남성)
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
        samples.append((fp, vowel, f'tts_{speaker}', 'tts'))
    return samples


def collect_ou_extra():
    """우 오 구분용 폴더의 데이터 수집."""
    samples = []
    if not os.path.isdir(OU_EXTRA_DIR):
        print(f'  경고: {OU_EXTRA_DIR} 없음')
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
        samples.append((os.path.join(OU_EXTRA_DIR, f), vowel, f'tts_{speaker}', 'tts_ou_extra'))
    return samples


def collect_remote():
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
            samples.append((os.path.join(d, f), vowel, speaker, 'remote'))
    return samples


def collect_live_test():
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


def collect_ahyun_live_test():
    """아현(서울여성) live 세션 — 모든 모음."""
    d = os.path.join(LIVE_DIR, 'session_20260310_145230')
    samples = []
    if not os.path.isdir(d):
        return samples
    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        vowel = f.split('_')[0]
        if vowel in VOWELS:
            samples.append((os.path.join(d, f), vowel, '아현(live)', f))
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


def run_loso(name, s1_scaler, s1_clf, s2_pool, remote_all, emb16_map, emb567_map):
    """Remote 3명 LOSO. Stage 2만 s2_pool로 교체."""
    remote_speakers = sorted(set(s[2] for s in remote_all))
    remote_2 = [s for s in remote_all if s[2] != '아현']
    remote_2_ou = [s for s in remote_2 if s[1] in ['오', '우']]

    all_results = []
    by_speaker = {}

    for held in remote_speakers:
        test = [s for s in remote_all if s[2] == held]
        s2_train = [s for s in s2_pool if s[2] != held]
        train_paths = set(s[0] for s in s2_train)
        for s in remote_2_ou:
            if s[2] != held and s[0] not in train_paths:
                s2_train.append(s)

        s2_ou = [s for s in s2_train if s[1] in ['오', '우']]
        X2 = np.array([emb567_map[s[0]] for s in s2_ou])
        y2 = np.array([s[1] for s in s2_ou])
        s2_scaler = StandardScaler()
        s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        s2_clf.fit(s2_scaler.fit_transform(X2), y2)

        sr = []
        for s in test:
            X1t = s1_scaler.transform(emb16_map[s[0]].reshape(1, -1))
            p1 = s1_clf.predict(X1t)[0]
            if p1 in ['오', '우']:
                X2t = s2_scaler.transform(emb567_map[s[0]].reshape(1, -1))
                pred = s2_clf.predict(X2t)[0]
            else:
                pred = p1
            sr.append((s[1], pred))
        all_results.extend(sr)
        by_speaker[held] = sr

    total_c = sum(1 for g, p in all_results if g == p)
    print(f'\n  [{name}] LOSO 결과')
    print(f'  전체: {total_c}/{len(all_results)} ({total_c/len(all_results)*100:.1f}%)')

    for spk in remote_speakers:
        sr = by_speaker[spk]
        sc = sum(1 for g, p in sr if g == p)
        ou = [(g, p) for g, p in sr if g in ['오', '우']]
        ouc = sum(1 for g, p in ou if g == p)
        oh_r = [x for x in ou if x[0] == '오']
        oo_r = [x for x in ou if x[0] == '우']
        oh_c = sum(1 for g, p in oh_r if g == p)
        oo_c = sum(1 for g, p in oo_r if g == p)
        print(f'    {spk:8s}: {sc:2d}/{len(sr)} ({sc/len(sr)*100:4.0f}%)  '
              f'오/우={ouc}/{len(ou)}  오={oh_c}/{len(oh_r)} 우={oo_c}/{len(oo_r)}')

    return total_c, len(all_results)


def run_live_test(name, s1_scaler, s1_clf, s2_pool, live_test, emb16_map, emb567_map):
    """Live 테스트."""
    s2_ou = [s for s in s2_pool if s[1] in ['오', '우']]
    X2 = np.array([emb567_map[s[0]] for s in s2_ou])
    y2 = np.array([s[1] for s in s2_ou])
    s2_scaler = StandardScaler()
    s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s2_clf.fit(s2_scaler.fit_transform(X2), y2)

    print(f'\n  [{name}] Live 테스트')
    results = []
    for path, gt, speaker, fname in live_test:
        X1 = s1_scaler.transform(emb16_map[path].reshape(1, -1))
        p1 = s1_clf.predict(X1)[0]
        proba1 = s1_clf.predict_proba(X1)[0]
        classes1 = s1_clf.classes_

        if p1 in ['오', '우']:
            X2t = s2_scaler.transform(emb567_map[path].reshape(1, -1))
            pred = s2_clf.predict(X2t)[0]
            proba2 = s2_clf.predict_proba(X2t)[0]
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

    total = len(results)
    correct = sum(1 for _, g, p in results if g == p)
    print(f'  전체: {correct}/{total} ({correct/total*100:.1f}%)')
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
    print('  Stage 2: 우오구분용 TTS 추가 실험')
    print('=' * 60)

    tts = collect_tts_no_eunseo()
    ou_extra = collect_ou_extra()
    remote = collect_remote()
    remote_2 = [s for s in remote if s[2] != '아현']
    live_test = collect_live_test()
    ahyun_test = collect_ahyun_live_test()

    # 임베딩 캐시 로드
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}

    # 새 데이터 임베딩 추출
    all_paths = [s[0] for s in ou_extra + live_test + ahyun_test]
    get_embeddings(all_paths, emb16_map, emb567_map)

    # 데이터 요약
    tts_ou = [s for s in tts if s[1] in ['오', '우']]
    remote_2_ou = [s for s in remote_2 if s[1] in ['오', '우']]

    print(f'\n  데이터 요약:')
    print(f'    TTS(Anna+김동규) 오/우: {len(tts_ou)}개')
    print(f'    우오구분용(4명) 오/우: {len(ou_extra)}개')
    for spk in sorted(set(s[2] for s in ou_extra)):
        oh = sum(1 for s in ou_extra if s[2] == spk and s[1] == '오')
        oo = sum(1 for s in ou_extra if s[2] == spk and s[1] == '우')
        print(f'      {spk}: 오={oh} 우={oo}')
    print(f'    Remote 2명 오/우: {len(remote_2_ou)}개')
    print(f'    Live 테스트: {len(live_test)}개')
    print(f'    아현 Live 테스트: {len(ahyun_test)}개')

    # Stage 1 (E3 고정)
    print('\n  Stage 1 학습 (E3)...')
    s1_all = tts + remote_2
    X1 = np.array([emb16_map[s[0]] for s in s1_all])
    y1 = np.array([s[1] for s in s1_all])
    s1_scaler = StandardScaler()
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(s1_scaler.fit_transform(X1), y1)

    # 조건 A: 현재 (TTS + remote2)
    pool_a = tts_ou + remote_2_ou

    # 조건 B: TTS + remote2 + 우오구분용
    pool_b = tts_ou + remote_2_ou + ou_extra

    # 조건 C: remote2 + 우오구분용 (기존 TTS 오/우 제외)
    pool_c = remote_2_ou + ou_extra

    print(f'\n  S2 풀 크기:')
    print(f'    A(현재): {len(pool_a)}')
    print(f'    B(+구분용): {len(pool_b)}')
    print(f'    C(구분용만): {len(pool_c)}')

    # LOSO 테스트
    print(f'\n{"="*60}')
    print(f'  LOSO (Remote 3명)')
    print(f'{"="*60}')
    loso_a = run_loso('A: 현재', s1_scaler, s1_clf, pool_a, remote, emb16_map, emb567_map)
    loso_b = run_loso('B: +구분용', s1_scaler, s1_clf, pool_b, remote, emb16_map, emb567_map)
    loso_c = run_loso('C: 구분용만', s1_scaler, s1_clf, pool_c, remote, emb16_map, emb567_map)

    # Live 테스트
    print(f'\n{"="*60}')
    print(f'  Live 테스트 (경상도여성, 20대남성)')
    print(f'{"="*60}')
    live_a = run_live_test('A: 현재', s1_scaler, s1_clf, pool_a, live_test, emb16_map, emb567_map)
    live_b = run_live_test('B: +구분용', s1_scaler, s1_clf, pool_b, live_test, emb16_map, emb567_map)
    live_c = run_live_test('C: 구분용만', s1_scaler, s1_clf, pool_c, live_test, emb16_map, emb567_map)

    # 아현 Live 테스트
    print(f'\n{"="*60}')
    print(f'  아현 Live 테스트 (서울여성 세션)')
    print(f'{"="*60}')
    ah_a = run_live_test('A: 현재', s1_scaler, s1_clf, pool_a, ahyun_test, emb16_map, emb567_map)
    ah_b = run_live_test('B: +구분용', s1_scaler, s1_clf, pool_b, ahyun_test, emb16_map, emb567_map)
    ah_c = run_live_test('C: 구분용만', s1_scaler, s1_clf, pool_c, ahyun_test, emb16_map, emb567_map)

    # 비교표
    print(f'\n\n{"#"*60}')
    print(f'  최종 비교')
    print(f'{"#"*60}')
    header = '조건'
    print(f'\n  {header:<20s} {"LOSO":>10s} {"Live(15)":>12s} {"아현(20)":>12s}')
    print(f'  {"─"*20} {"─"*10} {"─"*12} {"─"*12}')
    for name, loso, live, ah in [
        ('A: 현재', loso_a, live_a, ah_a),
        ('B: +구분용', loso_b, live_b, ah_b),
        ('C: 구분용만', loso_c, live_c, ah_c),
    ]:
        print(f'  {name:<20s} {loso[0]:>2d}/{loso[1]:<3d} ({loso[0]/loso[1]*100:4.1f}%)'
              f'  {live[0]:>2d}/{live[1]:<3d} ({live[0]/live[1]*100:4.1f}%)'
              f'  {ah[0]:>2d}/{ah[1]:<3d} ({ah[0]/ah[1]*100:4.1f}%)')


if __name__ == '__main__':
    main()
