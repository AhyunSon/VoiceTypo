"""Stage 2 실험: live 녹음 추가 효과.

Stage 1은 E3 고정.
Stage 2에 live_recordings 오/우 데이터를 추가.

테스트: remote 3명(kdg0534, lynn03, 아현) 전원 LOSO.

비교:
  S2-A: TTS+remote2 (현재)
  S2-L1: TTS+remote2+live (live 전부 추가)
  S2-L2: remote2+live (TTS 제외)
  S2-L3: live만
"""
import sys, os, io, wave, time, hashlib
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, '..', 'dataset')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']
LIVE_DIR = os.path.join(BASE, 'live_recordings')

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
    medial = (code % (28 * 21)) // 28
    return _MEDIAL_TO_VOWEL.get(medial)


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
        if os.path.splitext(f)[1].lower() not in audio_exts:
            continue
        if os.path.isdir(os.path.join(DATASET_DIR, f)):
            continue
        stem = os.path.splitext(f)[0]
        parts = stem.split('_')
        first = parts[0]
        speaker = parts[2] if len(parts) >= 3 else 'unknown'
        if speaker == '이은서':
            continue
        if first in VOWELS:
            vowel = first
        elif len(first) == 1:
            vowel = syllable_to_vowel(first)
        else:
            vowel = None
        if vowel is None or vowel not in VOWELS:
            continue
        samples.append((os.path.join(DATASET_DIR, f), vowel, f'tts_{speaker}', 'tts'))
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
            stem = os.path.splitext(f)[0]
            parts = stem.split('_')
            if len(parts) < 4:
                continue
            speaker = parts[0]
            vowel = parts[2]
            if vowel not in VOWELS:
                continue
            samples.append((os.path.join(d, f), vowel, speaker, 'remote'))
    return samples


def collect_live_ou():
    """live_recordings에서 오/우 파일 수집."""
    # 세션→화자 매핑 (중복 제거)
    session_map = {
        'session_20260310_145230': 'live_서울여성',
        'session_20260310_151524': 'live_경상도여성',  # = speaker_F_20s_gyeongsang
        'session_20260310_153047': 'live_20대남성',    # = speaker_M_20s
    }

    samples = []
    seen_files = set()

    for session, speaker in session_map.items():
        d = os.path.join(LIVE_DIR, session)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            vowel = f.split('_')[0]
            if vowel not in ['오', '우']:
                continue
            path = os.path.join(d, f)
            key = f'{speaker}_{f}'
            if key in seen_files:
                continue
            seen_files.add(key)
            samples.append((path, vowel, speaker, 'live'))

    return samples


def get_embeddings(paths, emb16_map, emb567_map):
    """캐시에 없는 것만 추출."""
    need = [p for p in paths if p not in emb16_map]
    if not need:
        return

    print(f'  live 임베딩 추출: {len(need)}개...', flush=True)
    model_name = 'facebook/wav2vec2-large-xlsr-53'
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
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
            outputs = model(**inputs, output_hidden_states=True)
        h = outputs.hidden_states

        emb16_map[p] = pool(h[16].squeeze(0))
        e5 = pool(h[5].squeeze(0))
        e6 = pool(h[6].squeeze(0))
        e7 = pool(h[7].squeeze(0))
        emb567_map[p] = (e5 + e6 + e7) / 3.0

    print(f'  추출 완료.')


def run_experiment(cond_name, s1_scaler, s1_clf, s2_ou_pool, remote_all, emb16_map, emb567_map):
    """remote 3명 LOSO로 Stage 2 평가."""
    remote_speakers = sorted(set(s[2] for s in remote_all))
    all_results = []
    by_speaker = {}

    oh_cnt = sum(1 for s in s2_ou_pool if s[1] == '오')
    oo_cnt = sum(1 for s in s2_ou_pool if s[1] == '우')
    print(f'\n  {cond_name}')
    print(f'  S2 pool: 오={oh_cnt} 우={oo_cnt} = {len(s2_ou_pool)}개')

    for held in remote_speakers:
        test = [s for s in remote_all if s[2] == held]
        # S2 학습: pool에서 held-out 화자 제거 (remote 오/우도 포함)
        other_remote_ou = [s for s in remote_all if s[2] != held and s[1] in ['오', '우']]
        s2_train = [s for s in s2_ou_pool if s[2] != held]
        # other_remote_ou 중 pool에 없는 것 추가
        pool_paths = set(s[0] for s in s2_train)
        for s in other_remote_ou:
            if s[0] not in pool_paths:
                s2_train.append(s)

        s2_ou = [s for s in s2_train if s[1] in ['오', '우']]
        X2 = np.array([emb567_map[s[0]] for s in s2_ou])
        y2 = np.array([s[1] for s in s2_ou])
        s2_scaler = StandardScaler()
        X2s = s2_scaler.fit_transform(X2)
        s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        s2_clf.fit(X2s, y2)

        speaker_results = []
        for s in test:
            X1t = s1_scaler.transform(emb16_map[s[0]].reshape(1, -1))
            pred1 = s1_clf.predict(X1t)[0]
            if pred1 in ['오', '우']:
                X2t = s2_scaler.transform(emb567_map[s[0]].reshape(1, -1))
                pred = s2_clf.predict(X2t)[0]
                conf = float(max(s2_clf.predict_proba(X2t)[0]))
            else:
                pred = pred1
                conf = float(max(s1_clf.predict_proba(X1t)[0]))
            speaker_results.append((s[1], pred, conf))

        all_results.extend(speaker_results)
        by_speaker[held] = speaker_results

    # 출력
    print(f'  {"─"*55}')
    total_c = sum(1 for g, p, _ in all_results if g == p)
    total_n = len(all_results)
    print(f'  전체: {total_c}/{total_n} ({total_c/total_n*100:.1f}%)')

    for spk in remote_speakers:
        sr = by_speaker[spk]
        sc = sum(1 for g, p, _ in sr if g == p)
        ou = [(g, p) for g, p, _ in sr if g in ['오', '우']]
        ouc = sum(1 for g, p in ou if g == p)
        oh_r = [x for x in ou if x[0] == '오']
        oo_r = [x for x in ou if x[0] == '우']
        oh_c = sum(1 for g, p in oh_r if g == p)
        oo_c = sum(1 for g, p in oo_r if g == p)
        print(f'    {spk:8s}: {sc:2d}/{len(sr):2d}({sc/len(sr)*100:4.0f}%)  '
              f'오/우={ouc}/{len(ou)}  오={oh_c}/{len(oh_r)} 우={oo_c}/{len(oo_r)}')

    return {
        'name': cond_name,
        'total_acc': total_c / total_n * 100,
        'by_speaker': {
            spk: {
                'acc': sum(1 for g, p, _ in sr if g == p) / len(sr) * 100,
                'ou': sum(1 for g, p, _ in sr if g in ['오','우'] and g == p),
                'ou_total': sum(1 for g, p, _ in sr if g in ['오','우']),
                'oh': f"{sum(1 for g,p,_ in sr if g=='오' and p=='오')}/{sum(1 for g,p,_ in sr if g=='오')}",
                'oo': f"{sum(1 for g,p,_ in sr if g=='우' and p=='우')}/{sum(1 for g,p,_ in sr if g=='우')}",
            }
            for spk, sr in by_speaker.items()
        }
    }


def main():
    print('=' * 60)
    print('  Stage 2: live 녹음 추가 효과 실험')
    print('=' * 60)

    tts = collect_tts_no_eunseo()
    remote = collect_remote()
    live_ou = collect_live_ou()

    # 캐시 로드
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}

    # live 임베딩 추출 (캐시에 없으면)
    live_paths = [s[0] for s in live_ou]
    get_embeddings(live_paths, emb16_map, emb567_map)

    # 데이터 요약
    tts_ou = [s for s in tts if s[1] in ['오', '우']]
    remote_ou = [s for s in remote if s[1] in ['오', '우']]
    remote_2 = [s for s in remote if s[2] != '아현']
    remote_2_ou = [s for s in remote_2 if s[1] in ['오', '우']]

    print(f'\n  데이터 요약:')
    print(f'    TTS(이은서제외) 오/우: {len(tts_ou)}개')
    print(f'    Remote 전체 오/우: {len(remote_ou)}개')
    print(f'    Live 오/우: {len(live_ou)}개')

    print(f'\n  Live 화자별:')
    for spk in sorted(set(s[2] for s in live_ou)):
        oh = sum(1 for s in live_ou if s[2] == spk and s[1] == '오')
        oo = sum(1 for s in live_ou if s[2] == spk and s[1] == '우')
        print(f'    {spk:16s}: 오={oh} 우={oo}')

    # Stage 1 학습 (E3 고정)
    print('\n  Stage 1 학습 (E3 고정)...')
    s1_train = tts + remote_2
    X1 = np.array([emb16_map[s[0]] for s in s1_train])
    y1 = np.array([s[1] for s in s1_train])
    s1_scaler = StandardScaler()
    X1s = s1_scaler.fit_transform(X1)
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(X1s, y1)

    # ── 조건별 실험 ──
    results = []

    # S2-A: TTS+remote2 (현재)
    r = run_experiment('S2-A: TTS+remote2 (현재)',
                       s1_scaler, s1_clf,
                       tts_ou + remote_2_ou,
                       remote, emb16_map, emb567_map)
    results.append(r)

    # S2-L1: TTS+remote2+live
    r = run_experiment('S2-L1: TTS+remote2+live',
                       s1_scaler, s1_clf,
                       tts_ou + remote_2_ou + live_ou,
                       remote, emb16_map, emb567_map)
    results.append(r)

    # S2-L2: remote2+live (TTS 제외)
    r = run_experiment('S2-L2: remote2+live (TTS 제외)',
                       s1_scaler, s1_clf,
                       remote_2_ou + live_ou,
                       remote, emb16_map, emb567_map)
    results.append(r)

    # S2-L3: live만
    r = run_experiment('S2-L3: live만',
                       s1_scaler, s1_clf,
                       live_ou,
                       remote, emb16_map, emb567_map)
    results.append(r)

    # S2-L4: remote2+live, TTS 다운샘플(1:1)
    np.random.seed(42)
    n_real = len(remote_2_ou) + len(live_ou)
    tts_oh = [s for s in tts_ou if s[1] == '오']
    tts_oo = [s for s in tts_ou if s[1] == '우']
    n_half = n_real // 2
    ds_oh = [tts_oh[i] for i in np.random.choice(len(tts_oh), min(n_half, len(tts_oh)), replace=False)]
    ds_oo = [tts_oo[i] for i in np.random.choice(len(tts_oo), min(n_half, len(tts_oo)), replace=False)]
    r = run_experiment(f'S2-L5: remote2+live+TTS균형({len(ds_oh)+len(ds_oo)})',
                       s1_scaler, s1_clf,
                       remote_2_ou + live_ou + ds_oh + ds_oo,
                       remote, emb16_map, emb567_map)
    results.append(r)

    # ── 최종 비교 ──
    print(f'\n\n{"#"*60}')
    print(f'  최종 비교 (remote 3명 LOSO)')
    print(f'{"#"*60}\n')

    remote_speakers = ['kdg0534', 'lynn03', '아현']

    print(f'  {"조건":<35s} {"전체":>6s}', end='')
    for spk in remote_speakers:
        print(f'  {spk:>8s}', end='')
    print()
    print(f'  {"─"*35} {"─"*6}', end='')
    for _ in remote_speakers:
        print(f'  {"─"*8}', end='')
    print()

    for r in results:
        name = r['name'][:35]
        print(f'  {name:<35s} {r["total_acc"]:5.1f}%', end='')
        for spk in remote_speakers:
            bs = r['by_speaker'].get(spk, {})
            acc = bs.get('acc', 0)
            print(f'  {acc:6.1f}%', end='')
        print()

    # 오/우 상세
    print(f'\n  오/우 상세:')
    print(f'  {"조건":<35s}', end='')
    for spk in remote_speakers:
        print(f'  {spk+" 오":>6s} {spk+" 우":>6s}', end='')
    print()
    print(f'  {"─"*35}', end='')
    for _ in remote_speakers:
        print(f'  {"─"*6} {"─"*6}', end='')
    print()

    for r in results:
        name = r['name'][:35]
        print(f'  {name:<35s}', end='')
        for spk in remote_speakers:
            bs = r['by_speaker'].get(spk, {})
            print(f'  {bs.get("oh","?"):>6s} {bs.get("oo","?"):>6s}', end='')
        print()


if __name__ == '__main__':
    main()
