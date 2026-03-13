"""Stage 2: live(서울여성 제외) 추가 실험.

Stage 1: E3 고정 (TTS이은서제외 + remote2)
Stage 2 비교:
  S2-A: TTS+remote2 (현재 기준선)
  S2-F: TTS+remote2+live(경상도여성,20대남성)
  S2-G: remote2+live(경상도여성,20대남성) — TTS 제외

평가: remote 3명 LOSO (아현은 unseen)
"""
import sys, os, io, wave, time
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
            parts = os.path.splitext(f)[0].split('_')
            if len(parts) < 4:
                continue
            speaker = parts[0]
            vowel = parts[2]
            if vowel not in VOWELS:
                continue
            samples.append((os.path.join(d, f), vowel, speaker, 'remote'))
    return samples


def collect_live_ou_no_seoul():
    """live에서 서울여성 제외, 경상도여성+20대남성 오/우만."""
    session_map = {
        'session_20260310_151524': 'live_경상도여성',
        'session_20260310_153047': 'live_20대남성',
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
            if vowel not in ['오', '우']:
                continue
            samples.append((os.path.join(d, f), vowel, speaker, 'live'))
    return samples


def get_embeddings(paths, emb16_map, emb567_map):
    need = [p for p in paths if p not in emb16_map]
    if not need:
        return
    print(f'  임베딩 추출: {len(need)}개...', flush=True)
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
        emb16_map[p] = pool(h[16].squeeze(0))
        emb567_map[p] = (pool(h[5].squeeze(0)) + pool(h[6].squeeze(0)) + pool(h[7].squeeze(0))) / 3.0
    print('  완료.')


def run_condition(name, s1_scaler, s1_clf, s2_pool, remote_all, emb16_map, emb567_map):
    """remote 3명 LOSO. Stage 2만 s2_pool로 교체."""
    remote_speakers = sorted(set(s[2] for s in remote_all))
    remote_2 = [s for s in remote_all if s[2] != '아현']
    remote_2_ou = [s for s in remote_2 if s[1] in ['오', '우']]

    all_results = []
    by_speaker = {}

    oh_n = sum(1 for s in s2_pool if s[1] == '오')
    oo_n = sum(1 for s in s2_pool if s[1] == '우')
    print(f'\n  {name}')
    print(f'  S2 pool: 오={oh_n} 우={oo_n} = {len(s2_pool)}개')

    for held in remote_speakers:
        test = [s for s in remote_all if s[2] == held]

        # S2 학습: pool에서 held-out 제거 + 나머지 remote 오/우 보장
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

    # 출력
    total_c = sum(1 for g, p in all_results if g == p)
    print(f'  전체: {total_c}/{len(all_results)} ({total_c/len(all_results)*100:.1f}%)')

    summary = {'name': name}
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
        summary[spk] = {'total': f'{sc}/{len(sr)}', 'acc': sc/len(sr)*100,
                        'oh': f'{oh_c}/{len(oh_r)}', 'oo': f'{oo_c}/{len(oo_r)}',
                        'ou_acc': ouc/len(ou)*100 if ou else 0}
    summary['total_acc'] = total_c/len(all_results)*100
    return summary


def main():
    print('=' * 60)
    print('  Stage 2: live(서울여성 제외) 실험')
    print('=' * 60)

    tts = collect_tts_no_eunseo()
    remote = collect_remote()
    live = collect_live_ou_no_seoul()
    remote_2 = [s for s in remote if s[2] != '아현']

    # 캐시 로드 + live 임베딩 추출
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}
    get_embeddings([s[0] for s in live], emb16_map, emb567_map)

    # 데이터 요약
    tts_ou = [s for s in tts if s[1] in ['오', '우']]
    remote_2_ou = [s for s in remote_2 if s[1] in ['오', '우']]

    print(f'\n  Stage 2 오/우 데이터:')
    print(f'    TTS(이은서제외):  오={sum(1 for s in tts_ou if s[1]=="오")} '
          f'우={sum(1 for s in tts_ou if s[1]=="우")} = {len(tts_ou)}')
    print(f'    Remote 2명:      오={sum(1 for s in remote_2_ou if s[1]=="오")} '
          f'우={sum(1 for s in remote_2_ou if s[1]=="우")} = {len(remote_2_ou)}')
    print(f'    Live(서울여성제외): 오={sum(1 for s in live if s[1]=="오")} '
          f'우={sum(1 for s in live if s[1]=="우")} = {len(live)}')
    for spk in sorted(set(s[2] for s in live)):
        oh = sum(1 for s in live if s[2] == spk and s[1] == '오')
        oo = sum(1 for s in live if s[2] == spk and s[1] == '우')
        print(f'      {spk}: 오={oh} 우={oo}')

    # Stage 1 (E3 고정)
    print('\n  Stage 1 학습 (E3)...')
    s1_all = tts + remote_2
    X1 = np.array([emb16_map[s[0]] for s in s1_all])
    y1 = np.array([s[1] for s in s1_all])
    s1_scaler = StandardScaler()
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(s1_scaler.fit_transform(X1), y1)

    # 조건별 실험
    results = []

    # A: TTS + remote2 (현재)
    results.append(run_condition(
        'S2-A: TTS+remote2 (현재)',
        s1_scaler, s1_clf, tts_ou + remote_2_ou,
        remote, emb16_map, emb567_map))

    # F: TTS + remote2 + live
    results.append(run_condition(
        'S2-F: TTS+remote2+live',
        s1_scaler, s1_clf, tts_ou + remote_2_ou + live,
        remote, emb16_map, emb567_map))

    # G: remote2 + live (TTS 제외)
    results.append(run_condition(
        'S2-G: remote2+live (TTS 대신)',
        s1_scaler, s1_clf, remote_2_ou + live,
        remote, emb16_map, emb567_map))

    # 비교표
    remote_speakers = ['kdg0534', 'lynn03', '아현']
    print(f'\n\n{"#"*65}')
    print(f'  비교 요약')
    print(f'{"#"*65}\n')

    print(f'  {"조건":<28s} {"전체":>6s}', end='')
    for spk in remote_speakers:
        print(f'  {spk:>8s}', end='')
    print()
    print(f'  {"─"*28} {"─"*6}  {"─"*8}  {"─"*8}  {"─"*8}')
    for r in results:
        print(f'  {r["name"]:<28s} {r["total_acc"]:5.1f}%', end='')
        for spk in remote_speakers:
            print(f'  {r[spk]["acc"]:6.1f}%', end='')
        print()

    print(f'\n  오/우 상세:')
    print(f'  {"조건":<28s}', end='')
    for spk in remote_speakers:
        print(f'  {spk+"오":>7s} {spk+"우":>7s}', end='')
    print()
    print(f'  {"─"*28}', end='')
    for _ in remote_speakers:
        print(f'  {"─"*7} {"─"*7}', end='')
    print()
    for r in results:
        print(f'  {r["name"]:<28s}', end='')
        for spk in remote_speakers:
            print(f'  {r[spk]["oh"]:>7s} {r[spk]["oo"]:>7s}', end='')
        print()

    # LOSO 일반화 지표 (아현 제외, kdg0534+lynn03 평균)
    print(f'\n  일반화 지표 (kdg0534+lynn03 평균):')
    for r in results:
        avg = (r['kdg0534']['acc'] + r['lynn03']['acc']) / 2
        avg_ou = (r['kdg0534']['ou_acc'] + r['lynn03']['ou_acc']) / 2
        print(f'    {r["name"]:<28s}  전체={avg:.1f}%  오/우={avg_ou:.1f}%')


if __name__ == '__main__':
    main()
