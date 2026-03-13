"""현재 모델(E3)로 live 15개(서울여성 제외) 테스트."""
import sys, os, io, wave, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

BASE = os.path.dirname(__file__)
LIVE_DIR = os.path.join(BASE, 'live_recordings')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']


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


def main():
    # 모델 로드
    with open(os.path.join(BASE, 'twostage_model.pkl'), 'rb') as f:
        data = pickle.load(f)
    s1_scaler = data['stage1']['scaler']
    s1_clf = data['stage1']['clf']
    s2_scaler = data['stage2']['scaler']
    s2_clf = data['stage2']['clf']
    s2_target = data['stage2']['target_vowels']

    # XLSR-53
    print('모델 로딩...', flush=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    model.eval()

    # live 파일 수집 (서울여성 제외)
    session_map = {
        'session_20260310_151524': '경상도여성',
        'session_20260310_153047': '20대남성',
    }

    files = []
    for session, speaker in session_map.items():
        d = os.path.join(LIVE_DIR, session)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            vowel = f.split('_')[0]
            if vowel in ['오', '우']:
                files.append((os.path.join(d, f), vowel, speaker, f))

    print(f'\n테스트 파일: {len(files)}개')
    for spk in sorted(set(s[2] for s in files)):
        oh = sum(1 for s in files if s[2] == spk and s[1] == '오')
        oo = sum(1 for s in files if s[2] == spk and s[1] == '우')
        print(f'  {spk}: 오={oh} 우={oo}')

    # 추론
    print(f'\n{"="*55}')
    results = []
    for path, gt, speaker, fname in files:
        audio, sr = load_audio(path)
        if sr != 16000:
            ratio = 16000 / sr
            n_out = int(len(audio) * ratio)
            idx = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(audio) - 1)
            audio = audio[idx]

        inputs = fe(audio, sampling_rate=16000, return_tensors='pt', padding=False)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states

        emb16 = pool(h[16].squeeze(0))
        emb567 = (pool(h[5].squeeze(0)) + pool(h[6].squeeze(0)) + pool(h[7].squeeze(0))) / 3.0

        X1 = s1_scaler.transform(emb16.reshape(1, -1))
        pred1 = s1_clf.predict(X1)[0]
        proba1 = s1_clf.predict_proba(X1)[0]
        classes1 = s1_clf.classes_

        if pred1 in s2_target:
            X2 = s2_scaler.transform(emb567.reshape(1, -1))
            pred2 = s2_clf.predict(X2)[0]
            proba2 = s2_clf.predict_proba(X2)[0]
            final = pred2
            conf = float(max(proba2))
            s2_str = f'S2: {" ".join(f"{c}={p:.0%}" for c,p in zip(s2_clf.classes_, proba2))}'
        else:
            final = pred1
            conf = float(max(proba1))
            s2_str = '(S2 미진입)'

        mark = '✓' if final == gt else '✗'
        top3 = sorted(zip(classes1, proba1), key=lambda x: -x[1])[:3]
        s1_str = ' '.join(f'{c}={p:.0%}' for c, p in top3)

        print(f'  {speaker:8s} {fname:12s} 정답={gt} 예측={final} {mark} ({conf:.0%})  '
              f'S1:[{s1_str}]  {s2_str}')
        results.append((speaker, gt, final, conf))

    # 요약
    print(f'\n{"="*55}')
    total = len(results)
    correct = sum(1 for _, g, p, _ in results if g == p)
    print(f'전체: {correct}/{total} ({correct/total*100:.1f}%)')

    for spk in sorted(set(s[0] for s in results)):
        sr = [(g, p) for s, g, p, _ in results if s == spk]
        sc = sum(1 for g, p in sr if g == p)
        oh = [(g, p) for g, p in sr if g == '오']
        oo = [(g, p) for g, p in sr if g == '우']
        oh_c = sum(1 for g, p in oh if g == p)
        oo_c = sum(1 for g, p in oo if g == p)
        print(f'  {spk:8s}: {sc}/{len(sr)}  오={oh_c}/{len(oh)} 우={oo_c}/{len(oo)}')


if __name__ == '__main__':
    main()
