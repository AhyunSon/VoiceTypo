"""현재 학습된 twostage_model.pkl로 remote 녹음 데이터를 테스트.

사용법:
  python test_remote.py
"""
import sys, os, io, wave, time, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

BASE = os.path.dirname(__file__)
VOWELS = ['아', '어', '오', '우', '으', '이', '에']

# remote 폴더들
REMOTE_DIRS = [
    'vowel-remote-001_kdg0534 (1)',
    'vowel-remote-001_lynn03 (1)',
    'vowel-remote-001_아현 (1)',
]


def load_audio(path):
    with wave.open(path, 'r') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        ch = wf.getnchannels()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return a, sr


def pool(frames):
    e = frames.norm(dim=1)
    k = max(1, len(e) // 2)
    return frames[torch.topk(e, k).indices].mean(dim=0).numpy().astype(np.float32)


def parse_remote_file(filename):
    """화자_번호_모음_음절_길이.wav → (speaker, vowel, syllable, duration)"""
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    if len(parts) < 4:
        return None
    speaker = parts[0]
    vowel = parts[2]
    syllable = parts[3]
    duration = parts[4] if len(parts) >= 5 else '?'
    if vowel not in VOWELS:
        return None
    return speaker, vowel, syllable, duration


def main():
    model_path = os.path.join(BASE, 'twostage_model.pkl')
    if not os.path.exists(model_path):
        print(f'모델 파일 없음: {model_path}')
        sys.exit(1)

    # 분류기 로드
    print('분류기 로딩...', flush=True)
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    s1_scaler = data['stage1']['scaler']
    s1_clf = data['stage1']['clf']
    s2_scaler = data['stage2']['scaler']
    s2_clf = data['stage2']['clf']
    s2_target = data['stage2']['target_vowels']
    print(f'  Stage1 classes: {list(s1_clf.classes_)}')
    print(f'  Stage2 target: {s2_target}, classes: {list(s2_clf.classes_)}')

    # XLSR-53 로드
    print('XLSR-53 로딩...', flush=True)
    model_name = 'facebook/wav2vec2-large-xlsr-53'
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    print('모델 로드 완료.\n')

    # 데이터 수집
    all_files = []
    for rd in REMOTE_DIRS:
        d = os.path.join(BASE, rd)
        if not os.path.isdir(d):
            print(f'  [경고] 폴더 없음: {rd}')
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            info = parse_remote_file(f)
            if info is None:
                continue
            speaker, vowel, syllable, duration = info
            all_files.append((os.path.join(d, f), speaker, vowel, syllable, duration))

    print(f'전체 테스트 파일: {len(all_files)}개')
    speakers = sorted(set(s for _, s, _, _, _ in all_files))
    for spk in speakers:
        counts = {}
        for _, s, v, _, _ in all_files:
            if s == spk:
                counts[v] = counts.get(v, 0) + 1
        cnt_str = '  '.join(f'{v}:{counts.get(v, 0)}' for v in VOWELS)
        print(f'  {spk:12s}: {cnt_str}  (총 {sum(counts.values())})')

    # 추론
    print(f'\n{"="*70}')
    print('  현재 모델로 remote 데이터 테스트')
    print(f'{"="*70}\n')

    results = []  # (speaker, gt_vowel, pred_vowel, confidence, syllable, duration)
    t_total = time.perf_counter()

    for i, (path, speaker, gt_vowel, syllable, duration) in enumerate(all_files):
        audio, sr = load_audio(path)

        # 리샘플링 to 16kHz
        if sr != 16000:
            ratio = 16000 / sr
            n_out = int(len(audio) * ratio)
            idx = np.clip((np.arange(n_out) / ratio).astype(int), 0, len(audio) - 1)
            audio = audio[idx]

        inputs = fe(audio, sampling_rate=16000, return_tensors='pt', padding=False)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states

        emb16 = pool(hidden[16].squeeze(0))
        emb5 = pool(hidden[5].squeeze(0))
        emb6 = pool(hidden[6].squeeze(0))
        emb7 = pool(hidden[7].squeeze(0))
        emb567 = (emb5 + emb6 + emb7) / 3.0

        # Stage 1
        X1 = s1_scaler.transform(emb16.reshape(1, -1))
        pred1 = s1_clf.predict(X1)[0]
        proba1 = s1_clf.predict_proba(X1)[0]

        # Stage 2
        if pred1 in s2_target:
            X2 = s2_scaler.transform(emb567.reshape(1, -1))
            pred2 = s2_clf.predict(X2)[0]
            proba2 = s2_clf.predict_proba(X2)[0]
            final = pred2
            conf = float(max(proba2))
        else:
            final = pred1
            conf = float(max(proba1))

        correct = '✓' if final == gt_vowel else '✗'
        results.append((speaker, gt_vowel, final, conf, syllable, duration))

        if (i + 1) % 20 == 0 or i == 0:
            print(f'  [{i+1:3d}/{len(all_files)}] {speaker:8s} {gt_vowel}→{final} {correct}  '
                  f'({conf:.0%})  {syllable}_{duration}', flush=True)

    elapsed = time.perf_counter() - t_total
    print(f'\n추론 완료: {elapsed:.1f}초 (평균 {elapsed/len(all_files)*1000:.0f}ms/파일)\n')

    # ── 결과 요약 ──
    print(f'{"="*70}')
    print('  결과 요약')
    print(f'{"="*70}\n')

    total = len(results)
    correct_cnt = sum(1 for r in results if r[1] == r[2])
    print(f'전체 정확도: {correct_cnt}/{total} ({correct_cnt/total*100:.1f}%)\n')

    # 화자별
    print(f'화자별 정확도:')
    for spk in speakers:
        spk_results = [r for r in results if r[0] == spk]
        spk_correct = sum(1 for r in spk_results if r[1] == r[2])
        spk_total = len(spk_results)
        print(f'  {spk:12s}: {spk_correct:2d}/{spk_total:2d} ({spk_correct/spk_total*100:5.1f}%)')

    # 모음별
    print(f'\n모음별 정확도:')
    for v in VOWELS:
        v_results = [r for r in results if r[1] == v]
        if not v_results:
            continue
        v_correct = sum(1 for r in v_results if r[1] == r[2])
        v_total = len(v_results)
        errors = {}
        for r in v_results:
            if r[1] != r[2]:
                errors[r[2]] = errors.get(r[2], 0) + 1
        err_str = ', '.join(f'{k}({v})' for k, v in sorted(errors.items(), key=lambda x: -x[1])) if errors else '없음'
        print(f'  {v}: {v_correct:2d}/{v_total:2d} ({v_correct/v_total*100:5.1f}%)  오인: {err_str}')

    # 혼동행렬
    print(f'\n혼동행렬:')
    header = '정답\\예측'
    print(f'  {header:>8s}', end='')
    for v in VOWELS:
        print(f'  {v:>4s}', end='')
    print('   정확도')

    for v in VOWELS:
        v_results = [r for r in results if r[1] == v]
        if not v_results:
            continue
        print(f'  {v:>8s}', end='')
        v_total = len(v_results)
        for v2 in VOWELS:
            cnt = sum(1 for r in v_results if r[2] == v2)
            if cnt == 0:
                print(f'  {"·":>4s}', end='')
            elif v == v2:
                print(f'  \033[92m{cnt:>4d}\033[0m', end='')
            else:
                print(f'  \033[91m{cnt:>4d}\033[0m', end='')
        v_correct = sum(1 for r in v_results if r[1] == r[2])
        print(f'  {v_correct/v_total*100:5.1f}%')

    # 화자×모음 상세
    print(f'\n화자×모음 상세:')
    for spk in speakers:
        print(f'\n  [{spk}]')
        for v in VOWELS:
            sv_results = [r for r in results if r[0] == spk and r[1] == v]
            if not sv_results:
                continue
            sv_correct = sum(1 for r in sv_results if r[1] == r[2])
            sv_total = len(sv_results)
            wrongs = [f'{r[4]}_{r[5]}→{r[2]}' for r in sv_results if r[1] != r[2]]
            wrong_str = f'  오인: {", ".join(wrongs)}' if wrongs else ''
            print(f'    {v}: {sv_correct}/{sv_total}{wrong_str}')

    # 길이(short/normal/long)별 정확도
    print(f'\n길이별 정확도:')
    for dur in ['short', 'normal', 'long']:
        d_results = [r for r in results if r[5] == dur]
        if not d_results:
            continue
        d_correct = sum(1 for r in d_results if r[1] == r[2])
        d_total = len(d_results)
        print(f'  {dur:8s}: {d_correct:2d}/{d_total:2d} ({d_correct/d_total*100:5.1f}%)')


if __name__ == '__main__':
    main()
