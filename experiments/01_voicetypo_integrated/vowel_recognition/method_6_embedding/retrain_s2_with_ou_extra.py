"""Stage 2에 우오구분용 TTS 4명 추가하여 모델 재학습.

Stage 1: E3 그대로 (TTS이은서제외 + remote2)
Stage 2: TTS(Anna+김동규) + remote2 + 우오구분용(박승준,김유진,유영현,윤서연) 오/우
"""
import sys, os, io, pickle
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, '..', 'dataset')
OU_EXTRA_DIR = os.path.join(DATASET_DIR, '우 오 구분용')
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
        import wave
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
        samples.append((fp, vowel, speaker))
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
        samples.append((os.path.join(OU_EXTRA_DIR, f), vowel, speaker))
    return samples


def main():
    print('=' * 60)
    print('  Stage 2 재학습: +우오구분용 TTS')
    print('=' * 60)

    tts = collect_tts_no_eunseo()
    remote2 = collect_remote2()
    ou_extra = collect_ou_extra()

    # 캐시 로드
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}

    # 우오구분용 임베딩 추출
    need = [s[0] for s in ou_extra if s[0] not in emb567_map]
    if need:
        print(f'\n  우오구분용 임베딩 추출: {len(need)}개...', flush=True)
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

    # === Stage 1: E3 그대로 ===
    print('\n[Stage 1] E3 재학습 (TTS이은서제외 + remote2)...')
    s1_all = tts + remote2
    X1_all = np.array([emb16_map[s[0]] for s in s1_all])
    y1_all = np.array([s[1] for s in s1_all])

    s1_scaler = StandardScaler()
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(s1_scaler.fit_transform(X1_all), y1_all)

    s1_preds = s1_clf.predict(s1_scaler.transform(X1_all))
    print(f'  학습 정확도: {np.mean(s1_preds == y1_all)*100:.1f}% ({len(s1_all)}개)')

    # === Stage 2: TTS오우 + remote2오우 + 우오구분용 ===
    print('\n[Stage 2] TTS + remote2 + 우오구분용 오/우 학습...')
    tts_ou = [(s[0], s[1], s[2]) for s in tts if s[1] in ['오', '우']]
    remote2_ou = [(s[0], s[1], s[2]) for s in remote2 if s[1] in ['오', '우']]
    s2_all = tts_ou + remote2_ou + ou_extra

    oh_n = sum(1 for s in s2_all if s[1] == '오')
    oo_n = sum(1 for s in s2_all if s[1] == '우')
    print(f'  오: {oh_n}, 우: {oo_n}, 합계: {len(s2_all)}')

    # 화자 상세
    speakers = sorted(set(s[2] for s in s2_all))
    for spk in speakers:
        oh = sum(1 for s in s2_all if s[2] == spk and s[1] == '오')
        oo = sum(1 for s in s2_all if s[2] == spk and s[1] == '우')
        print(f'    {spk}: 오={oh} 우={oo}')

    X2_all = np.array([emb567_map[s[0]] for s in s2_all])
    y2_all = np.array([s[1] for s in s2_all])

    s2_scaler = StandardScaler()
    s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s2_clf.fit(s2_scaler.fit_transform(X2_all), y2_all)

    s2_preds = s2_clf.predict(s2_scaler.transform(X2_all))
    print(f'  학습 정확도: {np.mean(s2_preds == y2_all)*100:.1f}%')
    print(f'  오: {np.mean(s2_preds[y2_all=="오"]=="오")*100:.0f}%  '
          f'우: {np.mean(s2_preds[y2_all=="우"]=="우")*100:.0f}%')

    # === 백업 + 저장 ===
    model_path = os.path.join(BASE, 'twostage_model.pkl')
    backup_path = os.path.join(BASE, 'twostage_model_e3_s2a.pkl')

    if os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f'\n  기존 모델 백업: {backup_path}')

    model_data = {
        'stage1': {
            'scaler': s1_scaler,
            'clf': s1_clf,
            'layers': [16],
            'description': 'XLSR-53 Layer 16 -> SVM (7 vowels)',
        },
        'stage2': {
            'scaler': s2_scaler,
            'clf': s2_clf,
            'layers': [5, 6, 7],
            'target_vowels': ['오', '우'],
            'description': 'XLSR-53 Layer 5-7 mean -> SVM (오/우 binary) + 우오구분용TTS',
        },
        'model_name': 'facebook/wav2vec2-large-xlsr-53',
        'n_train_s1': len(s1_all),
        'n_train_s2': len(s2_all),
        'config': 'E3+B: S1=TTS(이은서제외)+remote2, S2=TTS오우+remote2오우+우오구분용4명',
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f'\n  모델 저장: {model_path}')
    print(f'  파일 크기: {os.path.getsize(model_path) / 1024:.0f} KB')
    print(f'\n{"="*60}')
    print('  완료!')
    print(f'  S1: {len(s1_all)}개 (E3)')
    print(f'  S2: {len(s2_all)}개 (오{oh_n}+우{oo_n}, 구분용4명 포함)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
