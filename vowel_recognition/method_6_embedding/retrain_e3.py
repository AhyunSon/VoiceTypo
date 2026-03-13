"""E3 구성으로 twostage_model.pkl 재학습.

학습셋: TTS 이은서 제외(Anna+김동규 361) + remote 2명(kdg0534+lynn03 140) = 501개
아현은 제외 (라이브 테스트용 unseen speaker)
"""
import sys, os, io, pickle, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, '..', 'dataset')
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
    medial = (code % (28 * 21)) // 28
    return _MEDIAL_TO_VOWEL.get(medial)


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

        # 이은서 제외
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
        samples.append((os.path.join(DATASET_DIR, f), vowel, speaker))
    return samples


def collect_remote_2():
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
            samples.append((os.path.join(d, f), vowel, speaker))
    return samples


def main():
    print('=' * 60)
    print('  E3 모델 재학습')
    print('  TTS(이은서 제외) + remote 2명(kdg0534, lynn03)')
    print('=' * 60)

    tts = collect_tts_no_eunseo()
    remote = collect_remote_2()
    all_samples = tts + remote

    print(f'\n  TTS (이은서 제외): {len(tts)}개')
    tts_speakers = sorted(set(s[2] for s in tts))
    for spk in tts_speakers:
        cnt = sum(1 for s in tts if s[2] == spk)
        print(f'    {spk}: {cnt}')

    print(f'  Remote: {len(remote)}개')
    remote_speakers = sorted(set(s[2] for s in remote))
    for spk in remote_speakers:
        cnt = sum(1 for s in remote if s[2] == spk)
        print(f'    {spk}: {cnt}')

    print(f'  전체: {len(all_samples)}개')

    # 모음 분포
    vowel_counts = {}
    for _, v, _ in all_samples:
        vowel_counts[v] = vowel_counts.get(v, 0) + 1
    print(f'  모음: {" ".join(f"{v}:{vowel_counts.get(v,0)}" for v in VOWELS)}')

    # 캐시된 임베딩 로드
    print('\n  임베딩 로드...')
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}

    # 학습 데이터 조립
    train_paths = [s[0] for s in all_samples]
    train_labels = np.array([s[1] for s in all_samples])

    X_emb16 = np.array([emb16_map[p] for p in train_paths])
    X_emb567 = np.array([emb567_map[p] for p in train_paths])

    print(f'  임베딩 shape: emb16={X_emb16.shape}, emb567={X_emb567.shape}')

    # ── Stage 1: Layer 16 → 7모음 SVM ──
    print('\n[Stage 1] Layer 16 전체 7모음 SVM 학습...')
    s1_scaler = StandardScaler()
    X1 = s1_scaler.fit_transform(X_emb16)

    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale',
                 probability=True, random_state=42)
    s1_clf.fit(X1, train_labels)

    train_preds = s1_clf.predict(X1)
    train_acc = np.mean(train_preds == train_labels) * 100
    print(f'  학습 정확도: {train_acc:.1f}%')
    print(f'  클래스: {list(s1_clf.classes_)}')

    # 모음별 학습 정확도
    for v in VOWELS:
        mask = train_labels == v
        if mask.any():
            vacc = np.mean(train_preds[mask] == v) * 100
            print(f'    {v}: {vacc:.0f}% ({mask.sum()}개)')

    # ── Stage 2: Layer 5-7 → 오/우 SVM ──
    print('\n[Stage 2] Layer 5-7 오/우 이진 SVM 학습...')
    ou_mask = np.isin(train_labels, ['오', '우'])
    X2_ou = X_emb567[ou_mask]
    y_ou = train_labels[ou_mask]

    s2_scaler = StandardScaler()
    X2 = s2_scaler.fit_transform(X2_ou)

    s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale',
                 probability=True, random_state=42)
    s2_clf.fit(X2, y_ou)

    s2_preds = s2_clf.predict(X2)
    s2_acc = np.mean(s2_preds == y_ou) * 100
    print(f'  학습 정확도: {s2_acc:.1f}%')
    print(f'  클래스: {list(s2_clf.classes_)}')
    print(f'  오: {np.mean(s2_preds[y_ou=="오"]=="오")*100:.0f}%  '
          f'우: {np.mean(s2_preds[y_ou=="우"]=="우")*100:.0f}%')

    # ── 기존 모델 백업 ──
    model_path = os.path.join(BASE, 'twostage_model.pkl')
    backup_path = os.path.join(BASE, 'twostage_model_pre_e3.pkl')

    if os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f'\n  기존 모델 백업: {backup_path}')

    # ── 저장 ──
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
            'description': 'XLSR-53 Layer 5-7 mean -> SVM (오/우 binary)',
        },
        'model_name': 'facebook/wav2vec2-large-xlsr-53',
        'n_train': len(all_samples),
        'config': 'E3: TTS(이은서 제외) + remote(kdg0534, lynn03)',
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f'\n  모델 저장: {model_path}')
    print(f'  파일 크기: {os.path.getsize(model_path) / 1024:.0f} KB')
    print(f'\n{"="*60}')
    print('  완료! 이제 아현이 라이브 테스트할 수 있습니다.')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
