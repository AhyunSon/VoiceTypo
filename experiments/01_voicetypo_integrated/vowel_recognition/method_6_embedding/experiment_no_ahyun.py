"""아현 데이터 제외 실험.

비교:
  1) 아현 포함 LOSO (기존 결과 재현)
  2) 아현 완전 제외 LOSO (kdg0534/lynn03만)
  3) 아현 제외 학습 → 아현 테스트 (= 기존 LOSO 아현 fold와 동일하지만 명시적으로)

핵심 질문: 아현 데이터가 다른 화자 학습에 도움이 되는가?
"""
import sys, os, io, wave, time
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
    'vowel-remote-001_아현 (1)',
]


def syllable_to_vowel(ch):
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    medial = (code % (28 * 21)) // 28
    return _MEDIAL_TO_VOWEL.get(medial)


def collect_tts_data():
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
        if first in VOWELS:
            vowel = first
        elif len(first) == 1:
            vowel = syllable_to_vowel(first)
        else:
            vowel = None
        if vowel is None or vowel not in VOWELS:
            continue
        speaker = parts[2] if len(parts) >= 3 else 'unknown'
        samples.append((os.path.join(DATASET_DIR, f), vowel, f'tts_{speaker}', 'tts'))
    return samples


def collect_remote_data():
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


def load_cached_embeddings():
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}
    return emb16_map, emb567_map


def train_twostage(X16, X567, y):
    s1_scaler = StandardScaler()
    X1 = s1_scaler.fit_transform(X16)
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(X1, y)

    ou_mask = np.isin(y, ['오', '우'])
    s2_scaler = StandardScaler()
    X2 = s2_scaler.fit_transform(X567[ou_mask])
    s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s2_clf.fit(X2, y[ou_mask])

    return s1_scaler, s1_clf, s2_scaler, s2_clf


def predict(emb16, emb567, s1_scaler, s1_clf, s2_scaler, s2_clf):
    X1 = s1_scaler.transform(emb16.reshape(1, -1))
    pred1 = s1_clf.predict(X1)[0]
    proba1 = s1_clf.predict_proba(X1)[0]
    if pred1 in ['오', '우']:
        X2 = s2_scaler.transform(emb567.reshape(1, -1))
        pred2 = s2_clf.predict(X2)[0]
        proba2 = s2_clf.predict_proba(X2)[0]
        return pred2, float(max(proba2))
    return pred1, float(max(proba1))


def run_loso(train_pool, test_speakers_data, emb16_map, emb567_map, label):
    """LOSO: test_speakers_data 중 한 명씩 hold-out."""
    speakers = sorted(set(s[2] for s in test_speakers_data))
    all_results = []
    by_speaker = {}

    for held in speakers:
        test = [s for s in test_speakers_data if s[2] == held]
        other = [s for s in test_speakers_data if s[2] != held]
        train = train_pool + other

        X_tr_16 = np.array([emb16_map[s[0]] for s in train])
        X_tr_567 = np.array([emb567_map[s[0]] for s in train])
        y_tr = np.array([s[1] for s in train])

        s1s, s1c, s2s, s2c = train_twostage(X_tr_16, X_tr_567, y_tr)

        speaker_results = []
        for s in test:
            pred, conf = predict(emb16_map[s[0]], emb567_map[s[0]], s1s, s1c, s2s, s2c)
            speaker_results.append((s[1], pred, conf))
        all_results.extend(speaker_results)
        by_speaker[held] = speaker_results

    return all_results, by_speaker


def print_summary(all_results, by_speaker, label):
    total = len(all_results)
    correct = sum(1 for g, p, _ in all_results if g == p)
    print(f'\n  {label}')
    print(f'  전체: {correct}/{total} ({correct/total*100:.1f}%)')

    for v in VOWELS:
        vr = [(g, p) for g, p, _ in all_results if g == v]
        if not vr:
            continue
        vc = sum(1 for g, p in vr if g == p)
        errors = {}
        for g, p in vr:
            if g != p:
                errors[p] = errors.get(p, 0) + 1
        err_str = ', '.join(f'{k}({n})' for k, n in sorted(errors.items(), key=lambda x: -x[1]))
        mark = ' ◀' if vc / len(vr) < 0.7 else ''
        print(f'    {v}: {vc:2d}/{len(vr):2d} ({vc/len(vr)*100:5.1f}%)  {err_str}{mark}')

    ou = [(g, p) for g, p, _ in all_results if g in ['오', '우']]
    if ou:
        ouc = sum(1 for g, p in ou if g == p)
        oh_r = [x for x in ou if x[0] == '오']
        oo_r = [x for x in ou if x[0] == '우']
        oh_a = sum(1 for g, p in oh_r if g == p) / len(oh_r) * 100 if oh_r else 0
        oo_a = sum(1 for g, p in oo_r if g == p) / len(oo_r) * 100 if oo_r else 0
        print(f'  오/우: {ouc}/{len(ou)} ({ouc/len(ou)*100:.1f}%)  오={oh_a:.0f}% 우={oo_a:.0f}%')

    for spk in sorted(by_speaker.keys()):
        sr = by_speaker[spk]
        sc = sum(1 for g, p, _ in sr if g == p)
        ou_s = [(g, p) for g, p, _ in sr if g in ['오', '우']]
        ou_c = sum(1 for g, p in ou_s if g == p)
        ou_str = f'오/우={ou_c}/{len(ou_s)}' if ou_s else ''
        print(f'    {spk:8s}: {sc:2d}/{len(sr):2d} ({sc/len(sr)*100:5.1f}%)  {ou_str}')


def main():
    print('=' * 60)
    print('  아현 제외 실험')
    print('=' * 60)

    tts = collect_tts_data()
    remote = collect_remote_data()
    emb16_map, emb567_map = load_cached_embeddings()

    remote_with = remote  # 아현 포함
    remote_without = [s for s in remote if s[2] != '아현']  # 아현 제외
    remote_ahyun = [s for s in remote if s[2] == '아현']

    print(f'\n  TTS: {len(tts)}')
    print(f'  Remote (아현 포함): {len(remote_with)}')
    print(f'  Remote (아현 제외): {len(remote_without)}')
    print(f'  아현 데이터: {len(remote_ahyun)}')

    # ═══════════════════════════════════════
    #  실험 1: A 조건 — 아현 포함 vs 제외
    # ═══════════════════════════════════════
    print(f'\n{"="*60}')
    print('  조건 A: TTS + remote')
    print(f'{"="*60}')

    # 1a) 아현 포함 LOSO (3명)
    print('\n--- 아현 포함 (3명 LOSO) ---')
    r1a, bs1a = run_loso(tts, remote_with, emb16_map, emb567_map, '아현 포함')
    print_summary(r1a, bs1a, '아현 포함 LOSO (전체)')

    # 1b) 아현 제외 LOSO (2명만)
    print('\n--- 아현 제외 (2명 LOSO) ---')
    r1b, bs1b = run_loso(tts, remote_without, emb16_map, emb567_map, '아현 제외')
    print_summary(r1b, bs1b, '아현 제외 LOSO (kdg0534, lynn03만)')

    # 1c) 핵심 비교: kdg0534, lynn03의 성적이 아현 포함/제외에 따라 달라지는지
    print(f'\n{"─"*60}')
    print('  핵심 비교: 아현이 학습에 포함될 때 vs 제외될 때')
    print('  (kdg0534, lynn03 hold-out 결과)')
    print(f'{"─"*60}')

    print(f'\n  {"화자":<10s} {"아현포함":>10s} {"아현제외":>10s} {"차이":>8s}')
    print(f'  {"─"*10} {"─"*10} {"─"*10} {"─"*8}')
    for spk in ['kdg0534', 'lynn03']:
        # 아현 포함시: 이 화자 hold-out → TTS + 나머지 2명 remote(아현 포함)으로 학습
        r_with = bs1a.get(spk, [])
        acc_with = sum(1 for g, p, _ in r_with if g == p) / len(r_with) * 100 if r_with else 0

        # 아현 제외시: 이 화자 hold-out → TTS + 나머지 1명 remote(아현 미포함)으로 학습
        r_without = bs1b.get(spk, [])
        acc_without = sum(1 for g, p, _ in r_without if g == p) / len(r_without) * 100 if r_without else 0

        diff = acc_with - acc_without
        sign = '+' if diff > 0 else ''
        print(f'  {spk:<10s} {acc_with:8.1f}%  {acc_without:8.1f}%  {sign}{diff:5.1f}%')

    # 오/우 세부
    print(f'\n  오/우만:')
    print(f'  {"화자":<10s} {"아현포함":>10s} {"아현제외":>10s} {"차이":>8s}')
    print(f'  {"─"*10} {"─"*10} {"─"*10} {"─"*8}')
    for spk in ['kdg0534', 'lynn03']:
        r_with = bs1a.get(spk, [])
        ou_w = [(g, p) for g, p, _ in r_with if g in ['오', '우']]
        acc_w = sum(1 for g, p in ou_w if g == p) / len(ou_w) * 100 if ou_w else 0

        r_without = bs1b.get(spk, [])
        ou_wo = [(g, p) for g, p, _ in r_without if g in ['오', '우']]
        acc_wo = sum(1 for g, p in ou_wo if g == p) / len(ou_wo) * 100 if ou_wo else 0

        diff = acc_w - acc_wo
        sign = '+' if diff > 0 else ''
        print(f'  {spk:<10s} {acc_w:8.1f}%  {acc_wo:8.1f}%  {sign}{diff:5.1f}%')

    # 1d) 아현 제외 학습 → 아현 테스트 (unseen speaker)
    print(f'\n{"─"*60}')
    print('  아현을 완전히 unseen으로 테스트')
    print(f'{"─"*60}')

    # TTS + kdg0534 + lynn03으로 학습 → 아현 테스트
    train_no_ah = tts + remote_without
    X_tr_16 = np.array([emb16_map[s[0]] for s in train_no_ah])
    X_tr_567 = np.array([emb567_map[s[0]] for s in train_no_ah])
    y_tr = np.array([s[1] for s in train_no_ah])

    s1s, s1c, s2s, s2c = train_twostage(X_tr_16, X_tr_567, y_tr)

    ah_results = []
    for s in remote_ahyun:
        pred, conf = predict(emb16_map[s[0]], emb567_map[s[0]], s1s, s1c, s2s, s2c)
        ah_results.append((s[1], pred, conf))

    ac = sum(1 for g, p, _ in ah_results if g == p)
    at = len(ah_results)
    print(f'\n  TTS+kdg0534+lynn03 → 아현 테스트: {ac}/{at} ({ac/at*100:.1f}%)')
    for v in VOWELS:
        vr = [(g, p) for g, p, _ in ah_results if g == v]
        if not vr:
            continue
        vc = sum(1 for g, p in vr if g == p)
        errs = [(p, sum(1 for g2, p2 in vr if p2 == p and g2 != p2)) for p in VOWELS if any(g2 != p2 and p2 == p for g2, p2 in vr)]
        err_str = ', '.join(f'{p}({n})' for p, n in errs if n > 0)
        print(f'    {v}: {vc}/{len(vr)}  {err_str}')

    print(f'\n{"="*60}')
    print('  실험 완료')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
