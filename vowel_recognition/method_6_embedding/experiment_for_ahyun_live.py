"""아현 라이브 테스트를 위한 최적 학습셋 탐색.

아현은 학습에서 제외 → unseen speaker로 테스트.
이은서(여성 TTS)가 아현 대체 역할을 할 수 있는지 확인.

비교 조건:
  E1) TTS 전체(525) + remote 2명(140) — 기본 (이은서 164개 포함)
  E2) TTS 이은서만(164) + remote 2명(140) — 이은서 집중
  E3) TTS 이은서 제외(361) + remote 2명(140) — 이은서 빼면?
  E4) TTS 이은서 2배(164→328) + remote 2명(140) — 이은서 가중
  E5) TTS 여성만(이은서+Anna=330) + remote 2명(140) — 여성 집중

평가:
  - kdg0534/lynn03 LOSO (2-fold)
  - 아현 unseen 테스트
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
        gender = parts[1] if len(parts) >= 2 else '?'
        samples.append((os.path.join(DATASET_DIR, f), vowel, f'tts_{speaker}', 'tts', gender))
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
            samples.append((os.path.join(d, f), vowel, speaker, 'remote', '?'))
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


def predict_batch(test_samples, emb16_map, emb567_map, s1s, s1c, s2s, s2c):
    results = []
    for s in test_samples:
        emb16 = emb16_map[s[0]]
        emb567 = emb567_map[s[0]]
        X1 = s1s.transform(emb16.reshape(1, -1))
        pred1 = s1c.predict(X1)[0]
        proba1 = s1c.predict_proba(X1)[0]
        if pred1 in ['오', '우']:
            X2 = s2s.transform(emb567.reshape(1, -1))
            pred2 = s2c.predict(X2)[0]
            proba2 = s2c.predict_proba(X2)[0]
            results.append((s[1], pred2, float(max(proba2))))
        else:
            results.append((s[1], pred1, float(max(proba1))))
    return results


def evaluate(train_samples, remote_2, remote_ahyun, emb16_map, emb567_map, label):
    """2명 LOSO + 아현 unseen 평가."""
    speakers_2 = sorted(set(s[2] for s in remote_2))

    # --- 2명 LOSO ---
    loso_results = []
    loso_by_speaker = {}
    for held in speakers_2:
        test = [s for s in remote_2 if s[2] == held]
        other = [s for s in remote_2 if s[2] != held]
        train = train_samples + other

        X16 = np.array([emb16_map[s[0]] for s in train])
        X567 = np.array([emb567_map[s[0]] for s in train])
        y = np.array([s[1] for s in train])

        s1s, s1c, s2s, s2c = train_twostage(X16, X567, y)
        sr = predict_batch(test, emb16_map, emb567_map, s1s, s1c, s2s, s2c)
        loso_results.extend(sr)
        loso_by_speaker[held] = sr

    # --- 아현 unseen (전체 train으로 학습) ---
    full_train = train_samples + remote_2
    X16 = np.array([emb16_map[s[0]] for s in full_train])
    X567 = np.array([emb567_map[s[0]] for s in full_train])
    y = np.array([s[1] for s in full_train])

    s1s, s1c, s2s, s2c = train_twostage(X16, X567, y)
    ah_results = predict_batch(remote_ahyun, emb16_map, emb567_map, s1s, s1c, s2s, s2c)

    # --- 출력 ---
    loso_total = len(loso_results)
    loso_correct = sum(1 for g, p, _ in loso_results if g == p)
    ah_total = len(ah_results)
    ah_correct = sum(1 for g, p, _ in ah_results if g == p)

    print(f'\n  {label}')
    print(f'  {"─"*50}')
    print(f'  학습 데이터: {len(train_samples)} (기본) + remote 2명')

    # LOSO
    print(f'\n  [kdg0534/lynn03 LOSO] {loso_correct}/{loso_total} ({loso_correct/loso_total*100:.1f}%)')
    for spk in speakers_2:
        sr = loso_by_speaker[spk]
        sc = sum(1 for g, p, _ in sr if g == p)
        ou = [(g, p) for g, p, _ in sr if g in ['오', '우']]
        ouc = sum(1 for g, p in ou if g == p)
        print(f'    {spk:8s}: {sc:2d}/{len(sr):2d} ({sc/len(sr)*100:4.0f}%)  오/우={ouc}/{len(ou)}')

    # 아현
    print(f'\n  [아현 unseen] {ah_correct}/{ah_total} ({ah_correct/ah_total*100:.1f}%)')
    for v in VOWELS:
        vr = [(g, p) for g, p, _ in ah_results if g == v]
        if not vr:
            continue
        vc = sum(1 for g, p in vr if g == p)
        errs = {}
        for g, p in vr:
            if g != p:
                errs[p] = errs.get(p, 0) + 1
        err_str = '  '.join(f'→{k}({n})' for k, n in sorted(errs.items(), key=lambda x: -x[1]))
        mark = ' ◀' if vc < len(vr) * 0.7 else ''
        print(f'    {v}: {vc}/{len(vr)}{" " + err_str if err_str else ""}{mark}')

    # 오/우 요약
    ah_ou = [(g, p) for g, p, _ in ah_results if g in ['오', '우']]
    ah_ou_c = sum(1 for g, p in ah_ou if g == p)
    oh_r = [(g, p) for g, p in ah_ou if g == '오']
    oo_r = [(g, p) for g, p in ah_ou if g == '우']
    oh_a = sum(1 for g, p in oh_r if g == p)
    oo_a = sum(1 for g, p in oo_r if g == p)
    print(f'    오/우: {ah_ou_c}/{len(ah_ou)}  (오:{oh_a}/{len(oh_r)} 우:{oo_a}/{len(oo_r)})')

    return {
        'label': label,
        'loso_acc': loso_correct / loso_total * 100,
        'ah_acc': ah_correct / ah_total * 100,
        'ah_ou_acc': ah_ou_c / len(ah_ou) * 100 if ah_ou else 0,
        'ah_oh': f'{oh_a}/{len(oh_r)}',
        'ah_oo': f'{oo_a}/{len(oo_r)}',
        'loso_by_speaker': {s: sum(1 for g, p, _ in r if g == p) / len(r) * 100
                           for s, r in loso_by_speaker.items()},
    }


def main():
    print('=' * 60)
    print('  아현 라이브 테스트를 위한 최적 학습셋 탐색')
    print('=' * 60)

    tts_all = collect_tts_data()  # (path, vowel, speaker, domain, gender)
    remote_all = collect_remote_data()
    emb16_map, emb567_map = load_cached_embeddings()

    remote_2 = [s for s in remote_all if s[2] != '아현']
    remote_ahyun = [s for s in remote_all if s[2] == '아현']

    # TTS 분류
    tts_eunseo = [s for s in tts_all if '이은서' in s[2]]
    tts_anna = [s for s in tts_all if 'Anna' in s[2]]
    tts_dongkyu = [s for s in tts_all if '김동규' in s[2]]
    tts_female = tts_eunseo + tts_anna
    tts_no_eunseo = tts_anna + tts_dongkyu

    print(f'\n  TTS 전체: {len(tts_all)}개')
    print(f'    이은서(여): {len(tts_eunseo)}')
    print(f'    Anna(여): {len(tts_anna)}')
    print(f'    김동규(남): {len(tts_dongkyu)}')
    print(f'  Remote 2명: {len(remote_2)}개 (kdg0534, lynn03)')
    print(f'  아현 (테스트): {len(remote_ahyun)}개')

    # ═══════════════════════════════════════
    #  5가지 조건 비교
    # ═══════════════════════════════════════
    results = []

    print(f'\n{"#"*60}')

    # E1: TTS 전체 + remote 2명
    r = evaluate(tts_all, remote_2, remote_ahyun, emb16_map, emb567_map,
                 f'E1: TTS 전체({len(tts_all)}) + remote 2명')
    results.append(r)

    # E2: TTS 이은서만 + remote 2명
    r = evaluate(tts_eunseo, remote_2, remote_ahyun, emb16_map, emb567_map,
                 f'E2: TTS 이은서만({len(tts_eunseo)}) + remote 2명')
    results.append(r)

    # E3: TTS 이은서 제외 + remote 2명
    r = evaluate(tts_no_eunseo, remote_2, remote_ahyun, emb16_map, emb567_map,
                 f'E3: TTS 이은서 제외({len(tts_no_eunseo)}) + remote 2명')
    results.append(r)

    # E4: TTS 이은서 2배 + remote 2명
    tts_eunseo_2x = tts_eunseo + tts_eunseo  # 이은서 데이터를 2번 넣기
    tts_e4 = tts_no_eunseo + tts_eunseo_2x
    r = evaluate(tts_e4, remote_2, remote_ahyun, emb16_map, emb567_map,
                 f'E4: TTS(이은서 2배={len(tts_e4)}) + remote 2명')
    results.append(r)

    # E5: TTS 여성만 + remote 2명
    r = evaluate(tts_female, remote_2, remote_ahyun, emb16_map, emb567_map,
                 f'E5: TTS 여성만({len(tts_female)}) + remote 2명')
    results.append(r)

    # E6: remote 2명 only (TTS 없이)
    r = evaluate([], remote_2, remote_ahyun, emb16_map, emb567_map,
                 f'E6: remote 2명만({len(remote_2)}) — TTS 없음')
    results.append(r)

    # ═══════════════════════════════════════
    #  최종 비교표
    # ═══════════════════════════════════════
    print(f'\n\n{"#"*60}')
    print(f'  최종 비교표')
    print(f'{"#"*60}\n')

    print(f'  {"조건":<35s} {"LOSO":>6s} {"아현":>6s} {"아현오/우":>8s} {"오":>5s} {"우":>5s}')
    print(f'  {"─"*35} {"─"*6} {"─"*6} {"─"*8} {"─"*5} {"─"*5}')

    for r in results:
        label = r['label']
        if len(label) > 35:
            label = label[:32] + '...'
        loso = f'{r["loso_acc"]:.1f}%'
        ah = f'{r["ah_acc"]:.1f}%'
        ah_ou = f'{r["ah_ou_acc"]:.0f}%'
        print(f'  {label:<35s} {loso:>6s} {ah:>6s} {ah_ou:>8s} {r["ah_oh"]:>5s} {r["ah_oo"]:>5s}')

    # 최고 조건 표시
    best_ah = max(results, key=lambda r: r['ah_acc'])
    best_ou = max(results, key=lambda r: r['ah_ou_acc'])
    print(f'\n  아현 전체 최고: {best_ah["label"]} ({best_ah["ah_acc"]:.1f}%)')
    print(f'  아현 오/우 최고: {best_ou["label"]} ({best_ou["ah_ou_acc"]:.0f}%)')

    print(f'\n{"="*60}')
    print('  실험 완료')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
