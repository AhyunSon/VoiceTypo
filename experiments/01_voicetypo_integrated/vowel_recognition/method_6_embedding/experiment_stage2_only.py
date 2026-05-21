"""Stage 2(오/우) 분류기만 다양한 조건으로 비교.

Stage 1은 E3으로 고정 (TTS 이은서제외 + remote 2명).
Stage 2만 학습 데이터를 바꿔가며 아현 unseen 테스트.

비교 조건:
  S2-A) TTS+remote 전체 (현재 E3 그대로)
  S2-B) remote 2명만 (TTS 제외)
  S2-C) TTS 다운샘플(1:1) + remote 2명
  S2-D) remote 2명 + TTS 소량(10%)
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

REMOTE_DIRS_2 = [
    'vowel-remote-001_kdg0534 (1)',
    'vowel-remote-001_lynn03 (1)',
]
REMOTE_DIR_AH = 'vowel-remote-001_아현 (1)'


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


def collect_remote(dirs):
    samples = []
    for rd in dirs:
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


def load_cached_embeddings():
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    data = np.load(cache_path, allow_pickle=True)
    paths = list(data['paths'])
    emb16_map = {str(paths[i]): data['emb16'][i] for i in range(len(paths))}
    emb567_map = {str(paths[i]): data['emb567'][i] for i in range(len(paths))}
    return emb16_map, emb567_map


def main():
    print('=' * 60)
    print('  Stage 2 분류기 비교 실험')
    print('  (Stage 1은 E3 고정, Stage 2만 변경)')
    print('=' * 60)

    tts = collect_tts_no_eunseo()
    remote_2 = collect_remote(REMOTE_DIRS_2)
    remote_ah = collect_remote([REMOTE_DIR_AH])
    emb16_map, emb567_map = load_cached_embeddings()

    # Stage 1용 전체 학습셋 (E3)
    s1_train = tts + remote_2

    # 오/우 필터
    tts_ou = [s for s in tts if s[1] in ['오', '우']]
    remote_2_ou = [s for s in remote_2 if s[1] in ['오', '우']]
    remote_ah_ou = [s for s in remote_ah if s[1] in ['오', '우']]

    print(f'\n  Stage 2 오/우 데이터:')
    print(f'    TTS(이은서제외): 오={sum(1 for s in tts_ou if s[1]=="오")} '
          f'우={sum(1 for s in tts_ou if s[1]=="우")} = {len(tts_ou)}개')
    print(f'    Remote 2명:     오={sum(1 for s in remote_2_ou if s[1]=="오")} '
          f'우={sum(1 for s in remote_2_ou if s[1]=="우")} = {len(remote_2_ou)}개')
    print(f'    아현 (테스트):   오={sum(1 for s in remote_ah_ou if s[1]=="오")} '
          f'우={sum(1 for s in remote_ah_ou if s[1]=="우")} = {len(remote_ah_ou)}개')

    # 화자별 세부
    print(f'\n  화자별 오/우:')
    for label, samples in [('TTS', tts_ou), ('remote', remote_2_ou)]:
        speakers = sorted(set(s[2] for s in samples))
        for spk in speakers:
            oh = sum(1 for s in samples if s[2] == spk and s[1] == '오')
            oo = sum(1 for s in samples if s[2] == spk and s[1] == '우')
            print(f'    {spk:12s}: 오={oh:2d} 우={oo:2d}')

    # ── Stage 1 학습 (고정) ──
    print('\n  Stage 1 학습 (E3 고정)...')
    X1_all = np.array([emb16_map[s[0]] for s in s1_train])
    y1_all = np.array([s[1] for s in s1_train])
    s1_scaler = StandardScaler()
    X1_scaled = s1_scaler.fit_transform(X1_all)
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(X1_scaled, y1_all)
    print(f'  Stage 1 학습 완료 ({len(s1_train)}개)')

    # ── Stage 2 조건별 비교 ──
    conditions = []

    # S2-A: TTS + remote (현재)
    conditions.append(('S2-A: TTS+remote 전체', tts_ou + remote_2_ou))

    # S2-B: remote만
    conditions.append(('S2-B: remote 2명만 (TTS 제외)', remote_2_ou))

    # S2-C: TTS 다운샘플 + remote (1:1)
    np.random.seed(42)
    n_remote_ou = len(remote_2_ou)
    tts_ou_oh = [s for s in tts_ou if s[1] == '오']
    tts_ou_oo = [s for s in tts_ou if s[1] == '우']
    n_per = n_remote_ou // 2  # 오, 우 각각
    ds_oh = [tts_ou_oh[i] for i in np.random.choice(len(tts_ou_oh), min(n_per, len(tts_ou_oh)), replace=False)]
    ds_oo = [tts_ou_oo[i] for i in np.random.choice(len(tts_ou_oo), min(n_per, len(tts_ou_oo)), replace=False)]
    conditions.append((f'S2-C: TTS 다운샘플({len(ds_oh)+len(ds_oo)}) + remote', ds_oh + ds_oo + remote_2_ou))

    # S2-D: remote + TTS 소량(~10%)
    n_small = max(2, len(tts_ou) // 10)
    small_oh = [tts_ou_oh[i] for i in np.random.choice(len(tts_ou_oh), min(n_small, len(tts_ou_oh)), replace=False)]
    small_oo = [tts_ou_oo[i] for i in np.random.choice(len(tts_ou_oo), min(n_small, len(tts_ou_oo)), replace=False)]
    conditions.append((f'S2-D: remote + TTS 소량({len(small_oh)+len(small_oo)})', remote_2_ou + small_oh + small_oo))

    # ── LOSO + 아현 unseen 평가 ──
    print(f'\n{"="*60}')
    all_summaries = []

    for cond_name, s2_train_pool in conditions:
        print(f'\n  {cond_name}')
        oh_cnt = sum(1 for s in s2_train_pool if s[1] == '오')
        oo_cnt = sum(1 for s in s2_train_pool if s[1] == '우')
        print(f'  S2 학습: 오={oh_cnt} 우={oo_cnt} = {len(s2_train_pool)}개')
        print(f'  {"─"*50}')

        # --- remote 2명 LOSO ---
        remote_speakers = sorted(set(s[2] for s in remote_2))
        loso_results = {}

        for held in remote_speakers:
            test = [s for s in remote_2 if s[2] == held]
            # Stage 2 학습: pool에서 held-out의 오/우 제거
            s2_train = [s for s in s2_train_pool if s[2] != held]
            # held-out이 아닌 remote 오/우 추가 (이미 pool에 있으면 중복 방지)
            s2_train_paths = set(s[0] for s in s2_train)
            for s in remote_2_ou:
                if s[2] != held and s[0] not in s2_train_paths:
                    s2_train.append(s)

            s2_ou = [s for s in s2_train if s[1] in ['오', '우']]
            if len(s2_ou) < 2:
                continue

            X2 = np.array([emb567_map[s[0]] for s in s2_ou])
            y2 = np.array([s[1] for s in s2_ou])
            s2_scaler = StandardScaler()
            X2s = s2_scaler.fit_transform(X2)
            s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
            s2_clf.fit(X2s, y2)

            # 테스트
            speaker_results = []
            for s in test:
                e16 = emb16_map[s[0]]
                e567 = emb567_map[s[0]]
                X1t = s1_scaler.transform(e16.reshape(1, -1))
                pred1 = s1_clf.predict(X1t)[0]
                if pred1 in ['오', '우']:
                    X2t = s2_scaler.transform(e567.reshape(1, -1))
                    pred = s2_clf.predict(X2t)[0]
                    conf = float(max(s2_clf.predict_proba(X2t)[0]))
                else:
                    pred = pred1
                    conf = float(max(s1_clf.predict_proba(X1t)[0]))
                speaker_results.append((s[1], pred, conf))
            loso_results[held] = speaker_results

        # --- 아현 unseen (S2 전체 pool로 학습) ---
        s2_ou_full = [s for s in s2_train_pool if s[1] in ['오', '우']]
        X2 = np.array([emb567_map[s[0]] for s in s2_ou_full])
        y2 = np.array([s[1] for s in s2_ou_full])
        s2_scaler = StandardScaler()
        X2s = s2_scaler.fit_transform(X2)
        s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        s2_clf.fit(X2s, y2)

        ah_results = []
        for s in remote_ah:
            e16 = emb16_map[s[0]]
            e567 = emb567_map[s[0]]
            X1t = s1_scaler.transform(e16.reshape(1, -1))
            pred1 = s1_clf.predict(X1t)[0]
            if pred1 in ['오', '우']:
                X2t = s2_scaler.transform(e567.reshape(1, -1))
                pred = s2_clf.predict(X2t)[0]
                conf = float(max(s2_clf.predict_proba(X2t)[0]))
            else:
                pred = pred1
                conf = float(max(s1_clf.predict_proba(X1t)[0]))
            ah_results.append((s[1], pred, conf))

        # 출력
        for spk in sorted(loso_results.keys()):
            sr = loso_results[spk]
            sc = sum(1 for g, p, _ in sr if g == p)
            ou = [(g, p) for g, p, _ in sr if g in ['오', '우']]
            ouc = sum(1 for g, p in ou if g == p)
            oh_r = [(g, p) for g, p in ou if g == '오']
            oo_r = [(g, p) for g, p in ou if g == '우']
            oh_c = sum(1 for g, p in oh_r if g == p)
            oo_c = sum(1 for g, p in oo_r if g == p)
            print(f'    {spk:8s}: 전체={sc}/{len(sr)}({sc/len(sr)*100:.0f}%)  '
                  f'오/우={ouc}/{len(ou)}  오={oh_c}/{len(oh_r)} 우={oo_c}/{len(oo_r)}')

        ah_total = len(ah_results)
        ah_correct = sum(1 for g, p, _ in ah_results if g == p)
        ah_ou = [(g, p) for g, p, _ in ah_results if g in ['오', '우']]
        ah_ouc = sum(1 for g, p in ah_ou if g == p)
        ah_oh = [(g, p) for g, p in ah_ou if g == '오']
        ah_oo = [(g, p) for g, p in ah_ou if g == '우']
        ah_oh_c = sum(1 for g, p in ah_oh if g == p)
        ah_oo_c = sum(1 for g, p in ah_oo if g == p)
        print(f'    {"아현":8s}: 전체={ah_correct}/{ah_total}({ah_correct/ah_total*100:.0f}%)  '
              f'오/우={ah_ouc}/{len(ah_ou)}  오={ah_oh_c}/{len(ah_oh)} 우={ah_oo_c}/{len(ah_oo)}')

        # 아현 모음별 상세
        for v in VOWELS:
            vr = [(g, p) for g, p, _ in ah_results if g == v]
            if not vr:
                continue
            vc = sum(1 for g, p in vr if g == p)
            if vc < len(vr):
                errs = {}
                for g, p in vr:
                    if g != p:
                        errs[p] = errs.get(p, 0) + 1
                err_str = ' '.join(f'→{k}({n})' for k, n in sorted(errs.items(), key=lambda x: -x[1]))
                print(f'      {v}: {vc}/{len(vr)} {err_str}')

        all_summaries.append({
            'name': cond_name,
            'loso_results': loso_results,
            'ah_total': f'{ah_correct}/{ah_total}',
            'ah_acc': ah_correct / ah_total * 100,
            'ah_ou': f'{ah_ouc}/{len(ah_ou)}',
            'ah_ou_acc': ah_ouc / len(ah_ou) * 100 if ah_ou else 0,
            'ah_oh': f'{ah_oh_c}/{len(ah_oh)}',
            'ah_oo': f'{ah_oo_c}/{len(ah_oo)}',
        })

    # ── 최종 비교 ──
    print(f'\n\n{"#"*60}')
    print(f'  Stage 2 비교 요약')
    print(f'{"#"*60}\n')

    print(f'  {"조건":<35s} {"아현전체":>8s} {"아현오/우":>8s} {"오":>5s} {"우":>5s}')
    print(f'  {"─"*35} {"─"*8} {"─"*8} {"─"*5} {"─"*5}')
    for s in all_summaries:
        name = s['name']
        if len(name) > 35:
            name = name[:32] + '...'
        print(f'  {name:<35s} {s["ah_acc"]:6.1f}%  {s["ah_ou_acc"]:6.0f}%  '
              f'{s["ah_oh"]:>5s} {s["ah_oo"]:>5s}')

    best = max(all_summaries, key=lambda s: (s['ah_ou_acc'], s['ah_acc']))
    print(f'\n  최적: {best["name"]}')
    print(f'        아현 전체={best["ah_total"]} 오/우={best["ah_ou"]} 오={best["ah_oh"]} 우={best["ah_oo"]}')


if __name__ == '__main__':
    main()
