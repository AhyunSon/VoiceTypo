"""HJ/MT 추가 후 종합 평가.

학습 데이터: TTS(이은서 제외) + kdg0534 + lynn03 + HJ + MT + 우오구분용
테스트: 아현 (unseen speaker) remote 70개 + live 20개

평가:
  1) LOSO: 전체 학습 데이터에서 화자별 leave-one-out
  2) 아현 테스트: 전체 학습 데이터로 학습 → 아현 전체 테스트
  3) 이전(HJ/MT 없이) vs 이후 비교
"""
import sys, os, io, wave
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from collections import Counter

BASE = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE, '..', 'dataset')
OU_EXTRA_DIR = os.path.join(DATASET_DIR, '우 오 구분용')
LIVE_DIR = os.path.join(BASE, 'live_recordings')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}

REMOTE_TRAIN_DIRS = [
    ('vowel-remote-001_kdg0534 (1)', 'kdg0534'),
    ('vowel-remote-001_lynn03 (1)', 'lynn03'),
    ('vowel-remote-001_hj', 'hj'),
    ('vowel-remote-001_mt', 'mt'),
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


# ── 데이터 수집 ──

def collect_tts(exclude_eunseo=True):
    """TTS dataset 수집. (path, vowel, speaker)"""
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
        if exclude_eunseo and speaker == '이은서':
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
        samples.append((fp, vowel, f'tts_{speaker}'))
    return samples


def collect_ou_extra():
    """우 오 구분용 추가 데이터."""
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
        samples.append((os.path.join(OU_EXTRA_DIR, f), vowel, f'tts_{speaker}'))
    return samples


def collect_remote_train():
    """학습용 remote 화자 수집 (kdg0534, lynn03, hj, mt)."""
    samples = []
    for dirname, speaker_id in REMOTE_TRAIN_DIRS:
        d = os.path.join(BASE, dirname)
        if not os.path.isdir(d):
            print(f'  [경고] 디렉토리 없음: {dirname}')
            continue
        count = 0
        for f in sorted(os.listdir(d)):
            if not f.endswith('.wav'):
                continue
            parts = os.path.splitext(f)[0].split('_')
            if len(parts) < 4:
                continue
            vowel = parts[2]
            if vowel not in VOWELS:
                continue
            samples.append((os.path.join(d, f), vowel, speaker_id))
            count += 1
        print(f'    {speaker_id}: {count}개')
    return samples


def collect_live_ou():
    """live_recordings 오/우."""
    session_map = {
        'session_20260310_145230': 'live_서울여성',
        'session_20260310_151524': 'live_경상도여성',
        'session_20260310_153047': 'live_20대남성',
    }
    samples = []
    seen = set()
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
            key = f'{speaker}_{f}'
            if key in seen:
                continue
            seen.add(key)
            samples.append((os.path.join(d, f), vowel, speaker))
    return samples


def collect_ahyun_remote():
    """아현 remote 70개."""
    d = os.path.join(BASE, 'vowel-remote-001_아현 (1)')
    samples = []
    if not os.path.isdir(d):
        return samples
    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        fp = os.path.join(d, f)
        if os.path.isdir(fp):
            continue
        parts = os.path.splitext(f)[0].split('_')
        if len(parts) < 4:
            continue
        vowel = parts[2]
        if vowel not in VOWELS:
            continue
        samples.append((fp, vowel, '아현', 'remote', f))
    return samples


def collect_ahyun_live():
    """아현 live 20개 (session_20260310_145230)."""
    d = os.path.join(LIVE_DIR, 'session_20260310_145230')
    samples = []
    if not os.path.isdir(d):
        return samples
    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        vowel = f.split('_')[0]
        if vowel in VOWELS:
            samples.append((os.path.join(d, f), vowel, '아현', 'live', f))
    return samples


# ── 임베딩 ──

def get_embeddings(paths, emb16_map, emb567_map):
    """필요한 임베딩 추출."""
    need = [p for p in paths if p not in emb16_map]
    if not need:
        return

    print(f'  임베딩 추출: {len(need)}개...', flush=True)
    model_name = 'facebook/wav2vec2-large-xlsr-53'
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    for i, p in enumerate(need):
        if (i + 1) % 20 == 0 or i == 0:
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

    print(f'  추출 완료.', flush=True)


# ── 2-Stage 학습/평가 ──

def train_two_stage(train_samples, ou_extra_samples, live_ou_samples, emb16_map, emb567_map):
    """2-stage SVM 학습. 반환: (s1_scaler, s1_clf, s2_scaler, s2_clf)"""
    # Stage 1: 전체 7모음
    X1 = np.array([emb16_map[s[0]] for s in train_samples])
    y1 = np.array([s[1] for s in train_samples])
    s1_scaler = StandardScaler()
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s1_clf.fit(s1_scaler.fit_transform(X1), y1)

    # Stage 2: 오/우 이진
    s2_data = [s for s in train_samples if s[1] in ['오', '우']]
    s2_data += [(s[0], s[1], s[2]) for s in ou_extra_samples]
    s2_data += [(s[0], s[1], s[2]) for s in live_ou_samples]
    X2 = np.array([emb567_map[s[0]] for s in s2_data])
    y2 = np.array([s[1] for s in s2_data])
    s2_scaler = StandardScaler()
    s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    s2_clf.fit(s2_scaler.fit_transform(X2), y2)

    return s1_scaler, s1_clf, s2_scaler, s2_clf, len(X1), len(X2)


def predict_two_stage(path, s1_scaler, s1_clf, s2_scaler, s2_clf, emb16_map, emb567_map):
    """2-stage 예측."""
    X1 = s1_scaler.transform(emb16_map[path].reshape(1, -1))
    p1 = s1_clf.predict(X1)[0]
    proba1 = dict(zip(s1_clf.classes_, s1_clf.predict_proba(X1)[0]))

    if p1 in ['오', '우']:
        X2 = s2_scaler.transform(emb567_map[path].reshape(1, -1))
        pred = s2_clf.predict(X2)[0]
        conf = float(max(s2_clf.predict_proba(X2)[0]))
    else:
        pred = p1
        conf = float(proba1.get(pred, 0))

    return pred, conf, proba1


def print_vowel_summary(results, label):
    """모음별 정확도 출력."""
    total = len(results)
    correct = sum(1 for r in results if r['gt'] == r['pred'])
    print(f'\n  [{label}] 전체: {correct}/{total} ({correct/total*100:.1f}%)')

    for v in VOWELS:
        vr = [r for r in results if r['gt'] == v]
        if not vr:
            continue
        vc = sum(1 for r in vr if r['gt'] == r['pred'])
        wrong = [r['pred'] for r in vr if r['gt'] != r['pred']]
        wrong_str = ''
        if wrong:
            wc = Counter(wrong)
            wrong_str = '  오류: ' + ', '.join(f'→{k}({v})' for k, v in wc.most_common())
        print(f'    {v}: {vc}/{len(vr)} ({vc/len(vr)*100:.0f}%){wrong_str}')

    return correct, total


# ── LOSO ──

def run_loso(all_train, ou_extra, live_ou, emb16_map, emb567_map):
    """전체 학습 데이터에서 화자별 LOSO."""
    speakers = sorted(set(s[2] for s in all_train))
    print(f'\n  LOSO 화자 목록 ({len(speakers)}명): {", ".join(speakers)}')

    all_results = []
    by_speaker = {}

    for held in speakers:
        train = [s for s in all_train if s[2] != held]
        test = [s for s in all_train if s[2] == held]

        # Stage 2 pool에서도 held-out 제거
        ou_train = [s for s in train if s[1] in ['오', '우']]
        ou_train += [(s[0], s[1], s[2]) for s in ou_extra if s[2] != held]
        ou_train += [(s[0], s[1], s[2]) for s in live_ou if s[2] != held]

        # Stage 1
        X1 = np.array([emb16_map[s[0]] for s in train])
        y1 = np.array([s[1] for s in train])
        s1_scaler = StandardScaler()
        s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        s1_clf.fit(s1_scaler.fit_transform(X1), y1)

        # Stage 2
        X2 = np.array([emb567_map[s[0]] for s in ou_train])
        y2 = np.array([s[1] for s in ou_train])
        s2_scaler = StandardScaler()
        s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        s2_clf.fit(s2_scaler.fit_transform(X2), y2)

        speaker_results = []
        for s in test:
            pred, conf, _ = predict_two_stage(
                s[0], s1_scaler, s1_clf, s2_scaler, s2_clf, emb16_map, emb567_map)
            speaker_results.append({'gt': s[1], 'pred': pred, 'conf': conf})

        sc = sum(1 for r in speaker_results if r['gt'] == r['pred'])
        ou_test = [r for r in speaker_results if r['gt'] in ['오', '우']]
        ou_c = sum(1 for r in ou_test if r['gt'] == r['pred'])
        oh_test = [r for r in speaker_results if r['gt'] == '오']
        oo_test = [r for r in speaker_results if r['gt'] == '우']
        oh_c = sum(1 for r in oh_test if r['gt'] == r['pred'])
        oo_c = sum(1 for r in oo_test if r['gt'] == r['pred'])

        print(f'    {held:16s}: {sc:2d}/{len(speaker_results):2d} '
              f'({sc/len(speaker_results)*100:5.1f}%)  '
              f'오/우={ou_c}/{len(ou_test)}  오={oh_c}/{len(oh_test)} 우={oo_c}/{len(oo_test)}')

        all_results.extend(speaker_results)
        by_speaker[held] = speaker_results

    return all_results, by_speaker


# ── 메인 ──

def main():
    print('=' * 65)
    print('  HJ/MT 추가 후 종합 평가')
    print('  LOSO + 아현 unseen 테스트')
    print('=' * 65)

    # ── 데이터 수집 ──
    print('\n[1] 데이터 수집')
    tts = collect_tts()
    print(f'  TTS(이은서 제외): {len(tts)}개')

    ou_extra = collect_ou_extra()
    print(f'  우오구분용: {len(ou_extra)}개')

    print(f'  Remote 학습용:')
    remote_train = collect_remote_train()
    print(f'  Remote 학습용 합계: {len(remote_train)}개')

    live_ou = collect_live_ou()
    print(f'  Live 오/우: {len(live_ou)}개')

    ah_remote = collect_ahyun_remote()
    ah_live = collect_ahyun_live()
    ah_all = ah_remote + ah_live
    print(f'  아현 테스트: remote {len(ah_remote)} + live {len(ah_live)} = {len(ah_all)}개')

    # 전체 학습 데이터
    all_train = tts + remote_train
    print(f'\n  전체 학습 데이터 (S1): {len(all_train)}개')

    # 화자별 분포
    speaker_counts = Counter(s[2] for s in all_train)
    for spk, cnt in sorted(speaker_counts.items()):
        vowel_dist = Counter(s[1] for s in all_train if s[2] == spk)
        dist_str = ' '.join(f'{v}={vowel_dist.get(v, 0)}' for v in VOWELS)
        print(f'    {spk:16s}: {cnt:3d}개  [{dist_str}]')

    # ── 임베딩 ──
    print('\n[2] 임베딩 추출/로드')

    # 캐시 로드
    cache_path = os.path.join(BASE, 'cache_loso_compare.npz')
    emb16_map = {}
    emb567_map = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        paths = list(data['paths'])
        for i, p in enumerate(paths):
            emb16_map[str(p)] = data['emb16'][i]
            emb567_map[str(p)] = data['emb567'][i]
        print(f'  캐시 로드: {len(paths)}개')

    # 새로 추출 필요한 것들
    all_paths = set()
    for s in all_train:
        all_paths.add(s[0])
    for s in ou_extra:
        all_paths.add(s[0])
    for s in live_ou:
        all_paths.add(s[0])
    for s in ah_all:
        all_paths.add(s[0])

    get_embeddings(list(all_paths), emb16_map, emb567_map)

    # 캐시 업데이트 저장
    new_cache = os.path.join(BASE, 'cache_with_hj_mt.npz')
    all_cached_paths = list(emb16_map.keys())
    np.savez(new_cache,
             paths=np.array(all_cached_paths, dtype=object),
             emb16=np.array([emb16_map[p] for p in all_cached_paths]),
             emb567=np.array([emb567_map[p] for p in all_cached_paths]))
    print(f'  캐시 저장: {new_cache} ({len(all_cached_paths)}개)')

    # ══════════════════════════════════════════
    # 비교 A: 이전 (HJ/MT 없이)
    # ══════════════════════════════════════════
    print(f'\n\n{"#"*65}')
    print(f'  비교 A: 이전 (HJ/MT 없이) — TTS + kdg0534 + lynn03')
    print(f'{"#"*65}')

    old_remote = [s for s in remote_train if s[2] in ['kdg0534', 'lynn03']]
    old_train = tts + old_remote

    # LOSO
    print(f'\n  ── LOSO (이전) ──')
    old_loso_results, old_loso_by_speaker = run_loso(
        old_train, ou_extra, live_ou, emb16_map, emb567_map)
    old_loso_c, old_loso_t = print_vowel_summary(old_loso_results, 'LOSO 이전')

    # 아현 테스트 (이전)
    print(f'\n  ── 아현 테스트 (이전) ──')
    s1s, s1c, s2s, s2c, n1, n2 = train_two_stage(
        old_train, ou_extra, live_ou, emb16_map, emb567_map)
    print(f'  S1 학습: {n1}개, S2 학습: {n2}개')

    old_ah_results = []
    for s in ah_all:
        pred, conf, proba1 = predict_two_stage(
            s[0], s1s, s1c, s2s, s2c, emb16_map, emb567_map)
        mark = '✓' if pred == s[1] else '✗'
        old_ah_results.append({
            'gt': s[1], 'pred': pred, 'conf': conf,
            'source': s[3], 'fname': s[4]
        })
    old_ah_c, old_ah_t = print_vowel_summary(old_ah_results, '아현 이전')

    # source별
    for src in ['remote', 'live']:
        sr = [r for r in old_ah_results if r['source'] == src]
        if sr:
            sc = sum(1 for r in sr if r['gt'] == r['pred'])
            print(f'    ({src}): {sc}/{len(sr)} ({sc/len(sr)*100:.1f}%)')

    # ══════════════════════════════════════════
    # 비교 B: 이후 (HJ/MT 포함)
    # ══════════════════════════════════════════
    print(f'\n\n{"#"*65}')
    print(f'  비교 B: 이후 (HJ/MT 포함) — TTS + kdg0534 + lynn03 + hj + mt')
    print(f'{"#"*65}')

    new_train = tts + remote_train  # HJ/MT 포함

    # LOSO
    print(f'\n  ── LOSO (이후) ──')
    new_loso_results, new_loso_by_speaker = run_loso(
        new_train, ou_extra, live_ou, emb16_map, emb567_map)
    new_loso_c, new_loso_t = print_vowel_summary(new_loso_results, 'LOSO 이후')

    # 아현 테스트 (이후)
    print(f'\n  ── 아현 테스트 (이후) ──')
    s1s, s1c, s2s, s2c, n1, n2 = train_two_stage(
        new_train, ou_extra, live_ou, emb16_map, emb567_map)
    print(f'  S1 학습: {n1}개, S2 학습: {n2}개')

    new_ah_results = []
    for s in ah_all:
        pred, conf, proba1 = predict_two_stage(
            s[0], s1s, s1c, s2s, s2c, emb16_map, emb567_map)
        new_ah_results.append({
            'gt': s[1], 'pred': pred, 'conf': conf,
            'source': s[3], 'fname': s[4]
        })
    new_ah_c, new_ah_t = print_vowel_summary(new_ah_results, '아현 이후')

    for src in ['remote', 'live']:
        sr = [r for r in new_ah_results if r['source'] == src]
        if sr:
            sc = sum(1 for r in sr if r['gt'] == r['pred'])
            print(f'    ({src}): {sc}/{len(sr)} ({sc/len(sr)*100:.1f}%)')

    # ══════════════════════════════════════════
    # 최종 비교표
    # ══════════════════════════════════════════
    print(f'\n\n{"═"*65}')
    print(f'  최종 비교: HJ/MT 추가 효과')
    print(f'{"═"*65}')

    print(f'\n  {"지표":<25s} {"이전":>10s} {"이후":>10s} {"변화":>10s}')
    print(f'  {"─"*25} {"─"*10} {"─"*10} {"─"*10}')

    old_loso_pct = old_loso_c / old_loso_t * 100
    new_loso_pct = new_loso_c / new_loso_t * 100
    old_ah_pct = old_ah_c / old_ah_t * 100
    new_ah_pct = new_ah_c / new_ah_t * 100
    diff_loso = new_loso_pct - old_loso_pct
    diff_ah = new_ah_pct - old_ah_pct

    print(f'  {"LOSO 전체":<25s} {old_loso_pct:9.1f}% {new_loso_pct:9.1f}% {diff_loso:+9.1f}%p')
    print(f'  {"아현 전체":<25s} {old_ah_pct:9.1f}% {new_ah_pct:9.1f}% {diff_ah:+9.1f}%p')

    # 모음별 비교
    print(f'\n  모음별 비교 (아현 테스트):')
    print(f'  {"모음":<6s} {"이전":>12s} {"이후":>12s} {"변화":>8s}')
    print(f'  {"─"*6} {"─"*12} {"─"*12} {"─"*8}')
    for v in VOWELS:
        old_v = [r for r in old_ah_results if r['gt'] == v]
        new_v = [r for r in new_ah_results if r['gt'] == v]
        if not old_v:
            continue
        old_vc = sum(1 for r in old_v if r['gt'] == r['pred'])
        new_vc = sum(1 for r in new_v if r['gt'] == r['pred'])
        old_vpct = old_vc / len(old_v) * 100
        new_vpct = new_vc / len(new_v) * 100
        d = new_vpct - old_vpct
        print(f'  {v:<6s} {old_vc:3d}/{len(old_v):2d} ({old_vpct:4.0f}%)'
              f'  {new_vc:3d}/{len(new_v):2d} ({new_vpct:4.0f}%)'
              f'  {d:+6.0f}%p')

    # 아현 상세 오류 (이후)
    print(f'\n  아현 상세 결과 (이후):')
    for s, r in zip(ah_all, new_ah_results):
        mark = '✓' if r['gt'] == r['pred'] else '✗'
        print(f'    {r["source"]:6s} {r["fname"]:30s} {r["gt"]}→{r["pred"]} {mark} ({r["conf"]:.0%})')


if __name__ == '__main__':
    main()
