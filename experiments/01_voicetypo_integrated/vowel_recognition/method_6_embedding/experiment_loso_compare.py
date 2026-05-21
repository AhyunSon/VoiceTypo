"""LOSO 비교 실험: A vs B vs D

조건:
  A) TTS 525 + remote 210
  B) remote 210 only
  D) TTS 210 (다운샘플) + remote 210

평가: remote 화자 3명 기준 LOSO
  - kdg0534 hold-out → 나머지로 학습
  - lynn03 hold-out → 나머지로 학습
  - 아현 hold-out → 나머지로 학습

핵심 지표: 전체 정확도, 모음별 정확도 (특히 오/우), 혼동행렬
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

REMOTE_DIRS = [
    'vowel-remote-001_kdg0534 (1)',
    'vowel-remote-001_lynn03 (1)',
    'vowel-remote-001_아현 (1)',
]

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}


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


# ═══════════════════════════════════════
#  데이터 수집
# ═══════════════════════════════════════

def collect_tts_data():
    """기존 TTS dataset 수집. returns [(path, vowel, speaker, domain)]"""
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
        # 모음 파싱
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
    """Remote 녹음 데이터 수집. returns [(path, vowel, speaker, domain)]"""
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


# ═══════════════════════════════════════
#  임베딩 추출 (캐시 활용)
# ═══════════════════════════════════════

def extract_embeddings(paths, cache_name='loso_compare'):
    """모든 경로에 대해 Layer 16 + Layer 5-7 임베딩 추출."""
    cache_path = os.path.join(BASE, f'cache_{cache_name}.npz')

    # 캐시 확인
    cached_emb16 = {}
    cached_emb567 = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        cp = list(data['paths'])
        for i, p in enumerate(cp):
            cached_emb16[str(p)] = data['emb16'][i]
            cached_emb567[str(p)] = data['emb567'][i]
        print(f'  캐시에서 {len(cp)}개 로드')

    need = [p for p in paths if str(p) not in cached_emb16]
    if need:
        print(f'  새로 추출: {len(need)}개')
        model_name = 'facebook/wav2vec2-large-xlsr-53'
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.eval()

        t0 = time.perf_counter()
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

            emb16 = pool(h[16].squeeze(0))
            emb5 = pool(h[5].squeeze(0))
            emb6 = pool(h[6].squeeze(0))
            emb7 = pool(h[7].squeeze(0))
            emb567 = (emb5 + emb6 + emb7) / 3.0

            cached_emb16[str(p)] = emb16
            cached_emb567[str(p)] = emb567

            if (i + 1) % 30 == 0 or i == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (i + 1) * (len(need) - i - 1)
                print(f'    [{i+1}/{len(need)}] ETA: {eta:.0f}s', flush=True)

        print(f'  추출 완료: {time.perf_counter() - t0:.1f}초')

        # 캐시 저장
        all_paths = list(cached_emb16.keys())
        np.savez(cache_path,
                 paths=np.array(all_paths, dtype=object),
                 emb16=np.array([cached_emb16[p] for p in all_paths]),
                 emb567=np.array([cached_emb567[p] for p in all_paths]))
        print(f'  캐시 저장: {cache_path}')

    emb16_arr = np.array([cached_emb16[str(p)] for p in paths])
    emb567_arr = np.array([cached_emb567[str(p)] for p in paths])
    return emb16_arr, emb567_arr


# ═══════════════════════════════════════
#  2단계 분류 + LOSO 평가
# ═══════════════════════════════════════

def twostage_predict(emb16, emb567, s1_scaler, s1_clf, s2_scaler, s2_clf, s2_target):
    """2단계 예측."""
    X1 = s1_scaler.transform(emb16.reshape(1, -1))
    pred1 = s1_clf.predict(X1)[0]
    proba1 = s1_clf.predict_proba(X1)[0]

    if pred1 in s2_target:
        X2 = s2_scaler.transform(emb567.reshape(1, -1))
        pred2 = s2_clf.predict(X2)[0]
        proba2 = s2_clf.predict_proba(X2)[0]
        return pred2, float(max(proba2))
    else:
        return pred1, float(max(proba1))


def train_twostage(X_emb16, X_emb567, y, s2_target=['오', '우']):
    """2단계 분류기 학습."""
    # Stage 1: Layer 16 → 7모음 SVM
    s1_scaler = StandardScaler()
    X1 = s1_scaler.fit_transform(X_emb16)
    s1_clf = SVC(kernel='rbf', C=10.0, gamma='scale',
                 probability=True, random_state=42)
    s1_clf.fit(X1, y)

    # Stage 2: Layer 5-7 → 오/우 SVM
    ou_mask = np.isin(y, s2_target)
    s2_scaler = StandardScaler()
    X2 = s2_scaler.fit_transform(X_emb567[ou_mask])
    s2_clf = SVC(kernel='rbf', C=10.0, gamma='scale',
                 probability=True, random_state=42)
    s2_clf.fit(X2, y[ou_mask])

    return s1_scaler, s1_clf, s2_scaler, s2_clf, s2_target


def run_remote_loso(train_pool, remote_samples, emb16_map, emb567_map, condition_name):
    """Remote 화자 LOSO 평가.

    train_pool: LOSO에서 hold-out 화자를 제외한 후 학습에 쓸 기본 데이터
    remote_samples: remote 데이터 전체 (LOSO 대상)
    """
    remote_speakers = sorted(set(s[2] for s in remote_samples))
    all_results = []
    results_by_speaker = {}

    for held_out in remote_speakers:
        # 테스트: held_out 화자의 remote 데이터
        test_samples = [s for s in remote_samples if s[2] == held_out]
        # 학습: train_pool + held_out 제외한 remote
        other_remote = [s for s in remote_samples if s[2] != held_out]
        train_samples = train_pool + other_remote

        # 임베딩 조립
        train_paths = [s[0] for s in train_samples]
        train_labels = np.array([s[1] for s in train_samples])
        test_paths = [s[0] for s in test_samples]
        test_labels = np.array([s[1] for s in test_samples])

        X_train_16 = np.array([emb16_map[p] for p in train_paths])
        X_train_567 = np.array([emb567_map[p] for p in train_paths])
        X_test_16 = np.array([emb16_map[p] for p in test_paths])
        X_test_567 = np.array([emb567_map[p] for p in test_paths])

        # 학습
        s1_scaler, s1_clf, s2_scaler, s2_clf, s2_target = train_twostage(
            X_train_16, X_train_567, train_labels)

        # 예측
        speaker_results = []
        for i in range(len(test_samples)):
            pred, conf = twostage_predict(
                X_test_16[i], X_test_567[i],
                s1_scaler, s1_clf, s2_scaler, s2_clf, s2_target)
            entry = (test_labels[i], pred, conf)
            all_results.append(entry)
            speaker_results.append(entry)

        results_by_speaker[held_out] = speaker_results

        # 간단 요약
        acc = sum(1 for g, p, _ in speaker_results if g == p) / len(speaker_results) * 100
        ou_results = [(g, p) for g, p, _ in speaker_results if g in ['오', '우']]
        ou_acc = sum(1 for g, p in ou_results if g == p) / len(ou_results) * 100 if ou_results else -1
        print(f'    hold-out [{held_out:8s}]: {acc:5.1f}%  '
              f'(train={len(train_samples)}, test={len(test_samples)})  '
              f'오/우={ou_acc:.0f}%')

    return all_results, results_by_speaker


def print_results(all_results, results_by_speaker, condition_name):
    """상세 결과 출력."""
    total = len(all_results)
    correct = sum(1 for g, p, _ in all_results if g == p)

    print(f'\n  {"─"*55}')
    print(f'  {condition_name} 결과')
    print(f'  {"─"*55}')
    print(f'  전체 정확도: {correct}/{total} ({correct/total*100:.1f}%)')

    # 모음별
    print(f'\n  모음별:')
    for v in VOWELS:
        vr = [(g, p, c) for g, p, c in all_results if g == v]
        if not vr:
            continue
        vc = sum(1 for g, p, _ in vr if g == p)
        vt = len(vr)
        errors = {}
        for g, p, _ in vr:
            if g != p:
                errors[p] = errors.get(p, 0) + 1
        err_str = ', '.join(f'{k}({v})' for k, v in sorted(errors.items(), key=lambda x: -x[1]))
        mark = '  ◀' if vc / vt < 0.7 else ''
        print(f'    {v}: {vc:2d}/{vt:2d} ({vc/vt*100:5.1f}%)  {err_str}{mark}')

    # 오/우 세부
    ou_results = [(g, p) for g, p, _ in all_results if g in ['오', '우']]
    if ou_results:
        ou_correct = sum(1 for g, p in ou_results if g == p)
        oh_r = [(g, p) for g, p in ou_results if g == '오']
        oo_r = [(g, p) for g, p in ou_results if g == '우']
        oh_acc = sum(1 for g, p in oh_r if g == p) / len(oh_r) * 100 if oh_r else 0
        oo_acc = sum(1 for g, p in oo_r if g == p) / len(oo_r) * 100 if oo_r else 0
        print(f'\n  오/우 상세: {ou_correct}/{len(ou_results)} ({ou_correct/len(ou_results)*100:.1f}%)')
        print(f'    오: {oh_acc:.1f}%  우: {oo_acc:.1f}%')

    # 혼동행렬
    print(f'\n  혼동행렬:')
    header = '정답\\예측'
    print(f'  {header:>8s}', end='')
    for v in VOWELS:
        print(f' {v:>3s}', end='')
    print('   acc')
    for v in VOWELS:
        vr = [(g, p) for g, p, _ in all_results if g == v]
        if not vr:
            continue
        vt = len(vr)
        print(f'  {v:>8s}', end='')
        for v2 in VOWELS:
            cnt = sum(1 for g, p in vr if p == v2)
            if cnt == 0:
                print(f' {"·":>3s}', end='')
            elif v == v2:
                print(f' \033[92m{cnt:>3d}\033[0m', end='')
            else:
                print(f' \033[91m{cnt:>3d}\033[0m', end='')
        vc = sum(1 for g, p in vr if g == p)
        print(f'  {vc/vt*100:4.0f}%')

    # 화자별
    print(f'\n  화자별:')
    for spk in sorted(results_by_speaker.keys()):
        sr = results_by_speaker[spk]
        sc = sum(1 for g, p, _ in sr if g == p)
        st = len(sr)
        ou_r = [(g, p) for g, p, _ in sr if g in ['오', '우']]
        ou_c = sum(1 for g, p in ou_r if g == p)
        ou_t = len(ou_r)
        ou_str = f'오/우={ou_c}/{ou_t}' if ou_t > 0 else ''
        print(f'    {spk:8s}: {sc:2d}/{st:2d} ({sc/st*100:5.1f}%)  {ou_str}')


# ═══════════════════════════════════════
#  메인
# ═══════════════════════════════════════

def main():
    print('=' * 60)
    print('  LOSO 비교 실험: A vs B vs D')
    print('=' * 60)

    # ── 데이터 수집 ──
    print('\n[1] 데이터 수집')
    tts_samples = collect_tts_data()
    remote_samples = collect_remote_data()

    print(f'  TTS: {len(tts_samples)}개')
    tts_speakers = sorted(set(s[2] for s in tts_samples))
    for spk in tts_speakers:
        cnt = sum(1 for s in tts_samples if s[2] == spk)
        print(f'    {spk}: {cnt}개')

    print(f'  Remote: {len(remote_samples)}개')
    remote_speakers = sorted(set(s[2] for s in remote_samples))
    for spk in remote_speakers:
        cnt = sum(1 for s in remote_samples if s[2] == spk)
        vowel_counts = {}
        for s in remote_samples:
            if s[2] == spk:
                vowel_counts[s[1]] = vowel_counts.get(s[1], 0) + 1
        vc_str = ' '.join(f'{v}:{vowel_counts.get(v,0)}' for v in VOWELS)
        print(f'    {spk}: {cnt}개  ({vc_str})')

    # ── 임베딩 추출 ──
    print('\n[2] 임베딩 추출')
    all_paths = [s[0] for s in tts_samples + remote_samples]
    all_paths_unique = list(dict.fromkeys(all_paths))  # 중복 제거, 순서 유지
    emb16_arr, emb567_arr = extract_embeddings(all_paths_unique)

    # path → embedding 맵
    emb16_map = {p: emb16_arr[i] for i, p in enumerate(all_paths_unique)}
    emb567_map = {p: emb567_arr[i] for i, p in enumerate(all_paths_unique)}

    # ── 조건별 실험 ──
    print('\n[3] LOSO 실험 (remote 화자 3명 hold-out)')

    # ── 조건 A: TTS 525 + remote 210 ──
    print(f'\n{"="*60}')
    print(f'  조건 A: TTS {len(tts_samples)} + remote {len(remote_samples)}')
    print(f'{"="*60}')
    results_A, by_speaker_A = run_remote_loso(
        tts_samples, remote_samples, emb16_map, emb567_map, 'A')
    print_results(results_A, by_speaker_A, 'A: TTS+remote 전체')

    # ── 조건 B: remote 210만 ──
    print(f'\n{"="*60}')
    print(f'  조건 B: remote {len(remote_samples)} only')
    print(f'{"="*60}')
    results_B, by_speaker_B = run_remote_loso(
        [], remote_samples, emb16_map, emb567_map, 'B')
    print_results(results_B, by_speaker_B, 'B: remote only')

    # ── 조건 D: TTS 다운샘플 210 + remote 210 ──
    # 모음별 균등 다운샘플
    np.random.seed(42)
    tts_by_vowel = {}
    for s in tts_samples:
        tts_by_vowel.setdefault(s[1], []).append(s)

    target_per_vowel = len(remote_samples) // len(VOWELS)  # 30
    tts_downsampled = []
    for v in VOWELS:
        pool_v = tts_by_vowel.get(v, [])
        if len(pool_v) <= target_per_vowel:
            tts_downsampled.extend(pool_v)
        else:
            indices = np.random.choice(len(pool_v), target_per_vowel, replace=False)
            tts_downsampled.extend([pool_v[i] for i in indices])

    print(f'\n{"="*60}')
    print(f'  조건 D: TTS {len(tts_downsampled)} (다운샘플) + remote {len(remote_samples)}')
    print(f'{"="*60}')
    # TTS 다운샘플 모음별 분포
    ds_counts = {}
    for s in tts_downsampled:
        ds_counts[s[1]] = ds_counts.get(s[1], 0) + 1
    print(f'  TTS 다운샘플 분포: {" ".join(f"{v}:{ds_counts.get(v,0)}" for v in VOWELS)}')

    results_D, by_speaker_D = run_remote_loso(
        tts_downsampled, remote_samples, emb16_map, emb567_map, 'D')
    print_results(results_D, by_speaker_D, 'D: TTS(균형)+remote')

    # ═══════════════════════════════════════
    #  최종 비교 요약
    # ═══════════════════════════════════════
    print(f'\n\n{"#"*60}')
    print(f'  최종 비교 요약')
    print(f'{"#"*60}\n')

    conditions = [
        ('A: TTS+remote 전체', results_A),
        ('B: remote only', results_B),
        ('D: TTS(균형)+remote', results_D),
    ]

    # 표 헤더
    print(f'  {"조건":<22s} {"전체":>6s} {"아":>5s} {"어":>5s} {"오":>5s} '
          f'{"우":>5s} {"으":>5s} {"이":>5s} {"에":>5s} {"오/우":>6s}')
    print(f'  {"─"*22} {"─"*6} {"─"*5} {"─"*5} {"─"*5} '
          f'{"─"*5} {"─"*5} {"─"*5} {"─"*5} {"─"*6}')

    for name, results in conditions:
        total = len(results)
        correct = sum(1 for g, p, _ in results if g == p)
        overall = correct / total * 100

        vowel_accs = []
        for v in VOWELS:
            vr = [(g, p) for g, p, _ in results if g == v]
            if vr:
                va = sum(1 for g, p in vr if g == p) / len(vr) * 100
            else:
                va = 0
            vowel_accs.append(va)

        ou_r = [(g, p) for g, p, _ in results if g in ['오', '우']]
        ou_acc = sum(1 for g, p in ou_r if g == p) / len(ou_r) * 100 if ou_r else 0

        print(f'  {name:<22s} {overall:5.1f}%', end='')
        for va in vowel_accs:
            if va < 70:
                print(f' \033[91m{va:4.0f}%\033[0m', end='')
            elif va >= 90:
                print(f' \033[92m{va:4.0f}%\033[0m', end='')
            else:
                print(f' {va:4.0f}%', end='')
        if ou_acc < 70:
            print(f' \033[91m{ou_acc:5.1f}%\033[0m')
        else:
            print(f' {ou_acc:5.1f}%')

    # 화자별 비교
    print(f'\n  화자별 비교:')
    print(f'  {"화자":<10s}', end='')
    for name, _ in conditions:
        short_name = name.split(':')[0].strip()
        print(f'  {short_name:>12s}', end='')
    print()

    for spk in remote_speakers:
        print(f'  {spk:<10s}', end='')
        for _, results in conditions:
            sr = [(g, p) for g, p, _ in results
                  if any(s[2] == spk and s[1] == g for s in remote_samples)]
            # 좀 더 정확하게: by_speaker에서 가져오기
        print()

    # 간단하게 다시
    print(f'\n  화자별 비교 (전체 정확도):')
    all_by_speaker = [by_speaker_A, by_speaker_B, by_speaker_D]
    cond_names = ['A', 'B', 'D']
    print(f'  {"화자":<10s}  {"A":>8s}  {"B":>8s}  {"D":>8s}')
    print(f'  {"─"*10}  {"─"*8}  {"─"*8}  {"─"*8}')
    for spk in remote_speakers:
        print(f'  {spk:<10s}', end='')
        for bs in all_by_speaker:
            sr = bs.get(spk, [])
            if sr:
                sc = sum(1 for g, p, _ in sr if g == p)
                st = len(sr)
                print(f'  {sc:2d}/{st:2d} {sc/st*100:4.0f}%', end='')
            else:
                print(f'  {"—":>8s}', end='')
        print()

    # 화자별 오/우 비교
    print(f'\n  화자별 오/우 정확도:')
    print(f'  {"화자":<10s}  {"A":>8s}  {"B":>8s}  {"D":>8s}')
    print(f'  {"─"*10}  {"─"*8}  {"─"*8}  {"─"*8}')
    for spk in remote_speakers:
        print(f'  {spk:<10s}', end='')
        for bs in all_by_speaker:
            sr = bs.get(spk, [])
            ou = [(g, p) for g, p, _ in sr if g in ['오', '우']]
            if ou:
                oc = sum(1 for g, p in ou if g == p)
                ot = len(ou)
                print(f'  {oc:2d}/{ot:2d} {oc/ot*100:4.0f}%', end='')
            else:
                print(f'  {"—":>8s}', end='')
        print()

    print(f'\n{"="*60}')
    print('  실험 완료')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
