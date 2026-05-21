"""Stage 2 비교: remote 데이터(lynn03, kdg0534, 아현) + live 녹음 전체 테스트.

학습 데이터(525개 중 오/우 168개, Anna/김동규/이은서)로 SVM 학습,
새 화자 remote + live 녹음에서 오/우 분류 성능 비교.

사용법:
  python vowel_recognition/method_6_embedding/eval_kkonjeong_stage2_v2.py
"""

import sys, os, io, hashlib, wave, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import Wav2Vec2Model, AutoFeatureExtractor

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(OUT_DIR, '..', 'dataset')
LIVE_DIR = os.path.join(OUT_DIR, 'live_recordings')

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]
_MEDIAL_TO_VOWEL = {0: '아', 1: '애', 4: '어', 5: '에', 8: '오', 13: '우', 18: '으', 20: '이'}


def syllable_to_vowel(ch):
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    return _MEDIAL_TO_VOWEL.get((code % (28 * 21)) // 28)


def parse_vowel(fn):
    stem = os.path.splitext(fn)[0]
    first = stem.split('_')[0]
    if first in VOWELS:
        return first
    if len(first) == 1:
        return syllable_to_vowel(first)
    return None


def parse_vowel_remote(fn):
    """remote 파일명: speaker_###_모음_음절_길이.wav"""
    parts = os.path.splitext(fn)[0].split('_')
    if len(parts) >= 3:
        v = parts[2]
        if v in VOWELS:
            return v
    return None


def pool_energy_top50(emb):
    norms = np.linalg.norm(emb, axis=1)
    k = max(1, len(norms) // 2)
    top_idx = np.argpartition(norms, -k)[-k:]
    return emb[top_idx].mean(axis=0).astype(np.float32)


def read_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        d = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return d, sr


def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        return read_wav(path)
    from pydub import AudioSegment
    seg = AudioSegment.from_file(path)
    seg = seg.set_channels(1)
    sr = seg.frame_rate
    raw = seg.raw_data
    sw = seg.sample_width
    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    return audio, sr


def load_train_ou_from_cache(model_name, audio_dir, layers_to_extract):
    cache_key = f"allayers_{model_name}_{os.path.abspath(audio_dir)}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(OUT_DIR, f"layer_cache_{cache_hash}.npz")

    if not os.path.exists(cache_path):
        return None, None, None

    data = np.load(cache_path, allow_pickle=True)
    cached_files = list(data['filenames'])

    ou_indices = []
    ou_labels = []
    ou_speakers = []
    for i, f in enumerate(cached_files):
        v = parse_vowel(f)
        if v in ['오', '우']:
            ou_indices.append(i)
            ou_labels.append(v)
            parts = f.split('_')
            ou_speakers.append(parts[2] if len(parts) >= 3 else 'unknown')

    ou_indices = np.array(ou_indices)
    ou_labels = np.array(ou_labels)
    ou_speakers = np.array(ou_speakers)

    embeddings = {}
    for layer in layers_to_extract:
        key = f"layer_{layer}"
        if key in data:
            embeddings[layer] = data[key][ou_indices]

    return embeddings, ou_labels, ou_speakers


def extract_embeddings(model, fe, file_list, layers_to_extract):
    """파일 리스트에서 임베딩 추출. file_list: [(path, vowel, speaker), ...]"""
    results = {layer: [] for layer in layers_to_extract}

    for wav_path, vowel, speaker in file_list:
        audio, sr = load_audio(wav_path)
        if sr != 16000:
            ratio = 16000 / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            aidx = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[aidx]

        inp = fe(audio, sampling_rate=16000, return_tensors='pt', padding=False)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        for layer in layers_to_extract:
            hs = out.hidden_states[layer].squeeze(0).numpy()
            emb = pool_energy_top50(hs)
            results[layer].append(emb)

    for layer in layers_to_extract:
        results[layer] = np.array(results[layer])

    return results


def eval_config(name, X_train, y_train, X_test, test_labels, test_spks, test_types):
    """SVM 학습 + 테스트."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)

    print(f"\n  [{name}]")
    print(f"  {'-'*65}")

    # 화자별 상세
    for spk in sorted(set(test_spks)):
        smask = test_spks == spk
        for v in ['오', '우']:
            v_mask = smask & (test_labels == v)
            if not v_mask.any():
                continue
            v_preds = preds[v_mask]
            n_correct = (v_preds == v).sum()
            acc = n_correct * 100.0 / len(v_preds)
            print(f"    {spk:16s} {v}: {n_correct}/{len(v_preds)} = {acc:3.0f}%")

    # 데이터 타입별 (remote vs live)
    print(f"  {'-'*65}")
    for dtype in sorted(set(test_types)):
        dmask = test_types == dtype
        o_m = dmask & (test_labels == '오')
        u_m = dmask & (test_labels == '우')
        o_acc = (preds[o_m] == '오').sum() * 100.0 / o_m.sum() if o_m.sum() > 0 else -1
        u_acc = (preds[u_m] == '우').sum() * 100.0 / u_m.sum() if u_m.sum() > 0 else -1
        total_m = dmask
        t_acc = (preds[total_m] == test_labels[total_m]).sum() * 100.0 / total_m.sum()
        o_str = f"{o_acc:.0f}%" if o_acc >= 0 else "N/A"
        u_str = f"{u_acc:.0f}%" if u_acc >= 0 else "N/A"
        print(f"    {dtype:16s}  오:{o_str:>4s}  우:{u_str:>4s}  전체:{t_acc:.0f}%")

    # 전체
    o_mask = test_labels == '오'
    u_mask = test_labels == '우'
    o_acc = (preds[o_mask] == '오').sum() * 100.0 / o_mask.sum() if o_mask.sum() > 0 else 0
    u_acc = (preds[u_mask] == '우').sum() * 100.0 / u_mask.sum() if u_mask.sum() > 0 else 0
    total_acc = (preds == test_labels).sum() * 100.0 / len(test_labels)
    print(f"    {'합계':16s}  오:{o_acc:3.0f}%  우:{u_acc:3.0f}%  전체:{total_acc:.0f}%")

    return {'oh': o_acc, 'oo': u_acc, 'total': total_acc}


def main():
    # ══════════════════════════════════════
    # 테스트 파일 수집
    # ══════════════════════════════════════

    test_files = []  # (path, vowel, speaker, type)

    # 1) Remote 데이터 (오/우만)
    remote_dirs = [
        ('kdg0534(R)', os.path.join(OUT_DIR, 'vowel-remote-001_kdg0534 (1)')),
        ('lynn03(R)', os.path.join(OUT_DIR, 'vowel-remote-001_lynn03 (1)')),
        ('아현(R)', os.path.join(OUT_DIR, 'vowel-remote-001_아현 (1)')),
    ]
    for speaker, dirpath in remote_dirs:
        if not os.path.isdir(dirpath):
            # try without space
            dirpath2 = dirpath.replace(' (1)', '(1)')
            if os.path.isdir(dirpath2):
                dirpath = dirpath2
            else:
                print(f"  경고: {dirpath} 없음")
                continue
        for fn in sorted(os.listdir(dirpath)):
            if not fn.endswith('.wav'):
                continue
            v = parse_vowel_remote(fn)
            if v in ['오', '우']:
                test_files.append((os.path.join(dirpath, fn), v, speaker, 'remote'))

    # 2) Live 녹음
    live_sources = [
        ('아현(L)', os.path.join(LIVE_DIR, 'session_20260310_145230'), ['오', '우']),
        ('아현(L)', os.path.join(LIVE_DIR, 'session_20260310_151524'), ['오', '우']),
        ('아현(L)', os.path.join(LIVE_DIR, 'session_20260310_153047'), ['우']),
        ('경상도여성(L)', os.path.join(LIVE_DIR, 'speaker_F_20s_gyeongsang'), ['오', '우']),
        ('20대남성(L)', os.path.join(LIVE_DIR, 'speaker_M_20s'), ['우']),
    ]
    for speaker, path, vowels in live_sources:
        if not os.path.isdir(path):
            continue
        for v in vowels:
            for i in range(1, 20):
                wav = os.path.join(path, f'{v}_{i:02d}.wav')
                if os.path.exists(wav):
                    test_files.append((wav, v, speaker, 'live'))

    test_paths = [t[0] for t in test_files]
    test_labels = np.array([t[1] for t in test_files])
    test_spks = np.array([t[2] for t in test_files])
    test_types = np.array([t[3] for t in test_files])

    n_oh = (test_labels == '오').sum()
    n_oo = (test_labels == '우').sum()
    print(f"테스트 파일 총: {len(test_files)}개 (오:{n_oh}, 우:{n_oo})")
    print(f"  Remote: {(test_types=='remote').sum()}개, Live: {(test_types=='live').sum()}개")
    for spk in sorted(set(test_spks)):
        m = test_spks == spk
        print(f"  {spk:16s}: 오 {(test_labels[m]=='오').sum()}, 우 {(test_labels[m]=='우').sum()}")

    # ══════════════════════════════════════
    # 모델 로드 + 임베딩 추출
    # ══════════════════════════════════════

    # test file list for extraction (without type)
    test_for_extract = [(p, v, s) for p, v, s, _ in test_files]

    # --- XLSR-53 ---
    print("\n[1/3] XLSR-53 로딩...")
    xlsr_name = 'facebook/wav2vec2-large-xlsr-53'
    xlsr_fe = AutoFeatureExtractor.from_pretrained(xlsr_name)
    xlsr_model = Wav2Vec2Model.from_pretrained(xlsr_name)
    xlsr_model.eval()

    xlsr_train, ou_labels, ou_speakers = load_train_ou_from_cache(
        xlsr_name, os.path.abspath(AUDIO_DIR), [5, 6, 7])
    X_train_l567 = np.mean([xlsr_train[l] for l in [5, 6, 7]], axis=0)
    print(f"  학습: 오 {(ou_labels=='오').sum()}, 우 {(ou_labels=='우').sum()}")

    print(f"  테스트 임베딩 추출 ({len(test_files)}개)...", flush=True)
    t0 = time.perf_counter()
    xlsr_test = extract_embeddings(xlsr_model, xlsr_fe, test_for_extract, [5, 6, 7])
    X_test_l567 = np.mean([xlsr_test[l] for l in [5, 6, 7]], axis=0)
    print(f"  완료: {time.perf_counter()-t0:.0f}s")

    del xlsr_model  # 메모리 절약

    # --- Kkonjeong ---
    print("\n[2/3] Kkonjeong/wav2vec2-base-korean 로딩...")
    kk_name = 'Kkonjeong/wav2vec2-base-korean'
    kk_fe = AutoFeatureExtractor.from_pretrained(kk_name)
    kk_model = Wav2Vec2Model.from_pretrained(kk_name)
    kk_model.eval()

    kk_train, kk_labels, kk_speakers = load_train_ou_from_cache(
        kk_name, os.path.abspath(AUDIO_DIR), [0])
    X_train_kk_l0 = kk_train[0]

    print(f"  테스트 임베딩 추출...", flush=True)
    t0 = time.perf_counter()
    kk_test = extract_embeddings(kk_model, kk_fe, test_for_extract, [0])
    X_test_kk_l0 = kk_test[0]
    print(f"  완료: {time.perf_counter()-t0:.0f}s")

    del kk_model

    # --- kresnik ---
    print("\n[3/3] kresnik/wav2vec2-large-xlsr-korean 로딩...")
    kr_name = 'kresnik/wav2vec2-large-xlsr-korean'
    kr_fe = AutoFeatureExtractor.from_pretrained(kr_name)
    kr_model = Wav2Vec2Model.from_pretrained(kr_name)
    kr_model.eval()

    kr_train, kr_labels, kr_speakers = load_train_ou_from_cache(
        kr_name, os.path.abspath(AUDIO_DIR), [22, 24])

    print(f"  테스트 임베딩 추출...", flush=True)
    t0 = time.perf_counter()
    kr_test = extract_embeddings(kr_model, kr_fe, test_for_extract, [22, 24])
    print(f"  완료: {time.perf_counter()-t0:.0f}s")

    del kr_model

    # ══════════════════════════════════════
    # 결과
    # ══════════════════════════════════════
    print("\n" + "=" * 70)
    print("  새 화자 오/우 분류 — remote + live 전체")
    print("=" * 70)

    results = {}

    results['xlsr_l567'] = eval_config(
        "XLSR L5-7 (현재 Stage 2)",
        X_train_l567, ou_labels, X_test_l567, test_labels, test_spks, test_types)

    results['kk_l0'] = eval_config(
        "Kkonjeong L0",
        X_train_kk_l0, kk_labels, X_test_kk_l0, test_labels, test_spks, test_types)

    results['kr_l22'] = eval_config(
        "kresnik L22",
        kr_train[22], kr_labels, kr_test[22], test_labels, test_spks, test_types)

    results['kr_l24'] = eval_config(
        "kresnik L24",
        kr_train[24], kr_labels, kr_test[24], test_labels, test_spks, test_types)

    # Hybrid: XLSR L5-7 + kresnik L24
    X_train_hybrid = np.hstack([X_train_l567, kr_train[24]])
    X_test_hybrid = np.hstack([X_test_l567, kr_test[24]])
    results['hybrid_xlsr_kr'] = eval_config(
        "Hybrid: XLSR L5-7 + kresnik L24 (2048d)",
        X_train_hybrid, ou_labels, X_test_hybrid, test_labels, test_spks, test_types)

    # ── 요약 ──
    print("\n" + "=" * 70)
    print("  요약")
    print("=" * 70)
    print(f"  {'구성':45s} {'오':>5s} {'우':>5s} {'전체':>5s}")
    print(f"  {'─'*45} {'─'*5} {'─'*5} {'─'*5}")
    for key, r in results.items():
        label = {
            'xlsr_l567': 'XLSR L5-7 (현재)',
            'kk_l0': 'Kkonjeong L0',
            'kr_l22': 'kresnik L22',
            'kr_l24': 'kresnik L24',
            'hybrid_xlsr_kr': 'XLSR L5-7 + kresnik L24',
        }.get(key, key)
        print(f"  {label:45s} {r['oh']:4.0f}% {r['oo']:4.0f}% {r['total']:4.0f}%")


if __name__ == '__main__':
    main()
