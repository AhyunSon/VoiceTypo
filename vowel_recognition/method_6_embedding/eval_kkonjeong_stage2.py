"""Stage 2 교체 실험: Kkonjeong/wav2vec2-base-korean L0 vs XLSR L5-7.

학습 데이터(525개 중 오/우 168개)로 SVM 학습 후,
새 화자 실제 녹음 25개에서 오/우 분류 성능 비교.

사용법:
  python vowel_recognition/method_6_embedding/eval_kkonjeong_stage2.py
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


def pool_energy_top50(emb):
    """energy top-50% pooling (numpy)."""
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


# ────────────────────────────────────────
# 학습 데이터에서 오/우 임베딩 로드 (캐시)
# ────────────────────────────────────────
def load_train_ou_from_cache(model_name, audio_dir, layers_to_extract):
    """layer_cache에서 오/우 학습 데이터 임베딩 로드."""
    cache_key = f"allayers_{model_name}_{os.path.abspath(audio_dir)}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(OUT_DIR, f"layer_cache_{cache_hash}.npz")

    if not os.path.exists(cache_path):
        return None, None, None, None

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

    # 요청된 레이어 임베딩 추출
    embeddings = {}
    for layer in layers_to_extract:
        key = f"layer_{layer}"
        if key in data:
            embeddings[layer] = data[key][ou_indices]

    return embeddings, ou_labels, ou_speakers, cached_files


# ────────────────────────────────────────
# 실제 녹음 파일에서 임베딩 추출
# ────────────────────────────────────────
def extract_live_embeddings(model, fe, test_files, layers_to_extract):
    """모델로 live recording 임베딩 추출."""
    results = {layer: [] for layer in layers_to_extract}

    for wav_path, vowel, speaker in test_files:
        audio, sr = read_wav(wav_path)
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


def eval_config(name, X_train, y_train, X_test, test_labels, test_spks):
    """SVM 학습 + 테스트 + 결과 출력."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)

    print(f"\n  [{name}]")
    print(f"  {'-'*55}")

    for spk in sorted(set(test_spks)):
        mask = test_spks == spk
        for v in ['오', '우']:
            v_mask = mask & (test_labels == v)
            if not v_mask.any():
                continue
            v_preds = preds[v_mask]
            n_correct = (v_preds == v).sum()
            acc = n_correct * 100.0 / len(v_preds)
            print(f"    {spk} {v}: {n_correct}/{len(v_preds)} = {acc:3.0f}%  {list(v_preds)}")

    o_mask = test_labels == '오'
    u_mask = test_labels == '우'
    o_acc = (preds[o_mask] == '오').sum() * 100.0 / o_mask.sum() if o_mask.sum() > 0 else 0
    u_acc = (preds[u_mask] == '우').sum() * 100.0 / u_mask.sum() if u_mask.sum() > 0 else 0
    total_acc = (preds == test_labels).sum() * 100.0 / len(test_labels)
    print(f"    -- 합계 --")
    print(f"    오: {o_acc:.0f}%  우: {u_acc:.0f}%  전체: {total_acc:.0f}%")

    return {'oh': o_acc, 'oo': u_acc, 'total': total_acc}


def loso_eval(name, X, y, speakers):
    """LOSO 평가."""
    correct = total = 0
    oh_c = oh_t = oo_c = oo_t = 0
    for spk in sorted(set(speakers)):
        test_m = speakers == spk
        train_m = ~test_m
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_m])
        X_te = scaler.transform(X[test_m])
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        clf.fit(X_tr, y[train_m])
        preds = clf.predict(X_te)
        correct += (preds == y[test_m]).sum()
        total += test_m.sum()
        for i, is_test in enumerate(np.where(test_m)[0]):
            if y[is_test] == '오':
                oh_t += 1
                if preds[i] == '오': oh_c += 1
            elif y[is_test] == '우':
                oo_t += 1
                if preds[i] == '우': oo_c += 1

    acc = correct * 100.0 / total
    oh_acc = oh_c * 100.0 / oh_t if oh_t > 0 else 0
    oo_acc = oo_c * 100.0 / oo_t if oo_t > 0 else 0
    print(f"  {name:40s} LOSO={acc:5.1f}% (오:{oh_acc:5.1f}% 우:{oo_acc:5.1f}%)")
    return acc


def main():
    # ── 테스트 파일 목록 ──
    test_files = []
    for speaker, path, vowels in [
        ('서울여성', os.path.join(LIVE_DIR, 'session_20260310_145230'), ['오', '우']),
        ('서울여성', os.path.join(LIVE_DIR, 'session_20260310_151524'), ['오', '우']),
        ('서울여성', os.path.join(LIVE_DIR, 'session_20260310_153047'), ['우']),
        ('경상도여성', os.path.join(LIVE_DIR, 'speaker_F_20s_gyeongsang'), ['오', '우']),
        ('20대남성', os.path.join(LIVE_DIR, 'speaker_M_20s'), ['우']),
    ]:
        for v in vowels:
            for i in range(1, 20):
                wav = os.path.join(path, f'{v}_{i:02d}.wav')
                if os.path.exists(wav):
                    test_files.append((wav, v, speaker))

    test_labels = np.array([v for _, v, _ in test_files])
    test_spks = np.array([s for _, _, s in test_files])

    n_oh = (test_labels == '오').sum()
    n_oo = (test_labels == '우').sum()
    print(f"테스트 파일: {len(test_files)}개 (오:{n_oh}, 우:{n_oo})")
    for spk in sorted(set(test_spks)):
        m = test_spks == spk
        print(f"  {spk}: 오 {(test_labels[m]=='오').sum()}, 우 {(test_labels[m]=='우').sum()}")

    # ── 모델 1: XLSR-53 (기존) ──
    print("\n[1/2] XLSR-53 로딩...")
    xlsr_name = 'facebook/wav2vec2-large-xlsr-53'
    xlsr_fe = AutoFeatureExtractor.from_pretrained(xlsr_name)
    xlsr_model = Wav2Vec2Model.from_pretrained(xlsr_name)
    xlsr_model.eval()

    # 학습 데이터 캐시에서 로드
    xlsr_train, ou_labels, ou_speakers, _ = load_train_ou_from_cache(
        xlsr_name, os.path.abspath(AUDIO_DIR), [5, 6, 7, 16])

    if xlsr_train is None:
        print("XLSR-53 캐시 없음! layer_probing.py를 먼저 실행하세요.")
        return

    X_train_l567 = np.mean([xlsr_train[l] for l in [5, 6, 7]], axis=0)
    print(f"  학습 데이터: 오 {(ou_labels=='오').sum()}, 우 {(ou_labels=='우').sum()}")

    # 실제 녹음 임베딩 추출
    print("  live recording 임베딩 추출 (XLSR-53)...")
    xlsr_live = extract_live_embeddings(xlsr_model, xlsr_fe, test_files, [5, 6, 7])
    X_test_l567 = np.mean([xlsr_live[l] for l in [5, 6, 7]], axis=0)

    # ── 모델 2: Kkonjeong (새 후보) ──
    print("\n[2/2] Kkonjeong/wav2vec2-base-korean 로딩...")
    kk_name = 'Kkonjeong/wav2vec2-base-korean'
    kk_fe = AutoFeatureExtractor.from_pretrained(kk_name)
    kk_model = Wav2Vec2Model.from_pretrained(kk_name)
    kk_model.eval()

    # 학습 데이터 캐시에서 로드
    kk_train, kk_labels, kk_speakers, _ = load_train_ou_from_cache(
        kk_name, os.path.abspath(AUDIO_DIR), [0])

    if kk_train is None:
        print("Kkonjeong 캐시 없음! probe_korean_models.py를 먼저 실행하세요.")
        return

    X_train_kk_l0 = kk_train[0]

    # 실제 녹음 임베딩 추출
    print("  live recording 임베딩 추출 (Kkonjeong)...")
    kk_live = extract_live_embeddings(kk_model, kk_fe, test_files, [0])
    X_test_kk_l0 = kk_live[0]

    # ── 모델 3: kresnik도 테스트 ──
    print("\n[bonus] kresnik/wav2vec2-large-xlsr-korean L22, L24...")
    kr_name = 'kresnik/wav2vec2-large-xlsr-korean'
    kr_train, kr_labels, kr_speakers, _ = load_train_ou_from_cache(
        kr_name, os.path.abspath(AUDIO_DIR), [22, 24])

    kr_live_embs = None
    if kr_train is not None:
        kr_fe = AutoFeatureExtractor.from_pretrained(kr_name)
        kr_model = Wav2Vec2Model.from_pretrained(kr_name)
        kr_model.eval()
        print("  live recording 임베딩 추출 (kresnik)...")
        kr_live_embs = extract_live_embeddings(kr_model, kr_fe, test_files, [22, 24])

    # ════════════════════════════════════════
    # 결과 비교
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  LOSO 평가 (학습 데이터 168개)")
    print("=" * 60)

    loso_eval("XLSR L5-7 (현재 Stage 2)", X_train_l567, ou_labels, ou_speakers)
    loso_eval("Kkonjeong L0", X_train_kk_l0, kk_labels, kk_speakers)
    if kr_train is not None:
        loso_eval("kresnik L22", kr_train[22], kr_labels, kr_speakers)
        loso_eval("kresnik L24", kr_train[24], kr_labels, kr_speakers)

    print("\n" + "=" * 60)
    print("  새 화자 실제 녹음 테스트")
    print("=" * 60)

    results = {}
    results['xlsr_l567'] = eval_config(
        "XLSR L5-7 (현재 Stage 2)",
        X_train_l567, ou_labels, X_test_l567, test_labels, test_spks)

    results['kk_l0'] = eval_config(
        "Kkonjeong L0 (후보)",
        X_train_kk_l0, kk_labels, X_test_kk_l0, test_labels, test_spks)

    if kr_train is not None and kr_live_embs is not None:
        results['kr_l22'] = eval_config(
            "kresnik L22",
            kr_train[22], kr_labels, kr_live_embs[22], test_labels, test_spks)
        results['kr_l24'] = eval_config(
            "kresnik L24",
            kr_train[24], kr_labels, kr_live_embs[24], test_labels, test_spks)

    # ── 하이브리드: XLSR L5-7 + Kkonjeong L0 concat ──
    X_train_hybrid = np.hstack([X_train_l567, X_train_kk_l0])
    X_test_hybrid = np.hstack([X_test_l567, X_test_kk_l0])
    results['hybrid'] = eval_config(
        "Hybrid: XLSR L5-7 + Kkonjeong L0 (1792d)",
        X_train_hybrid, ou_labels, X_test_hybrid, test_labels, test_spks)

    # ── 요약 ──
    print("\n" + "=" * 60)
    print("  요약")
    print("=" * 60)
    print(f"  {'구성':40s} {'오':>5s} {'우':>5s} {'전체':>5s}")
    print(f"  {'─'*40} {'─'*5} {'─'*5} {'─'*5}")
    for key, r in results.items():
        label = {
            'xlsr_l567': 'XLSR L5-7 (현재)',
            'kk_l0': 'Kkonjeong L0',
            'kr_l22': 'kresnik L22',
            'kr_l24': 'kresnik L24',
            'hybrid': 'XLSR L5-7 + Kkonjeong L0',
        }.get(key, key)
        print(f"  {label:40s} {r['oh']:4.0f}% {r['oo']:4.0f}% {r['total']:4.0f}%")


if __name__ == '__main__':
    main()
