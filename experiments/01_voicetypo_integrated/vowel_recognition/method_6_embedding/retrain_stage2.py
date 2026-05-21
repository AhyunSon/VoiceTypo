"""Stage 2 (오/우 이진) 재학습: 기존 데이터 + TTS 보강 데이터.

학습: dataset/ (3화자) + dataset/우 오 구분용/ (TTS 4화자)
테스트: live_recordings/ (서울여성, 경상도여성, 20대남성)

사용법:
  python -m vowel_recognition.method_6_embedding.retrain_stage2
"""

import sys, os, io, time, wave, pickle
import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ── 상수 ──
VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]
TARGET = ["오", "우"]
MODEL_NAME = 'facebook/wav2vec2-large-xlsr-53'
STAGE2_LAYERS = [5, 6, 7]

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')
TTS_DIR = os.path.join(DATASET_DIR, '우 오 구분용')
LIVE_DIR = os.path.join(BASE_DIR, 'live_recordings')
MODEL_PATH = os.path.join(BASE_DIR, 'twostage_model.pkl')

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}


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


def parse_speaker(fn):
    parts = os.path.splitext(fn)[0].split('_')
    return parts[2] if len(parts) >= 3 else 'unknown'


def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        with wave.open(path, 'r') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            ch = wf.getnchannels()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            audio = audio.reshape(-1, ch)[:, 0]
        return audio, sr
    else:
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


def pool_energy(frames):
    """에너지 상위 50% 프레임 평균."""
    energy = frames.norm(dim=1)
    k = max(1, len(energy) // 2)
    top_idx = torch.topk(energy, k).indices
    return frames[top_idx].mean(dim=0).numpy().astype(np.float32)


def extract_stage2_embedding(hidden_states):
    """Layer 5,6,7 평균 임베딩 추출."""
    embs = []
    for l in STAGE2_LAYERS:
        embs.append(pool_energy(hidden_states[l].squeeze(0)))
    return np.mean(embs, axis=0)


def extract_stage1_embedding(hidden_states):
    """Layer 16 임베딩 추출."""
    return pool_energy(hidden_states[16].squeeze(0))


def extract_embeddings_batch(fe, model, file_list, label=""):
    """파일 목록에서 (stage1_emb, stage2_emb) 추출."""
    emb1_list, emb2_list = [], []
    t0 = time.perf_counter()

    for i, (filepath, vowel, speaker) in enumerate(file_list):
        audio, sr = load_audio(filepath)

        # 리샘플링 to 16kHz
        if sr != 16000:
            ratio = 16000 / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            idx = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[idx]

        inputs = fe(audio, sampling_rate=16000, return_tensors='pt', padding=False)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        emb1_list.append(extract_stage1_embedding(out.hidden_states))
        emb2_list.append(extract_stage2_embedding(out.hidden_states))

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(file_list) - i - 1)
            print(f"  {label} [{i+1}/{len(file_list)}] ETA: {eta:.0f}s", flush=True)

    total = time.perf_counter() - t0
    print(f"  {label} 완료: {total:.1f}초 ({len(file_list)}개)")
    return np.array(emb1_list), np.array(emb2_list)


def collect_dataset_ou(dataset_dir):
    """기존 dataset에서 오/우 파일 수집."""
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    files = []
    for f in sorted(os.listdir(dataset_dir)):
        if os.path.isdir(os.path.join(dataset_dir, f)):
            continue
        if os.path.splitext(f)[1].lower() not in audio_exts:
            continue
        vowel = parse_vowel(f)
        if vowel in TARGET:
            speaker = parse_speaker(f)
            files.append((os.path.join(dataset_dir, f), vowel, speaker))
    return files


def collect_tts_ou(tts_dir):
    """TTS 보강 데이터에서 오/우 파일 수집."""
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    files = []
    for f in sorted(os.listdir(tts_dir)):
        if os.path.splitext(f)[1].lower() not in audio_exts:
            continue
        vowel = parse_vowel(f)
        if vowel in TARGET:
            speaker = parse_speaker(f)
            files.append((os.path.join(tts_dir, f), vowel, f"TTS_{speaker}"))
    return files


def collect_live_test():
    """live_recordings에서 오/우 테스트 파일 수집."""
    files = []
    test_dirs = [
        ('서울여성', os.path.join(LIVE_DIR, 'session_20260310_145230')),
        ('서울여성_2', os.path.join(LIVE_DIR, 'session_20260310_151524')),
        ('경상도여성', os.path.join(LIVE_DIR, 'speaker_F_20s_gyeongsang')),
        ('20대남성', os.path.join(LIVE_DIR, 'speaker_M_20s')),
        ('20대남성_2', os.path.join(LIVE_DIR, 'session_20260310_153047')),
    ]
    for speaker, dirpath in test_dirs:
        if not os.path.isdir(dirpath):
            continue
        for f in sorted(os.listdir(dirpath)):
            if not f.endswith('.wav'):
                continue
            vowel = f.split('_')[0]
            if vowel in TARGET:
                files.append((os.path.join(dirpath, f), vowel, speaker))
    return files


def evaluate(clf, scaler, X_test, y_test, speakers, label=""):
    """평가 및 결과 출력."""
    X_scaled = scaler.transform(X_test)
    preds = clf.predict(X_scaled)
    proba = clf.predict_proba(X_scaled)

    total_acc = np.mean(preds == y_test) * 100
    oh_mask = y_test == '오'
    oo_mask = y_test == '우'
    oh_acc = np.mean(preds[oh_mask] == '오') * 100 if oh_mask.any() else 0
    oo_acc = np.mean(preds[oo_mask] == '우') * 100 if oo_mask.any() else 0

    print(f"\n  [{label}]")
    print(f"  전체: {total_acc:.1f}%  |  오: {oh_acc:.1f}%  |  우: {oo_acc:.1f}%")

    cm = confusion_matrix(y_test, preds, labels=['오', '우'])
    print(f"  혼동행렬:  오→오:{cm[0,0]}  오→우:{cm[0,1]}  |  우→오:{cm[1,0]}  우→우:{cm[1,1]}")

    # 화자별
    unique_spk = sorted(set(speakers))
    for spk in unique_spk:
        mask = speakers == spk
        if not mask.any():
            continue
        spk_preds = preds[mask]
        spk_y = y_test[mask]
        spk_acc = np.mean(spk_preds == spk_y) * 100
        details = []
        for v in TARGET:
            vm = spk_y == v
            if vm.any():
                vacc = np.mean(spk_preds[vm] == v) * 100
                details.append(f"{v}:{vacc:.0f}%({vm.sum()})")
        print(f"    {spk}: {spk_acc:.0f}%  {' '.join(details)}")

    return total_acc, preds, proba


def main():
    print("=" * 60)
    print("  Stage 2 재학습: 기존 + TTS 보강 데이터")
    print("=" * 60)

    # ── 1. 데이터 수집 ──
    existing = collect_dataset_ou(DATASET_DIR)
    tts = collect_tts_ou(TTS_DIR)
    live_test = collect_live_test()

    print(f"\n기존 데이터: {len(existing)}개")
    for v in TARGET:
        n = sum(1 for _, vv, _ in existing if vv == v)
        print(f"  {v}: {n}개")

    print(f"\nTTS 보강: {len(tts)}개")
    for v in TARGET:
        n = sum(1 for _, vv, _ in tts if vv == v)
        print(f"  {v}: {n}개")

    print(f"\n테스트 (live): {len(live_test)}개")
    for v in TARGET:
        n = sum(1 for _, vv, _ in live_test if vv == v)
        print(f"  {v}: {n}개")

    train_existing = existing
    train_combined = existing + tts

    # ── 2. XLSR-53 로드 ──
    print(f"\nXLSR-53 로드 중...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    print("로드 완료.\n")

    # ── 3. 임베딩 추출 ──
    print("=" * 60)
    print("  임베딩 추출")
    print("=" * 60)

    # 기존 데이터 임베딩 추출을 위한 캐시
    cache_path = os.path.join(BASE_DIR, 'retrain_stage2_cache.npz')
    all_train = train_combined + live_test
    all_paths = [p for p, _, _ in all_train]

    cached = {}
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        cached_paths = list(data['paths'])
        for i, p in enumerate(cached_paths):
            cached[p] = (data['emb1'][i], data['emb2'][i])
        print(f"캐시에서 {len(cached)}개 로드")

    # 캐시에 없는 것만 추출
    need_extract = [(p, v, s) for p, v, s in all_train if p not in cached]
    if need_extract:
        print(f"\n새로 추출할 파일: {len(need_extract)}개")
        emb1_new, emb2_new = extract_embeddings_batch(fe, model, need_extract, "추출")
        for i, (p, _, _) in enumerate(need_extract):
            cached[p] = (emb1_new[i], emb2_new[i])

        # 캐시 업데이트
        all_cached_paths = list(cached.keys())
        all_emb1 = np.array([cached[p][0] for p in all_cached_paths])
        all_emb2 = np.array([cached[p][1] for p in all_cached_paths])
        np.savez(cache_path,
                 paths=np.array(all_cached_paths, dtype=object),
                 emb1=all_emb1, emb2=all_emb2)
        print(f"캐시 저장: {len(all_cached_paths)}개")

    # 데이터셋별 임베딩 정리
    def get_embs(file_list):
        e2 = np.array([cached[p][1] for p, _, _ in file_list])
        labels = np.array([v for _, v, _ in file_list])
        speakers = np.array([s for _, _, s in file_list])
        return e2, labels, speakers

    X_existing, y_existing, s_existing = get_embs(train_existing)
    X_combined, y_combined, s_combined = get_embs(train_combined)
    X_test, y_test, s_test = get_embs(live_test)

    print(f"\n기존 학습 임베딩: {X_existing.shape}")
    print(f"합산 학습 임베딩: {X_combined.shape}")
    print(f"테스트 임베딩: {X_test.shape}")

    # ── 4. 기존 모델 (Before) 평가 ──
    print("\n" + "=" * 60)
    print("  Before: 기존 twostage_model.pkl Stage 2")
    print("=" * 60)

    with open(MODEL_PATH, 'rb') as f:
        orig_model = pickle.load(f)
    orig_scaler = orig_model['stage2']['scaler']
    orig_clf = orig_model['stage2']['clf']

    evaluate(orig_clf, orig_scaler, X_test, y_test, s_test, "기존 모델 → live 테스트")

    # ── 5. 기존 데이터만으로 재학습 (베이스라인 재현) ──
    print("\n" + "=" * 60)
    print("  베이스라인: 기존 데이터만 재학습")
    print("=" * 60)

    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(X_existing)
    clf_base = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    clf_base.fit(X_train_base, y_existing)

    evaluate(clf_base, scaler_base, X_test, y_test, s_test, "기존 데이터 재학습 → live 테스트")

    # ── 6. 기존 + TTS 합산 재학습 (After) ──
    print("\n" + "=" * 60)
    print("  After: 기존 + TTS 합산 재학습")
    print("=" * 60)

    scaler_new = StandardScaler()
    X_train_new = scaler_new.fit_transform(X_combined)
    clf_new = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    clf_new.fit(X_train_new, y_combined)

    acc_after, _, _ = evaluate(clf_new, scaler_new, X_test, y_test, s_test,
                               "기존+TTS 재학습 → live 테스트")

    # ── 7. 교차 검증 (TTS를 hold-out) ──
    print("\n" + "=" * 60)
    print("  교차 검증: 기존으로 학습 → TTS 테스트")
    print("=" * 60)

    X_tts, y_tts, s_tts = get_embs(tts)
    evaluate(clf_base, scaler_base, X_tts, y_tts, s_tts,
             "기존 데이터 학습 → TTS 테스트")

    # ── 8. 모델 저장 ──
    print("\n" + "=" * 60)
    print("  모델 저장")
    print("=" * 60)

    # 기존 모델 백업
    backup_path = MODEL_PATH.replace('.pkl', '_backup.pkl')
    if not os.path.exists(backup_path):
        with open(MODEL_PATH, 'rb') as f:
            backup_data = f.read()
        with open(backup_path, 'wb') as f:
            f.write(backup_data)
        print(f"기존 모델 백업: {backup_path}")

    # Stage 1은 유지, Stage 2만 교체
    new_model = dict(orig_model)
    new_model['stage2'] = {
        'scaler': scaler_new,
        'clf': clf_new,
        'layers': STAGE2_LAYERS,
        'target_vowels': TARGET,
        'description': 'XLSR-53 L5-7 mean -> SVM (오/우), retrained with TTS augmentation',
    }
    new_model['n_train_stage2'] = len(y_combined)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(new_model, f)
    print(f"새 모델 저장: {MODEL_PATH}")
    print(f"  Stage 2 학습 데이터: {len(y_combined)}개 (기존 {len(y_existing)} + TTS {len(y_tts)})")
    print(f"  Support vectors: {clf_new.n_support_}")

    print("\n완료!")


if __name__ == '__main__':
    main()
