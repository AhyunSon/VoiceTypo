"""한국어 fine-tuned wav2vec2 모델 레이어 프로빙.

kresnik/wav2vec2-large-xlsr-korean (24 layers, 1024d)
Kkonjeong/wav2vec2-base-korean (12 layers, 768d)

기존 layer_probing.py와 동일한 파이프라인:
  오디오 → hidden_states 추출 → energy top-50% 풀링 → SVM(RBF) → Stratified/LOSO

사용법:
  python vowel_recognition/method_6_embedding/probe_korean_models.py
"""

import sys, os, time, hashlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]
AUDIO_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

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


def parse_vowel_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    first = stem.split('_')[0]
    if first in VOWELS:
        return first
    if len(first) == 1:
        return syllable_to_vowel(first)
    return None


def parse_metadata(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    meta = {}
    if len(parts) >= 3:
        meta['speaker'] = parts[2]
    return meta


def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        import wave
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


def extract_all_layers(model_name, samples, audio_dir):
    """모든 레이어의 임베딩 추출. 캐시 활용."""
    import torch
    from transformers import Wav2Vec2Model, AutoFeatureExtractor

    cache_dir = os.path.dirname(os.path.abspath(__file__))
    cache_key = f"allayers_{model_name}_{os.path.abspath(audio_dir)}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"layer_cache_{cache_hash}.npz")

    filenames = [f for f, _, _ in samples]

    if os.path.exists(cache_path):
        print(f"  cache found: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cached_files = list(data['filenames'])
        if set(filenames) <= set(cached_files):
            idx_map = {f: i for i, f in enumerate(cached_files)}
            n_layers = int(data['n_layers'])
            all_embeddings = {}
            for layer in range(n_layers + 1):
                key = f"layer_{layer}"
                if key in data:
                    emb = data[key]
                    all_embeddings[layer] = np.array([emb[idx_map[f]] for f in filenames])
            print(f"  loaded {len(all_embeddings)} layers x {len(filenames)} files from cache")
            return all_embeddings

    # 모델 로드 — CTC fine-tuned 모델에서도 base encoder만 추출
    print(f"  loading model: {model_name}...")
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    target_sr = 16000

    n_layers = model.config.num_hidden_layers
    layer_embeddings = {i: [] for i in range(n_layers + 1)}

    print(f"  extracting {len(samples)} files x {n_layers+1} layers...")
    t_start = time.perf_counter()

    for idx, (filename, vowel, meta) in enumerate(samples):
        filepath = os.path.join(audio_dir, filename)
        audio, sr = load_audio(filepath)

        if sr != target_sr:
            ratio = target_sr / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            aidx = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[aidx]

        inputs = fe(audio, sampling_rate=target_sr, return_tensors="pt", padding=False)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for layer in range(n_layers + 1):
            hs = hidden_states[layer].squeeze(0)  # (T, dim)
            energy = hs.norm(dim=1)
            k = max(1, len(energy) // 2)
            top_idx = torch.topk(energy, k).indices
            emb = hs[top_idx].mean(dim=0).numpy().astype(np.float32)
            layer_embeddings[layer].append(emb)

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (idx + 1) * (len(samples) - idx - 1)
            print(f"    [{idx+1:3d}/{len(samples)}] ETA: {eta:.0f}s", flush=True)

    total = time.perf_counter() - t_start
    print(f"  done: {total:.1f}s")

    # 캐시 저장
    all_embeddings = {}
    save_dict = {'filenames': np.array(filenames, dtype=object),
                 'n_layers': np.array(n_layers)}
    for layer in range(n_layers + 1):
        emb_array = np.array(layer_embeddings[layer], dtype=np.float32)
        all_embeddings[layer] = emb_array
        save_dict[f'layer_{layer}'] = emb_array

    np.savez(cache_path, **save_dict)
    print(f"  cache saved: {cache_path}")

    return all_embeddings


def eval_stratified(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    correct = total = 0
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_test)
        correct += sum(preds == y[test_idx])
        total += len(test_idx)
    return correct / total


def eval_loso(X, y, speakers):
    unique = sorted(set(speakers))
    correct = total = 0
    for held_out in unique:
        test_mask = speakers == held_out
        train_mask = ~test_mask
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)
        correct += sum(preds == y[test_mask])
        total += sum(test_mask)
    return correct / total


def eval_loso_per_vowel(X, y, speakers, target_vowels=None):
    """LOSO with per-vowel accuracy breakdown."""
    unique_speakers = sorted(set(speakers))
    if target_vowels is None:
        target_vowels = sorted(set(y))

    vowel_correct = {v: 0 for v in target_vowels}
    vowel_total = {v: 0 for v in target_vowels}

    for held_out in unique_speakers:
        test_mask = speakers == held_out
        train_mask = ~test_mask
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)

        for i, is_test in enumerate(test_mask):
            if is_test and y[i] in target_vowels:
                vowel_total[y[i]] += 1
                if preds[sum(test_mask[:i+1])-1] == y[i]:
                    vowel_correct[y[i]] += 1

    # Simpler approach: collect all test predictions
    all_preds = []
    all_true = []
    for held_out in unique_speakers:
        test_mask = speakers == held_out
        train_mask = ~test_mask
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)
        all_preds.extend(preds)
        all_true.extend(y[test_mask])

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    result = {}
    for v in target_vowels:
        mask = all_true == v
        if mask.sum() > 0:
            result[v] = (all_preds[mask] == v).sum() / mask.sum()
        else:
            result[v] = 0.0
    return result


def eval_ou_binary_loso(X, y, speakers):
    """오/우 이진 LOSO 정확도."""
    mask = np.isin(y, ['오', '우'])
    X_ou = X[mask]
    y_ou = y[mask]
    sp_ou = speakers[mask]
    if len(X_ou) == 0:
        return 0.0, 0.0, 0.0
    unique = sorted(set(sp_ou))
    all_preds, all_true = [], []
    for held_out in unique:
        test_m = sp_ou == held_out
        train_m = ~test_m
        if test_m.sum() == 0 or train_m.sum() == 0:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_ou[train_m])
        X_test = scaler.transform(X_ou[test_m])
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
        clf.fit(X_train, y_ou[train_m])
        preds = clf.predict(X_test)
        all_preds.extend(preds)
        all_true.extend(y_ou[test_m])
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    total_acc = (all_preds == all_true).mean()
    oh_mask = all_true == '오'
    oo_mask = all_true == '우'
    oh_acc = (all_preds[oh_mask] == '오').mean() if oh_mask.sum() > 0 else 0
    oo_acc = (all_preds[oo_mask] == '우').mean() if oo_mask.sum() > 0 else 0
    return total_acc, oh_acc, oo_acc


def run_probing(model_name, samples, labels, speakers, audio_dir):
    """한 모델에 대해 전체 레이어 프로빙 실행."""
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*70}")

    all_embeddings = extract_all_layers(model_name, samples, audio_dir)
    n_layers = max(all_embeddings.keys())
    dim = all_embeddings[0].shape[1]
    print(f"  {n_layers+1} layers, {dim}d embeddings\n")

    # 1. 단일 레이어 probing (7모음 전체)
    print(f"  {'Layer':>6s}  {'Strat':>7s}  {'LOSO':>7s}  {'OU_bin':>7s}  {'Oh':>5s}  {'Oo':>5s}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*5}")

    strat_results = {}
    loso_results = {}
    ou_results = {}

    for layer in range(n_layers + 1):
        X = all_embeddings[layer]
        acc_strat = eval_stratified(X, labels)
        acc_loso = eval_loso(X, labels, speakers)
        ou_total, oh_acc, oo_acc = eval_ou_binary_loso(X, labels, speakers)

        strat_results[layer] = acc_strat
        loso_results[layer] = acc_loso
        ou_results[layer] = (ou_total, oh_acc, oo_acc)

        print(f"  L{layer:4d}  {acc_strat*100:6.1f}%  {acc_loso*100:6.1f}%  "
              f"{ou_total*100:6.1f}%  {oh_acc*100:4.1f}%  {oo_acc*100:4.1f}%")

    best_loso = max(loso_results, key=loso_results.get)
    best_ou = max(ou_results, key=lambda k: ou_results[k][0])

    print(f"\n  Best 7-vowel LOSO: L{best_loso} ({loso_results[best_loso]*100:.1f}%)")
    print(f"  Best O/U binary:  L{best_ou} ({ou_results[best_ou][0]*100:.1f}%  "
          f"Oh={ou_results[best_ou][1]*100:.1f}%  Oo={ou_results[best_ou][2]*100:.1f}%)")

    # 2. 레이어 범위 조합
    print(f"\n  Layer range combinations:")
    print(f"  {'Range':>12s}  {'Strat':>7s}  {'LOSO':>7s}  {'OU_bin':>7s}")
    print(f"  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}")

    ranges = []
    for w in [2, 3, 4]:
        for s in range(n_layers + 1 - w):
            ranges.append((s, s + w - 1))

    combo_results = []
    for start, end in ranges:
        layers = list(range(start, end + 1))
        X = np.mean([all_embeddings[l] for l in layers], axis=0)
        acc_strat = eval_stratified(X, labels)
        acc_loso = eval_loso(X, labels, speakers)
        ou_total, _, _ = eval_ou_binary_loso(X, labels, speakers)
        combo_results.append((f"L{start}-{end}", acc_strat, acc_loso, ou_total))

    # Top 5 by LOSO
    combo_results.sort(key=lambda x: x[2], reverse=True)
    for label, s, l, ou in combo_results[:8]:
        print(f"  {label:>12s}  {s*100:6.1f}%  {l*100:6.1f}%  {ou*100:6.1f}%")

    return {
        'strat': strat_results,
        'loso': loso_results,
        'ou': ou_results,
        'n_layers': n_layers,
        'dim': dim,
    }


def main():
    audio_dir = os.path.abspath(AUDIO_DIR)
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    all_files = sorted([f for f in os.listdir(audio_dir)
                        if os.path.splitext(f)[1].lower() in audio_exts])

    samples = []
    for f in all_files:
        vowel = parse_vowel_from_filename(f)
        if vowel is None:
            continue
        meta = parse_metadata(f)
        samples.append((f, vowel, meta))

    labels = np.array([v for _, v, _ in samples])
    speakers = np.array([m.get('speaker', '?') for _, _, m in samples])

    print(f"Dataset: {len(samples)} files, {len(set(labels))} vowels, {len(set(speakers))} speakers")
    for v in sorted(set(labels)):
        print(f"  {v}: {sum(labels == v)}")

    models = [
        'kresnik/wav2vec2-large-xlsr-korean',
        'Kkonjeong/wav2vec2-base-korean',
    ]

    results = {}
    for model_name in models:
        try:
            results[model_name] = run_probing(model_name, samples, labels, speakers, audio_dir)
        except Exception as e:
            print(f"\n  ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # 비교 요약
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*70}")

        # 기존 XLSR-53 baseline
        print(f"\n  Reference: XLSR-53 L16 LOSO=89.7%, OU binary L5-7=88.1%\n")

        for name, r in results.items():
            short = name.split('/')[-1]
            best_l = max(r['loso'], key=r['loso'].get)
            best_ou = max(r['ou'], key=lambda k: r['ou'][k][0])
            print(f"  {short}")
            print(f"    Layers: {r['n_layers']+1}, Dim: {r['dim']}")
            print(f"    Best 7-vowel LOSO: L{best_l} = {r['loso'][best_l]*100:.1f}%")
            ou_t, ou_oh, ou_oo = r['ou'][best_ou]
            print(f"    Best O/U binary:   L{best_ou} = {ou_t*100:.1f}% (Oh={ou_oh*100:.1f}%, Oo={ou_oo*100:.1f}%)")
            print()


if __name__ == '__main__':
    main()
