"""Layer Probing Experiment.

각 레이어별로 임베딩 추출 → SVM 학습 → 정확도 측정.
어떤 레이어가 모음 정보를 가장 잘 담는지 확인.

사용법:
  python -m vowel_recognition.method_6_embedding.layer_probing \
    --audio_dir vowel_recognition/dataset
"""

import sys
import os
import argparse
import time
import wave
import hashlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]

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
    if len(parts) >= 3: meta['speaker'] = parts[2]
    return meta


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


def extract_all_layers(model_name, samples, audio_dir):
    """모든 레이어의 임베딩을 한 번에 추출. 캐시 활용."""
    import torch
    from transformers import AutoModel, AutoFeatureExtractor

    cache_dir = os.path.dirname(__file__)
    cache_key = f"allayers_{model_name}_{os.path.abspath(audio_dir)}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"layer_cache_{cache_hash}.npz")

    filenames = [f for f, _, _ in samples]

    if os.path.exists(cache_path):
        print(f"캐시 로드: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cached_files = list(data['filenames'])
        if set(filenames) <= set(cached_files):
            idx_map = {f: i for i, f in enumerate(cached_files)}
            n_layers = int(data['n_layers'])
            all_embeddings = {}
            for layer in range(n_layers + 1):  # 0~12
                key = f"layer_{layer}"
                if key in data:
                    emb = data[key]
                    all_embeddings[layer] = np.array([emb[idx_map[f]] for f in filenames])
            print(f"캐시에서 {len(all_embeddings)} 레이어 x {len(filenames)} 파일 로드.")
            return all_embeddings

    # 모델 로드
    print(f"모델 로드: {model_name}...")
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    target_sr = 16000

    n_layers = model.config.num_hidden_layers  # 12
    # {layer: [embeddings]}
    layer_embeddings = {i: [] for i in range(n_layers + 1)}

    print(f"\n{len(samples)}개 파일 x {n_layers+1} 레이어 임베딩 추출 중...")
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

        hidden_states = outputs.hidden_states  # (13, 1, T, 768)

        for layer in range(n_layers + 1):
            hs = hidden_states[layer].squeeze(0)  # (T, 768)
            # 에너지 상위 50% 프레임
            energy = hs.norm(dim=1)
            k = max(1, len(energy) // 2)
            top_idx = torch.topk(energy, k).indices
            emb = hs[top_idx].mean(dim=0).numpy().astype(np.float32)
            layer_embeddings[layer].append(emb)

        if (idx + 1) % 100 == 0 or idx == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (idx + 1) * (len(samples) - idx - 1)
            print(f"  [{idx+1:3d}/{len(samples)}] ETA: {eta:.0f}s", flush=True)

    total = time.perf_counter() - t_start
    print(f"추출 완료: {total:.1f}초\n")

    # numpy 변환
    all_embeddings = {}
    save_dict = {'filenames': np.array(filenames, dtype=object),
                 'n_layers': np.array(n_layers)}
    for layer in range(n_layers + 1):
        emb_array = np.array(layer_embeddings[layer], dtype=np.float32)
        all_embeddings[layer] = emb_array
        save_dict[f'layer_{layer}'] = emb_array

    np.savez(cache_path, **save_dict)
    print(f"캐시 저장: {cache_path}")

    return all_embeddings


def eval_stratified_quick(X, y, n_splits=5):
    """빠른 Stratified K-Fold 평가. 정확도만 반환."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    correct = 0
    total = 0
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


def eval_loso_quick(X, y, speakers):
    """빠른 LOSO 평가. 정확도만 반환."""
    unique = sorted(set(speakers))
    correct = 0
    total = 0
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


def main():
    parser = argparse.ArgumentParser(description="Layer Probing Experiment")
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--model', default='facebook/wav2vec2-base')
    args = parser.parse_args()

    audio_dir = args.audio_dir
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

    print(f"데이터셋: {len(samples)}개, 모델: {args.model}\n")

    all_embeddings = extract_all_layers(args.model, samples, audio_dir)

    n_layers = max(all_embeddings.keys())

    # 단일 레이어 probing
    print(f"{'='*60}")
    print("단일 레이어별 정확도")
    print(f"{'='*60}")
    print(f"  {'Layer':>6s}  {'Stratified':>10s}  {'LOSO':>10s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}")

    strat_results = {}
    loso_results = {}

    for layer in range(n_layers + 1):
        X = all_embeddings[layer]
        acc_strat = eval_stratified_quick(X, labels)
        acc_loso = eval_loso_quick(X, labels, speakers)
        strat_results[layer] = acc_strat
        loso_results[layer] = acc_loso
        marker = ""
        if acc_strat == max(strat_results.values()):
            marker = " <-- best stratified"
        print(f"  {layer:6d}  {acc_strat*100:9.1f}%  {acc_loso*100:9.1f}%{marker}")

    # 최적 레이어
    best_strat_layer = max(strat_results, key=strat_results.get)
    best_loso_layer = max(loso_results, key=loso_results.get)

    print(f"\n최적 레이어:")
    print(f"  Stratified: layer {best_strat_layer} ({strat_results[best_strat_layer]*100:.1f}%)")
    print(f"  LOSO:       layer {best_loso_layer} ({loso_results[best_loso_layer]*100:.1f}%)")

    # 주요 레이어 범위 조합도 테스트
    print(f"\n{'='*60}")
    print("레이어 범위 조합 정확도")
    print(f"{'='*60}")
    print(f"  {'Layers':>12s}  {'Stratified':>10s}  {'LOSO':>10s}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}")

    combos = [
        (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
        (3, 6), (5, 10), (4, 9), (6, 11),
    ]

    combo_results = []
    for start, end in combos:
        layers = list(range(start, end + 1))
        X = np.mean([all_embeddings[l] for l in layers], axis=0)
        acc_strat = eval_stratified_quick(X, labels)
        acc_loso = eval_loso_quick(X, labels, speakers)
        label = f"{start}-{end}"
        combo_results.append((label, acc_strat, acc_loso))
        print(f"  {label:>12s}  {acc_strat*100:9.1f}%  {acc_loso*100:9.1f}%")

    # 그래프 (텍스트)
    print(f"\n{'='*60}")
    print("Stratified Accuracy by Layer")
    print(f"{'='*60}")
    max_acc = max(strat_results.values())
    for layer in range(n_layers + 1):
        acc = strat_results[layer]
        bar_len = int(acc / max_acc * 40)
        bar = '#' * bar_len
        best = ' *' if acc == max_acc else ''
        print(f"  L{layer:2d} |{bar} {acc*100:.1f}%{best}")

    print(f"\n{'='*60}")
    print("LOSO Accuracy by Layer")
    print(f"{'='*60}")
    max_acc_loso = max(loso_results.values())
    for layer in range(n_layers + 1):
        acc = loso_results[layer]
        bar_len = int(acc / max_acc_loso * 40)
        bar = '#' * bar_len
        best = ' *' if acc == max_acc_loso else ''
        print(f"  L{layer:2d} |{bar} {acc*100:.1f}%{best}")


if __name__ == '__main__':
    main()
