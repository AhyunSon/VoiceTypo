"""Method 6: 임베딩 + 분류기 학습 및 평가.

사용법:
  # Stratified 5-fold (기본)
  python -m vowel_recognition.method_6_embedding.train_and_eval \
    --audio_dir vowel_recognition/dataset

  # Leave-One-Speaker-Out (화자 독립성 평가)
  python -m vowel_recognition.method_6_embedding.train_and_eval \
    --audio_dir vowel_recognition/dataset --eval_mode loso

  # 분류기 변경
  --classifier svm|mlp|lr

  # 모델 변경
  --model facebook/wav2vec2-base
  --model facebook/hubert-base-ls960

  # 레이어 지정
  --layers 6-9
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
    if len(parts) >= 1: meta['syllable'] = parts[0]
    if len(parts) >= 2: meta['gender'] = parts[1]
    if len(parts) >= 3: meta['speaker'] = parts[2]
    if len(parts) >= 4: meta['number'] = parts[3]
    if len(parts) >= 5: meta['condition'] = parts[4]
    if len(parts) >= 6: meta['pitch'] = parts[5]
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


def print_confusion_matrix(results, title=""):
    if not results:
        print("결과 없음.")
        return
    correct = sum(1 for gt, pred, _ in results if gt == pred)
    total = len(results)

    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
    print(f"{'='*60}")
    print(f"전체 정확도: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}\n")

    matrix = {v: {v2: 0 for v2 in VOWELS} for v in VOWELS}
    counts = {v: 0 for v in VOWELS}
    for gt, pred, _ in results:
        if gt in matrix and pred in VOWELS:
            matrix[gt][pred] += 1
            counts[gt] += 1

    header = "정답\\예측"
    print(f"  {header:>8s}", end="")
    for v in VOWELS:
        print(f"  {v:>4s}", end="")
    print("   정확도")
    print(f"  {'─'*8}", end="")
    for _ in VOWELS:
        print(f"  {'─'*4}", end="")
    print(f"  {'─'*6}")

    for v in VOWELS:
        print(f"  {v:>8s}", end="")
        row_total = counts[v]
        for v2 in VOWELS:
            cnt = matrix[v][v2]
            if cnt == 0:
                print(f"  {'·':>4s}", end="")
            elif v == v2:
                print(f"  \033[92m{cnt:>4d}\033[0m", end="")
            else:
                print(f"  \033[91m{cnt:>4d}\033[0m", end="")
        if row_total > 0:
            acc = matrix[v][v] / row_total * 100
            print(f"  {acc:5.1f}%")
        else:
            print(f"    -")

    print(f"\n모음별 상세:")
    for v in VOWELS:
        if counts[v] == 0:
            continue
        acc = matrix[v][v] / counts[v] * 100
        errors = [(v2, matrix[v][v2]) for v2 in VOWELS if v2 != v and matrix[v][v2] > 0]
        err_str = ", ".join(f"{v2}({c})" for v2, c in errors) if errors else "없음"
        print(f"  {v}: {acc:.0f}% ({matrix[v][v]}/{counts[v]})  오인: {err_str}")

    avg_conf_correct = np.mean([c for gt, pred, c in results if gt == pred]) if correct > 0 else 0
    avg_conf_wrong = np.mean([c for gt, pred, c in results if gt != pred]) if correct < total else 0
    print(f"\n평균 신뢰도:")
    print(f"  정답: {avg_conf_correct:.3f}")
    if correct < total:
        print(f"  오답: {avg_conf_wrong:.3f}")


def parse_layers(s):
    """'6-9' → (6,7,8,9), '8' → (8,)"""
    if '-' in s:
        a, b = s.split('-')
        return tuple(range(int(a), int(b) + 1))
    return (int(s),)


def get_cache_path(model_name, layers, audio_dir):
    """캐시 파일 경로 생성."""
    key = f"{model_name}_{layers}_{os.path.abspath(audio_dir)}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    cache_dir = os.path.dirname(__file__)
    return os.path.join(cache_dir, f"embeddings_cache_{h}.npz")


def extract_all_embeddings(extractor, samples, audio_dir, cache_path):
    """모든 샘플의 임베딩 추출 (캐시 활용)."""
    filenames = [f for f, _, _ in samples]

    # 캐시 확인
    if os.path.exists(cache_path):
        print(f"캐시 로드: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        cached_files = list(data['filenames'])
        cached_embeddings = data['embeddings']
        # 모든 파일이 캐시에 있는지 확인
        if set(filenames) <= set(cached_files):
            idx_map = {f: i for i, f in enumerate(cached_files)}
            embeddings = np.array([cached_embeddings[idx_map[f]] for f in filenames])
            print(f"캐시에서 {len(embeddings)}개 임베딩 로드 완료.")
            return embeddings

    # 추출
    print(f"\n{len(samples)}개 파일 임베딩 추출 중...")
    embeddings = []
    t_start = time.perf_counter()

    for i, (filename, vowel, meta) in enumerate(samples):
        filepath = os.path.join(audio_dir, filename)
        audio, sr = load_audio(filepath)

        t0 = time.perf_counter()
        emb = extractor.extract(audio, sr)
        elapsed = (time.perf_counter() - t0) * 1000

        embeddings.append(emb)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed_total = time.perf_counter() - t_start
            eta = elapsed_total / (i + 1) * (len(samples) - i - 1)
            print(f"  [{i+1:3d}/{len(samples)}] {filename:40s} {elapsed:.0f}ms  "
                  f"ETA: {eta:.0f}s", flush=True)

    embeddings = np.array(embeddings, dtype=np.float32)
    total_time = time.perf_counter() - t_start
    print(f"임베딩 추출 완료: {total_time:.1f}초 "
          f"(평균 {total_time/len(samples)*1000:.0f}ms/파일)\n")

    # 캐시 저장
    np.savez(cache_path,
             filenames=np.array(filenames, dtype=object),
             embeddings=embeddings)
    print(f"캐시 저장: {cache_path}")

    return embeddings


def eval_stratified(embeddings, labels, speakers, conditions, classifier_type, n_splits=5):
    """Stratified K-Fold 평가."""
    from vowel_recognition.method_6_embedding.classifier import EmbeddingVowelClassifier

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results = []
    results_by_speaker = {}
    results_by_condition = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = EmbeddingVowelClassifier(classifier_type)
        clf.train(X_train, y_train)
        preds, confs = clf.predict(X_test)

        for i, ti in enumerate(test_idx):
            gt = y_test[i]
            pred = preds[i]
            conf = confs[i]
            entry = (gt, pred, conf)
            all_results.append(entry)

            spk = speakers[ti]
            results_by_speaker.setdefault(spk, []).append(entry)
            cond = conditions[ti]
            results_by_condition.setdefault(cond, []).append(entry)

        fold_acc = sum(1 for g, p, _ in zip(y_test, preds, confs) if g == p) / len(y_test)
        print(f"  Fold {fold+1}: {fold_acc*100:.1f}%")

    return all_results, results_by_speaker, results_by_condition


def eval_loso(embeddings, labels, speakers, conditions, classifier_type):
    """Leave-One-Speaker-Out 평가."""
    from vowel_recognition.method_6_embedding.classifier import EmbeddingVowelClassifier

    unique_speakers = sorted(set(speakers))
    all_results = []
    results_by_speaker = {}
    results_by_condition = {}

    for held_out in unique_speakers:
        test_mask = np.array([s == held_out for s in speakers])
        train_mask = ~test_mask

        X_train, X_test = embeddings[train_mask], embeddings[test_mask]
        y_train, y_test = labels[train_mask], labels[test_mask]

        clf = EmbeddingVowelClassifier(classifier_type)
        clf.train(X_train, y_train)
        preds, confs = clf.predict(X_test)

        for i in range(len(y_test)):
            gt = y_test[i]
            pred = preds[i]
            conf = confs[i]
            entry = (gt, pred, conf)
            all_results.append(entry)

            results_by_speaker.setdefault(held_out, []).append(entry)
            # condition
            test_indices = np.where(test_mask)[0]
            cond = conditions[test_indices[i]]
            results_by_condition.setdefault(cond, []).append(entry)

        fold_acc = sum(1 for g, p in zip(y_test, preds) if g == p) / len(y_test)
        print(f"  Hold-out [{held_out}]: {fold_acc*100:.1f}% ({len(y_test)}개)")

    return all_results, results_by_speaker, results_by_condition


def main():
    parser = argparse.ArgumentParser(description="Method 6: 임베딩 + 분류기")
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--model', default='facebook/wav2vec2-base',
                        help='모델 (wav2vec2-base, hubert-base-ls960 등)')
    parser.add_argument('--layers', default='6-9',
                        help='레이어 범위 (예: 6-9, 8)')
    parser.add_argument('--classifier', default='svm',
                        choices=['svm', 'mlp', 'lr'])
    parser.add_argument('--pooling', default='mean',
                        choices=['mean', 'mean_std'],
                        help='풀링 방식 (mean: 768d, mean_std: 1536d)')
    parser.add_argument('--formants', action='store_true',
                        help='F1/F2 포먼트 특성 추가')
    parser.add_argument('--eval_mode', default='stratified',
                        choices=['stratified', 'loso', 'both'])
    parser.add_argument('--detail', action='store_true')
    args = parser.parse_args()

    audio_dir = args.audio_dir
    layers = parse_layers(args.layers)

    # 파일 수집
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

    if not samples:
        print("평가할 샘플이 없습니다.")
        sys.exit(1)

    labels = np.array([v for _, v, _ in samples])
    speakers = np.array([m.get('speaker', '?') for _, _, m in samples])
    conditions = np.array([m.get('condition', '?') for _, _, m in samples])

    vowel_counts = {}
    for v in labels:
        vowel_counts[v] = vowel_counts.get(v, 0) + 1

    print(f"데이터셋: {len(samples)}개")
    print(f"  화자: {', '.join(sorted(set(speakers)))}")
    print(f"  조건: {', '.join(sorted(set(conditions)))}")
    print(f"  모음별: {', '.join(f'{v}({c})' for v, c in sorted(vowel_counts.items()))}")
    print(f"\n설정:")
    print(f"  모델: {args.model}")
    print(f"  레이어: {layers}")
    print(f"  풀링: {args.pooling}")
    print(f"  포먼트: {args.formants}")
    print(f"  분류기: {args.classifier}")
    print(f"  평가: {args.eval_mode}")

    # 임베딩 추출
    from vowel_recognition.method_6_embedding.features import EmbeddingExtractor
    extractor = EmbeddingExtractor(
        model_name=args.model, layers=layers,
        pooling=args.pooling, use_formants=args.formants)

    cache_key_extra = f"_{args.pooling}_f{int(args.formants)}"
    cache_path = get_cache_path(args.model + cache_key_extra, layers, audio_dir)
    embeddings = extract_all_embeddings(extractor, samples, audio_dir, cache_path)

    print(f"\n임베딩 shape: {embeddings.shape}")

    # 평가
    run_stratified = args.eval_mode in ('stratified', 'both')
    run_loso = args.eval_mode in ('loso', 'both')

    if run_stratified:
        print(f"\n{'#'*60}")
        print("Stratified 5-Fold Cross-Validation")
        print(f"{'#'*60}")
        results, by_speaker, by_condition = eval_stratified(
            embeddings, labels, speakers, conditions, args.classifier)
        print_confusion_matrix(results, "Stratified 5-Fold 결과")

        if len(by_speaker) > 1:
            print(f"\n{'─'*60}")
            print("화자별 정확도:")
            for spk in sorted(by_speaker):
                r = by_speaker[spk]
                acc = sum(1 for g, p, _ in r if g == p) / len(r) * 100
                print(f"  {spk:12s}: {acc:5.1f}% ({len(r)}개)")

        if len(by_condition) > 1:
            print(f"\n{'─'*60}")
            print("조건별 정확도:")
            for cond in sorted(by_condition):
                r = by_condition[cond]
                acc = sum(1 for g, p, _ in r if g == p) / len(r) * 100
                print(f"  {cond:12s}: {acc:5.1f}% ({len(r)}개)")

        if args.detail and len(by_speaker) > 1:
            for spk in sorted(by_speaker):
                print_confusion_matrix(by_speaker[spk], f"화자: {spk}")

    if run_loso:
        print(f"\n{'#'*60}")
        print("Leave-One-Speaker-Out (LOSO)")
        print(f"{'#'*60}")
        results, by_speaker, by_condition = eval_loso(
            embeddings, labels, speakers, conditions, args.classifier)
        print_confusion_matrix(results, "LOSO 결과")

        if len(by_speaker) > 1:
            print(f"\n{'─'*60}")
            print("화자별 정확도 (각 화자가 테스트셋):")
            for spk in sorted(by_speaker):
                r = by_speaker[spk]
                acc = sum(1 for g, p, _ in r if g == p) / len(r) * 100
                print(f"  {spk:12s}: {acc:5.1f}% ({len(r)}개)")

        if args.detail and len(by_speaker) > 1:
            for spk in sorted(by_speaker):
                print_confusion_matrix(by_speaker[spk], f"LOSO 화자: {spk}")


if __name__ == '__main__':
    main()
