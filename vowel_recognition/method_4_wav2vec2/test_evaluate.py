"""Method 4 (Korean wav2vec2) 정량 평가 스크립트.

사용법:
  python -m vowel_recognition.method_4_wav2vec2.test_evaluate eval --audio_dir vowel_recognition/dataset
  python -m vowel_recognition.method_4_wav2vec2.test_evaluate eval --audio_dir vowel_recognition/dataset --speaker 김동규
  python -m vowel_recognition.method_4_wav2vec2.test_evaluate eval --audio_dir vowel_recognition/dataset --detail
"""

import sys
import os
import argparse
import time
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]
SR = 44100

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


def load_model():
    print("모델 로딩 중...", flush=True)
    from vowel_recognition.method_4_wav2vec2.features import Wav2Vec2KoreanCTC
    detector = Wav2Vec2KoreanCTC()
    print("모델 로딩 완료.\n")
    return detector


def predict_vowel(detector, audio, sr):
    probs = detector.get_vowel_probs(audio, sr)
    if not probs:
        return None, 0.0, {}
    best = max(probs, key=probs.get)
    total = sum(probs.values())
    conf = probs[best] / total if total > 0 else 0.0
    return best, conf, probs


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


def cmd_eval(args):
    audio_dir = args.audio_dir
    if not os.path.isdir(audio_dir):
        print(f"오류: 디렉토리를 찾을 수 없습니다: {audio_dir}")
        sys.exit(1)

    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    all_files = sorted([f for f in os.listdir(audio_dir)
                        if os.path.splitext(f)[1].lower() in audio_exts])
    if not all_files:
        print(f"오류: {audio_dir}/ 에 오디오 파일이 없습니다.")
        sys.exit(1)

    samples = []
    skipped = []
    for f in all_files:
        vowel = parse_vowel_from_filename(f)
        if vowel is None:
            skipped.append(f)
            continue
        meta = parse_metadata(f)
        if args.speaker and meta.get('speaker') != args.speaker:
            continue
        if args.gender and meta.get('gender') != args.gender:
            continue
        if args.condition and meta.get('condition') != args.condition:
            continue
        samples.append((f, vowel, meta))

    if skipped:
        print(f"경고: 모음 파싱 불가 파일 {len(skipped)}개 건너뜀")
        print()

    if not samples:
        print("평가할 샘플이 없습니다.")
        sys.exit(1)

    speakers = set(m.get('speaker', '?') for _, _, m in samples)
    conditions = set(m.get('condition', '?') for _, _, m in samples)
    vowel_counts = {}
    for _, v, _ in samples:
        vowel_counts[v] = vowel_counts.get(v, 0) + 1

    print(f"평가 대상: {len(samples)}개 파일")
    print(f"  화자: {', '.join(sorted(speakers))}")
    print(f"  조건: {', '.join(sorted(conditions))}")
    print(f"  모음별: {', '.join(f'{v}({c})' for v, c in sorted(vowel_counts.items()))}")
    print()

    detector = load_model()
    results = []
    results_by_speaker = {}
    results_by_condition = {}

    for i, (filename, gt_vowel, meta) in enumerate(samples):
        filepath = os.path.join(audio_dir, filename)
        audio, sr = load_audio(filepath)

        t0 = time.perf_counter()
        pred, conf, probs = predict_vowel(detector, audio, sr)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ok = "O" if pred == gt_vowel else "X"
        print(f"  [{i+1:3d}/{len(samples)}] {filename:45s}  "
              f"정답={gt_vowel}  예측={pred}  신뢰도={conf:.3f}  "
              f"{elapsed_ms:.0f}ms  {ok}")

        if pred is not None:
            entry = (gt_vowel, pred, conf)
            results.append(entry)
            spk = meta.get('speaker', '?')
            results_by_speaker.setdefault(spk, []).append(entry)
            cond = meta.get('condition', '?')
            results_by_condition.setdefault(cond, []).append(entry)

    print_confusion_matrix(results, "전체 결과")

    if len(results_by_speaker) > 1:
        print(f"\n{'─'*60}")
        print("화자별 정확도:")
        for spk in sorted(results_by_speaker):
            r = results_by_speaker[spk]
            acc = sum(1 for gt, pred, _ in r if gt == pred) / len(r) * 100
            print(f"  {spk:12s}: {acc:5.1f}% ({len(r)}개)")

    if len(results_by_condition) > 1:
        print(f"\n{'─'*60}")
        print("조건별 정확도:")
        for cond in sorted(results_by_condition):
            r = results_by_condition[cond]
            acc = sum(1 for gt, pred, _ in r if gt == pred) / len(r) * 100
            print(f"  {cond:12s}: {acc:5.1f}% ({len(r)}개)")

    if args.detail and len(results_by_speaker) > 1:
        for spk in sorted(results_by_speaker):
            print_confusion_matrix(results_by_speaker[spk], f"화자: {spk}")


def main():
    parser = argparse.ArgumentParser(description="Method 4 (Korean wav2vec2) 정량 평가")
    sub = parser.add_subparsers(dest='cmd')

    p_eval = sub.add_parser('eval', help='오디오 파일로 평가')
    p_eval.add_argument('--audio_dir', required=True)
    p_eval.add_argument('--speaker', default=None)
    p_eval.add_argument('--gender', default=None)
    p_eval.add_argument('--condition', default=None)
    p_eval.add_argument('--detail', action='store_true')

    args = parser.parse_args()
    if args.cmd == 'eval':
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
