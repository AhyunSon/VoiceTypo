"""포먼트 분류기 오프라인 테스트 v3.

하모닉 봉투 + LPC 폴백 + 오/우 보조 피처 + Lobanov 정규화 테스트.
remote 폴더(5화자 × 70개) + live_recordings 데이터 사용.
"""

import os, sys
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(__file__))

from vowel_recognition.formant_classifier import (
    FormantVowelClassifier, FormantCalibrator, DEFAULT_PROTOTYPES,
)
from pitch_detection.yin import YinDetector


# ── 오디오 로딩 ──

def load_wav(path):
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        pass
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        samples /= 32768.0
        return samples, sr
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return None, None


# ── 데이터 수집 ──

def collect_remote_files():
    base = os.path.join(os.path.dirname(__file__),
                        'vowel_recognition', 'method_6_embedding')
    files = []
    for entry in os.listdir(base):
        if not entry.startswith('vowel-remote-001_'):
            continue
        folder = os.path.join(base, entry)
        if not os.path.isdir(folder):
            continue
        speaker = entry.replace('vowel-remote-001_', '').strip()
        speaker = speaker.rstrip(')').rstrip(' (1').rstrip('(').strip()
        for fname in os.listdir(folder):
            if not fname.endswith('.wav'):
                continue
            parts = fname.replace('.wav', '').split('_')
            if len(parts) >= 3:
                vowel = parts[2]
                if vowel in DEFAULT_PROTOTYPES:
                    files.append((os.path.join(folder, fname), vowel, speaker))
    return files


def collect_live_files():
    base = os.path.join(os.path.dirname(__file__),
                        'vowel_recognition', 'method_6_embedding', 'live_recordings')
    files = []
    if not os.path.isdir(base):
        return files
    for entry in os.listdir(base):
        folder = os.path.join(base, entry)
        if not os.path.isdir(folder):
            continue
        speaker = entry
        for fname in os.listdir(folder):
            if not fname.endswith('.wav'):
                continue
            parts = fname.replace('.wav', '').split('_')
            if len(parts) >= 1:
                vowel = parts[0]
                if vowel in DEFAULT_PROTOTYPES:
                    files.append((os.path.join(folder, fname), vowel, speaker))
    return files


# ── 분류 ──

def classify_file(clf, path, target_sr=44100):
    """파일 하나를 프레임 단위로 분류 (YIN + FFT 전달), 다수결 반환."""
    audio, sr = load_wav(path)
    if audio is None:
        return None, 0.0, 0.0, 0.0, 'none'

    if sr != target_sr:
        ratio = target_sr / sr
        n_out = int(len(audio) * ratio)
        indices = np.arange(n_out) / ratio
        idx = np.clip(indices.astype(int), 0, len(audio) - 1)
        audio = audio[idx]

    frame_size = 2048
    hop = 1024
    yin = YinDetector(target_sr)
    window = np.hanning(frame_size).astype(np.float32)

    votes = defaultdict(float)
    f1_list, f2_list = [], []
    method_counts = defaultdict(int)

    clf.reset()
    for start in range(0, len(audio) - frame_size, hop):
        chunk = audio[start:start + frame_size]
        freq, rms = yin.detect(chunk)

        fft_mag = np.abs(np.fft.rfft(chunk * window))
        fft_freqs = np.fft.rfftfreq(frame_size, d=1.0 / target_sr)

        vowel, conf, f1, f2 = clf.classify(
            chunk, f0=freq, fft_mag=fft_mag, fft_freqs=fft_freqs)

        if vowel is not None and conf > 0.3:
            votes[vowel] += conf
            f1_list.append(f1)
            f2_list.append(f2)

    if not votes:
        return None, 0.0, 0.0, 0.0, 'none'

    best = max(votes, key=votes.get)
    total_conf = sum(votes.values())
    best_conf = votes[best] / total_conf if total_conf > 0 else 0
    f1_med = float(np.median(f1_list)) if f1_list else 0
    f2_med = float(np.median(f2_list)) if f2_list else 0
    return best, best_conf, f1_med, f2_med, 'ok'


def run_test(files, clf, label=""):
    if not files:
        print(f"  [{label}] 파일 없음")
        return {}

    correct = 0
    total = 0
    per_vowel = defaultdict(lambda: {'correct': 0, 'total': 0, 'preds': []})
    confusion = defaultdict(lambda: defaultdict(int))
    f1f2_by_vowel = defaultdict(list)

    for path, gt_vowel, speaker in files:
        pred, conf, f1, f2, status = classify_file(clf, path)
        if pred is None:
            continue
        total += 1
        per_vowel[gt_vowel]['total'] += 1
        per_vowel[gt_vowel]['preds'].append(pred)
        confusion[gt_vowel][pred] += 1
        if f1 > 0 and f2 > 0:
            f1f2_by_vowel[gt_vowel].append((f1, f2))
        if pred == gt_vowel:
            correct += 1
            per_vowel[gt_vowel]['correct'] += 1

    if total == 0:
        print(f"  [{label}] 유효 파일 없음")
        return {}

    acc = correct / total * 100
    print(f"\n  [{label}] 전체: {correct}/{total} ({acc:.1f}%)")

    for v in ['아', '어', '오', '우', '으', '이', '에']:
        d = per_vowel.get(v)
        if not d or d['total'] == 0:
            continue
        vacc = d['correct'] / d['total'] * 100
        wrong = [p for p in d['preds'] if p != v]
        wrong_str = ""
        if wrong:
            wc = Counter(wrong).most_common(3)
            wrong_str = " <- " + ", ".join(f"{w}({n})" for w, n in wc)
        print(f"    {v}: {d['correct']}/{d['total']} ({vacc:.0f}%){wrong_str}")

    print(f"\n  [{label}] F1/F2 중간값:")
    for v in ['아', '어', '오', '우', '으', '이', '에']:
        pts = f1f2_by_vowel.get(v, [])
        if not pts:
            continue
        f1s = [p[0] for p in pts]
        f2s = [p[1] for p in pts]
        proto = DEFAULT_PROTOTYPES.get(v, (0, 0))
        print(f"    {v}: F1={np.median(f1s):.0f}+/-{np.std(f1s):.0f}  "
              f"F2={np.median(f2s):.0f}+/-{np.std(f2s):.0f}  "
              f"(proto: {proto[0]},{proto[1]})")

    if confusion['오']['우'] + confusion['우']['오'] > 0:
        print(f"\n  [{label}] 오<->우 혼동:")
        print(f"    오->우: {confusion['오']['우']}/{per_vowel['오']['total']}")
        print(f"    우->오: {confusion['우']['오']}/{per_vowel['우']['total']}")

    return f1f2_by_vowel


def run_calibrated_test(files_by_speaker, label=""):
    total_correct = 0
    total_count = 0
    per_vowel_all = defaultdict(lambda: {'correct': 0, 'total': 0})
    confusion_all = defaultdict(lambda: defaultdict(int))

    for speaker, speaker_files in sorted(files_by_speaker.items()):
        clf = FormantVowelClassifier(sample_rate=44100)
        cal = FormantCalibrator(clf)

        vowel_counts = defaultdict(int)
        cal_limit = 3
        cal_files = []
        test_files = []

        for path, vowel, spk in speaker_files:
            vowel_counts[vowel] += 1
            if vowel_counts[vowel] <= cal_limit:
                cal_files.append((path, vowel, spk))
            else:
                test_files.append((path, vowel, spk))

        # 캘리브레이션 F1/F2 수집
        yin = YinDetector(44100)
        window = np.hanning(2048).astype(np.float32)
        for path, vowel, spk in cal_files:
            audio, sr = load_wav(path)
            if audio is None:
                continue
            if sr != 44100:
                ratio = 44100 / sr
                n_out = int(len(audio) * ratio)
                indices = np.arange(n_out) / ratio
                idx_arr = np.clip(indices.astype(int), 0, len(audio) - 1)
                audio = audio[idx_arr]

            clf_temp = FormantVowelClassifier(sample_rate=44100)
            for start in range(0, len(audio) - 2048, 1024):
                chunk = audio[start:start + 2048]
                freq, rms = yin.detect(chunk)
                fft_mag = np.abs(np.fft.rfft(chunk * window))
                fft_freqs = np.fft.rfftfreq(2048, d=1.0 / 44100)
                _, _, f1, f2 = clf_temp.classify(
                    chunk, f0=freq, fft_mag=fft_mag, fft_freqs=fft_freqs)
                if f1 > 0 and f2 > 0:
                    cal.record_frame(vowel, f1, f2)

        cal.apply()

        if not test_files:
            test_files = speaker_files

        correct = 0
        total = 0
        for path, gt, spk in test_files:
            pred, conf, f1, f2, _ = classify_file(clf, path)
            if pred is None:
                continue
            total += 1
            per_vowel_all[gt]['total'] += 1
            confusion_all[gt][pred] += 1
            if pred == gt:
                correct += 1
                per_vowel_all[gt]['correct'] += 1

        total_correct += correct
        total_count += total
        acc = correct / total * 100 if total > 0 else 0
        cal_str = "cal+lobanov" if clf._normalizer.is_fitted else "cal"
        print(f"    {speaker}: {correct}/{total} ({acc:.0f}%) [{cal_str}]")

    if total_count > 0:
        acc = total_correct / total_count * 100
        print(f"\n  [{label}] 캘리브 전체: {total_correct}/{total_count} ({acc:.1f}%)")
        for v in ['아', '어', '오', '우', '으', '이', '에']:
            d = per_vowel_all.get(v)
            if not d or d['total'] == 0:
                continue
            vacc = d['correct'] / d['total'] * 100
            print(f"    {v}: {d['correct']}/{d['total']} ({vacc:.0f}%)")
        if confusion_all['오']['우'] + confusion_all['우']['오'] > 0:
            print(f"    오->우: {confusion_all['오']['우']}/{per_vowel_all['오']['total']}")
            print(f"    우->오: {confusion_all['우']['오']}/{per_vowel_all['우']['total']}")


def main():
    print("=" * 60)
    print("  포먼트 분류기 v3 오프라인 테스트")
    print("  (하모닉 봉투 + LPC 폴백 + 오/우 보조 + Lobanov)")
    print("=" * 60)

    remote_files = collect_remote_files()
    live_files = collect_live_files()

    print(f"\nRemote: {len(remote_files)} files")
    speakers_r = defaultdict(list)
    for f in remote_files:
        speakers_r[f[2]].append(f)
    for spk, files in sorted(speakers_r.items()):
        vowels = defaultdict(int)
        for _, v, _ in files:
            vowels[v] += 1
        vstr = " ".join(f"{v}:{n}" for v, n in sorted(vowels.items()))
        print(f"  {spk}: {len(files)} ({vstr})")

    print(f"\nLive: {len(live_files)} files")
    speakers_l = defaultdict(list)
    for f in live_files:
        speakers_l[f[2]].append(f)
    for spk, files in sorted(speakers_l.items()):
        vowels = defaultdict(int)
        for _, v, _ in files:
            vowels[v] += 1
        vstr = " ".join(f"{v}:{n}" for v, n in sorted(vowels.items()))
        print(f"  {spk}: {len(files)} ({vstr})")

    all_files = remote_files + live_files

    # ── 테스트 1: 기본 프로토타입 ──
    print("\n" + "=" * 60)
    print("  테스트 1: 기본 프로토타입 (하모닉 봉투, 캘리브 없음)")
    print("=" * 60)

    clf = FormantVowelClassifier(sample_rate=44100)
    run_test(all_files, clf, "전체")

    # ── 테스트 2: 화자별 캘리브레이션 + Lobanov ──
    print("\n" + "=" * 60)
    print("  테스트 2: 화자별 캘리브레이션 + Lobanov")
    print("=" * 60)

    all_by_speaker = defaultdict(list)
    for f in all_files:
        all_by_speaker[f[2]].append(f)
    run_calibrated_test(all_by_speaker, "캘리브+Lobanov")

    # ── 테스트 3: 화자별 오/우 F1/F2 분포 ──
    print("\n" + "=" * 60)
    print("  테스트 3: 화자별 오/우 F1/F2 (하모닉 봉투)")
    print("=" * 60)

    for speaker, sfiles in sorted(all_by_speaker.items()):
        ou_files = [(p, v, s) for p, v, s in sfiles if v in ('오', '우')]
        if not ou_files:
            continue
        clf = FormantVowelClassifier(sample_rate=44100)
        print(f"\n  {speaker}:")
        for target_v in ['오', '우']:
            vfiles = [(p, v, s) for p, v, s in ou_files if v == target_v]
            f1s, f2s = [], []
            for path, _, _ in vfiles:
                _, _, f1, f2, _ = classify_file(clf, path)
                if f1 > 0 and f2 > 0:
                    f1s.append(f1)
                    f2s.append(f2)
            if f1s:
                print(f"    {target_v}: F1={np.median(f1s):.0f}+/-{np.std(f1s):.0f}  "
                      f"F2={np.median(f2s):.0f}+/-{np.std(f2s):.0f}  (n={len(f1s)})")


if __name__ == '__main__':
    main()
