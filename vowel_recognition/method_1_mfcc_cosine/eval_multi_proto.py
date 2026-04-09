"""프로토타입 다중화 실험.

조건:
  A: 평균 1개 (baseline, 89.3%)
  B: k-means k=3
  C: k-means k=5
  D: k-means k=7
  E: k-means k=10
  F: 전체 프레임 보관 (k-NN, k=5 투표)
  G: 전체 프레임 보관 (k-NN, k=11 투표)
"""
import sys, os, io, wave
import numpy as np
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

M6_BASE = os.path.join(os.path.dirname(__file__), '..', 'method_6_embedding')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']
FFT_SIZE = 2048
HOP_SIZE = 1024
MIN_RMS = 0.005

SPEAKER_DIRS = [
    ('vowel-remote-001_hj', 'hj', 'HJ (여 60대)'),
    ('vowel-remote-001_mt', 'mt', 'MT (남 60대)'),
    ('vowel-remote-001_kdg0534 (1)', 'kdg0534', 'KDG (남)'),
    ('vowel-remote-001_lynn03 (1)', 'lynn03', 'Lynn (여)'),
    ('vowel-remote-001_아현 (1)', '아현', '아현 (여 20대)'),
]


def load_audio(path):
    with wave.open(path, 'r') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        ch = wf.getnchannels()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return a, sr


def slice_frames(audio):
    frames = []
    for start in range(0, len(audio) - FFT_SIZE + 1, HOP_SIZE):
        frames.append(audio[start:start + FFT_SIZE])
    return frames


_fb_cache = {}
_dct_cache = {}


def extract_mfcc(audio_frame, sr):
    n_fft = min(FFT_SIZE, len(audio_frame))

    key = (sr, n_fft)
    if key not in _fb_cache:
        num_bins = n_fft // 2 + 1
        bfw = sr / n_fft
        min_mel = 2595 * np.log10(1 + 80 / 700)
        max_mel = 2595 * np.log10(1 + 4000 / 700)
        mel_pts = [700 * (10 ** ((min_mel + (max_mel - min_mel) * i / 27) / 2595) - 1) for i in range(28)]
        bins = [int(np.floor(f / bfw)) for f in mel_pts]
        fb = np.zeros((26, num_bins), dtype=np.float32)
        for m in range(26):
            l, c, r = bins[m], bins[m+1], bins[m+2]
            for k in range(l, min(c, num_bins)):
                if c != l: fb[m, k] = (k - l) / (c - l)
            for k in range(c, min(r+1, num_bins)):
                if r != c: fb[m, k] = (r - k) / (r - c)
        _fb_cache[key] = fb

        dct = np.zeros((14, 26), dtype=np.float32)
        for ki in range(14):
            for n in range(26):
                dct[ki, n] = np.cos(np.pi * ki * (n + 0.5) / 26)
        _dct_cache[key] = dct

    windowed = audio_frame[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    fb = _fb_cache[key]
    mel_e = np.array([np.sum(spectrum[:fb.shape[1]] ** 2 * fb[m]) for m in range(26)])
    log_mel = np.log(np.maximum(mel_e, 1e-10))
    dct = _dct_cache[key]
    mfcc = np.array([np.sum(log_mel * dct[k+1]) for k in range(13)], dtype=np.float32)
    return mfcc


def cosine_distance(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 2.0
    return 1 - dot / (na * nb)


def kmeans_cosine(vectors, k, max_iter=50):
    """코사인 거리 기반 k-means."""
    n = len(vectors)
    if n <= k:
        return vectors

    # 초기화: k-means++ 방식
    indices = [np.random.randint(n)]
    for _ in range(k - 1):
        dists = np.array([min(cosine_distance(vectors[i], vectors[j]) for j in indices) for i in range(n)])
        dists = np.maximum(dists, 0)  # 음수 방지
        total = dists.sum()
        if total < 1e-10:
            probs = np.ones(n) / n
        else:
            probs = dists / total
        indices.append(np.random.choice(n, p=probs))
    centers = [vectors[i].copy() for i in indices]

    for _ in range(max_iter):
        # 할당
        assignments = []
        for v in vectors:
            dists = [cosine_distance(v, c) for c in centers]
            assignments.append(np.argmin(dists))

        # 갱신
        new_centers = []
        for ci in range(k):
            members = [vectors[i] for i in range(n) if assignments[i] == ci]
            if members:
                new_centers.append(np.mean(members, axis=0).astype(np.float32))
            else:
                new_centers.append(centers[ci])

        if all(np.allclose(c, nc, atol=1e-6) for c, nc in zip(centers, new_centers)):
            break
        centers = new_centers

    return centers


def collect_speaker_files(dirname):
    d = os.path.join(M6_BASE, dirname)
    if not os.path.isdir(d):
        return [], []
    cal, test = [], []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        parts = os.path.splitext(f)[0].split('_')
        if len(parts) < 4:
            continue
        vowel, syllable = parts[2], parts[3]
        if vowel not in VOWELS:
            continue
        path = os.path.join(d, f)
        if vowel == syllable:
            cal.append((path, vowel, f))
        else:
            test.append((path, vowel, f))
    return cal, test


def extract_cal_features(cal_files):
    """캘리브레이션 파일에서 모음별 MFCC 벡터 수집."""
    cal_mfccs = {v: [] for v in VOWELS}
    for path, vowel, fname in cal_files:
        audio, sr = load_audio(path)
        for frame in slice_frames(audio):
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < MIN_RMS:
                continue
            mfcc = extract_mfcc(frame, sr)
            cal_mfccs[vowel].append(mfcc)
    return cal_mfccs


def classify_single_proto(feat, prototypes):
    """단일 프로토타입 (평균)."""
    best, min_d = '아', float('inf')
    for v, proto in prototypes.items():
        d = cosine_distance(feat, proto)
        if d < min_d:
            min_d = d
            best = v
    return best


def classify_multi_proto(feat, multi_prototypes):
    """다중 프로토타입: 가장 가까운 프로토타입의 모음."""
    best, min_d = '아', float('inf')
    for v, protos in multi_prototypes.items():
        for proto in protos:
            d = cosine_distance(feat, proto)
            if d < min_d:
                min_d = d
                best = v
    return best


def classify_knn(feat, all_vectors, k=5):
    """k-NN: 전체 캘리브레이션 벡터에서 k개 최근접 투표."""
    dists = []
    for v, vecs in all_vectors.items():
        for vec in vecs:
            dists.append((cosine_distance(feat, vec), v))
    dists.sort(key=lambda x: x[0])
    top_k = [d[1] for d in dists[:k]]
    vote = Counter(top_k)
    return vote.most_common(1)[0][0]


def evaluate(test_files, classify_fn):
    """테스트 파일 평가."""
    results = []
    for path, gt, fname in test_files:
        audio, sr = load_audio(path)
        frame_preds = []
        for frame in slice_frames(audio):
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < MIN_RMS:
                continue
            mfcc = extract_mfcc(frame, sr)
            pred = classify_fn(mfcc)
            frame_preds.append(pred)

        if frame_preds:
            vote = Counter(frame_preds)
            file_pred = vote.most_common(1)[0][0]
        else:
            file_pred = '?'
        results.append({'gt': gt, 'pred': file_pred, 'fname': fname})
    return results


def run_condition(name, all_speakers_data, build_classifier_fn):
    """하나의 조건으로 전체 화자 평가."""
    all_results = []
    by_speaker = {}

    for label, cal_mfccs, test_files in all_speakers_data:
        classify_fn = build_classifier_fn(cal_mfccs)
        results = evaluate(test_files, classify_fn)
        for r in results:
            r['speaker'] = label
        all_results.extend(results)
        by_speaker[label] = results

    correct = sum(1 for r in all_results if r['gt'] == r['pred'])
    total = len(all_results)
    return correct, total, all_results, by_speaker


def print_result(name, correct, total, all_results, by_speaker):
    acc = correct / total * 100
    print(f'\n  [{name}] 전체: {correct}/{total} ({acc:.1f}%)')
    for label in sorted(by_speaker.keys()):
        rs = by_speaker[label]
        sc = sum(1 for r in rs if r['gt'] == r['pred'])
        ou = [r for r in rs if r['gt'] in ['오', '우']]
        ou_c = sum(1 for r in ou if r['gt'] == r['pred'])
        print(f'    {label:<16s}: {sc:2d}/{len(rs):2d} ({sc/len(rs)*100:5.1f}%)  오/우={ou_c}/{len(ou)}')

    print(f'    모음별:')
    for v in VOWELS:
        vr = [r for r in all_results if r['gt'] == v]
        if not vr:
            continue
        vc = sum(1 for r in vr if r['gt'] == r['pred'])
        wrong = [r['pred'] for r in vr if r['gt'] != r['pred']]
        ws = ''
        if wrong:
            wc = Counter(wrong)
            ws = '  ' + ', '.join(f'→{k}({n})' for k, n in wc.most_common())
        print(f'      {v}: {vc:2d}/{len(vr):2d} ({vc/len(vr)*100:5.1f}%){ws}')

    return acc


def main():
    np.random.seed(42)

    print('=' * 70)
    print('  프로토타입 다중화 실험')
    print('=' * 70)

    # 데이터 로드
    all_speakers_data = []
    for dirname, speaker_id, label in SPEAKER_DIRS:
        cal_files, test_files = collect_speaker_files(dirname)
        if not cal_files or not test_files:
            continue
        cal_mfccs = extract_cal_features(cal_files)
        all_speakers_data.append((label, cal_mfccs, test_files))
        n_cal = sum(len(v) for v in cal_mfccs.values())
        print(f'  {label}: cal={n_cal} vectors, test={len(test_files)} files')

    summary = []

    # A: baseline (평균 1개)
    def build_single(cal_mfccs):
        protos = {v: np.mean(vecs, axis=0).astype(np.float32)
                  for v, vecs in cal_mfccs.items() if len(vecs) > 10}
        return lambda feat: classify_single_proto(feat, protos)

    c, t, ar, bs = run_condition('A', all_speakers_data, build_single)
    acc = print_result('A: 평균 1개 (baseline)', c, t, ar, bs)
    summary.append(('A: 평균 1개 (baseline)', c, t, acc, ar))

    # B~E: k-means
    for k_val in [3, 5, 7, 10]:
        label = f'k-means k={k_val}'

        def build_kmeans(cal_mfccs, k=k_val):
            multi = {}
            for v, vecs in cal_mfccs.items():
                if len(vecs) < k:
                    multi[v] = vecs if vecs else [np.zeros(13, dtype=np.float32)]
                else:
                    multi[v] = kmeans_cosine(vecs, k)
            return lambda feat, mp=multi: classify_multi_proto(feat, mp)

        c, t, ar, bs = run_condition(label, all_speakers_data, build_kmeans)
        acc = print_result(label, c, t, ar, bs)
        summary.append((label, c, t, acc, ar))

    # F: k-NN k=5
    def build_knn5(cal_mfccs):
        return lambda feat: classify_knn(feat, cal_mfccs, k=5)

    c, t, ar, bs = run_condition('k-NN k=5', all_speakers_data, build_knn5)
    acc = print_result('k-NN k=5', c, t, ar, bs)
    summary.append(('k-NN k=5', c, t, acc, ar))

    # G: k-NN k=11
    def build_knn11(cal_mfccs):
        return lambda feat: classify_knn(feat, cal_mfccs, k=11)

    c, t, ar, bs = run_condition('k-NN k=11', all_speakers_data, build_knn11)
    acc = print_result('k-NN k=11', c, t, ar, bs)
    summary.append(('k-NN k=11', c, t, acc, ar))

    # H: k-NN k=21
    def build_knn21(cal_mfccs):
        return lambda feat: classify_knn(feat, cal_mfccs, k=21)

    c, t, ar, bs = run_condition('k-NN k=21', all_speakers_data, build_knn21)
    acc = print_result('k-NN k=21', c, t, ar, bs)
    summary.append(('k-NN k=21', c, t, acc, ar))

    # ── 최종 비교 ──
    print(f'\n\n{"═"*70}')
    print(f'  최종 비교')
    print(f'{"═"*70}')
    print(f'\n  {"조건":<25s} {"전체":>12s}  {"으":>6s}  {"오/우":>6s}')
    print(f'  {"─"*25} {"─"*12}  {"─"*6}  {"─"*6}')

    for name, c, t, acc, ar in summary:
        eu = [r for r in ar if r['gt'] == '으']
        eu_c = sum(1 for r in eu if r['gt'] == r['pred'])
        oh = [r for r in ar if r['gt'] == '오']
        oo = [r for r in ar if r['gt'] == '우']
        ou_c = sum(1 for r in oh if r['gt'] == r['pred']) + sum(1 for r in oo if r['gt'] == r['pred'])
        ou_t = len(oh) + len(oo)
        print(f'  {name:<25s} {c:3d}/{t:3d} ({acc:5.1f}%)'
              f'  {eu_c:2d}/{len(eu):2d}'
              f'  {ou_c:2d}/{ou_t:2d}')


if __name__ == '__main__':
    main()
