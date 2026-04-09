"""MFCC+코사인 + 오/우 스펙트럴 보조 판별 실험.

조건:
  A: baseline (MFCC 13 + 코사인, 89.3%)
  B: A + spectral tilt 보조 (오/우 예측 시)
  C: A + spectral tilt + 고주파 에너지 비율 보조
  D: A + 스펙트럼 기반 F3 추정 + spectral tilt 보조
  E: A + 으/오/우 삼각 보조 (spectral centroid)
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
NUM_MEL_BANDS = 26
NUM_COEFFS = 13
MIN_FREQ = 80
MAX_FREQ = 4000

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


def build_mel_filterbank(sr, fft_size):
    num_bins = fft_size // 2 + 1
    bfw = sr / fft_size
    min_mel = 2595 * np.log10(1 + MIN_FREQ / 700)
    max_mel = 2595 * np.log10(1 + MAX_FREQ / 700)
    mel_pts = []
    for i in range(NUM_MEL_BANDS + 2):
        mel = min_mel + (max_mel - min_mel) * i / (NUM_MEL_BANDS + 1)
        mel_pts.append(700 * (10 ** (mel / 2595) - 1))
    bins = [int(np.floor(f / bfw)) for f in mel_pts]
    fb = np.zeros((NUM_MEL_BANDS, num_bins), dtype=np.float32)
    for m in range(NUM_MEL_BANDS):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        for k in range(l, min(c, num_bins)):
            if c != l:
                fb[m, k] = (k - l) / (c - l)
        for k in range(c, min(r + 1, num_bins)):
            if r != c:
                fb[m, k] = (r - k) / (r - c)
    return fb


def build_dct_matrix():
    matrix = np.zeros((NUM_COEFFS + 1, NUM_MEL_BANDS), dtype=np.float32)
    for k in range(NUM_COEFFS + 1):
        for n in range(NUM_MEL_BANDS):
            matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / NUM_MEL_BANDS)
    return matrix


def extract_mfcc(frame, sr):
    n_fft = min(FFT_SIZE, len(frame))
    key = (sr, n_fft)
    if key not in _fb_cache:
        _fb_cache[key] = build_mel_filterbank(sr, n_fft)
        _dct_cache[key] = build_dct_matrix()

    windowed = frame[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    fb = _fb_cache[key]
    mel_e = np.array([np.sum(spectrum[:fb.shape[1]] ** 2 * fb[m])
                       for m in range(NUM_MEL_BANDS)])
    log_mel = np.log(np.maximum(mel_e, 1e-10))
    dct = _dct_cache[key]
    mfcc = np.array([np.sum(log_mel * dct[k + 1]) for k in range(NUM_COEFFS)],
                     dtype=np.float32)
    return mfcc, spectrum


def cosine_distance(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 2.0
    return 1 - dot / (na * nb)


def classify_base(mfcc, prototypes):
    """기본 코사인 거리 분류."""
    dists = {}
    for v, proto in prototypes.items():
        dists[v] = cosine_distance(mfcc, proto)
    best = min(dists, key=dists.get)
    sorted_d = sorted(dists.values())
    conf = 0.0
    if sorted_d[1] > 0:
        conf = min(1.0, (sorted_d[1] - sorted_d[0]) / sorted_d[1])
    return best, conf, dists


def spectral_tilt(spectrum, sr, n_fft):
    """저주파(0~1kHz) vs 고주파(1~4kHz) 에너지 비율 (dB)."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    low = freqs < 1000
    high = (freqs >= 1000) & (freqs < 4000)
    low_e = np.sum(spectrum[low] ** 2) if np.any(low) else 1e-10
    high_e = np.sum(spectrum[high] ** 2) if np.any(high) else 1e-10
    return 10.0 * np.log10(low_e / max(high_e, 1e-10))


def high_freq_ratio(spectrum, sr, n_fft):
    """2~4kHz 대역 에너지 비율 (F3 영역)."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mid = (freqs >= 500) & (freqs < 2000)
    hi = (freqs >= 2000) & (freqs < 4000)
    mid_e = np.sum(spectrum[mid] ** 2) if np.any(mid) else 1e-10
    hi_e = np.sum(spectrum[hi] ** 2) if np.any(hi) else 1e-10
    return 10.0 * np.log10(hi_e / max(mid_e, 1e-10))


def estimate_f3(spectrum, sr, n_fft):
    """스펙트럼에서 F3 추정 (2000~3500Hz 대역 피크)."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mask = (freqs >= 2000) & (freqs <= 3500)
    if not np.any(mask):
        return 0.0
    region = spectrum.copy()
    region[~mask] = 0
    peak_bin = np.argmax(region)
    return float(freqs[peak_bin])


def spectral_centroid(spectrum, sr, n_fft, min_f=200, max_f=4000):
    """스펙트럴 센트로이드 (주파수 무게중심)."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mask = (freqs >= min_f) & (freqs <= max_f)
    power = spectrum[mask] ** 2
    total = np.sum(power)
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs[mask] * power) / total)


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


def build_prototypes(cal_files):
    cal_mfccs = {v: [] for v in VOWELS}
    for path, vowel, fname in cal_files:
        audio, sr = load_audio(path)
        for frame in slice_frames(audio):
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < MIN_RMS:
                continue
            mfcc, _ = extract_mfcc(frame, sr)
            cal_mfccs[vowel].append(mfcc)
    prototypes = {}
    for v in VOWELS:
        if len(cal_mfccs[v]) > 10:
            prototypes[v] = np.mean(cal_mfccs[v], axis=0).astype(np.float32)
    return prototypes, cal_mfccs


def evaluate_condition(name, all_data, classify_fn):
    """전체 화자 평가."""
    all_results = []
    by_speaker = {}

    for label, prototypes, test_files in all_data:
        results = []
        for path, gt, fname in test_files:
            audio, sr = load_audio(path)
            frame_preds = []
            for frame in slice_frames(audio):
                rms = np.sqrt(np.mean(frame ** 2))
                if rms < MIN_RMS:
                    continue
                mfcc, spectrum = extract_mfcc(frame, sr)
                pred = classify_fn(mfcc, spectrum, sr, prototypes)
                frame_preds.append(pred)

            if frame_preds:
                vote = Counter(frame_preds)
                file_pred = vote.most_common(1)[0][0]
            else:
                file_pred = '?'
            results.append({'gt': gt, 'pred': file_pred, 'fname': fname,
                            'speaker': label})
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
        eu = [r for r in rs if r['gt'] == '으']
        eu_c = sum(1 for r in eu if r['gt'] == r['pred'])
        print(f'    {label:<16s}: {sc:2d}/{len(rs):2d} ({sc/len(rs)*100:5.1f}%)  '
              f'오/우={ou_c}/{len(ou)}  으={eu_c}/{len(eu)}')

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
    print('=' * 70)
    print('  MFCC+코사인 + 오/우 스펙트럴 보조 판별 실험')
    print('=' * 70)

    # 데이터 로드
    all_data = []
    for dirname, speaker_id, label in SPEAKER_DIRS:
        cal_files, test_files = collect_speaker_files(dirname)
        if not cal_files or not test_files:
            continue
        prototypes, cal_mfccs = build_prototypes(cal_files)
        all_data.append((label, prototypes, test_files))
        n_cal = sum(len(v) for v in cal_mfccs.values())
        print(f'  {label}: cal={n_cal} vectors, test={len(test_files)} files')

    summary = []
    n_fft = FFT_SIZE

    # ── A: baseline ──
    def clf_a(mfcc, spectrum, sr, protos):
        best, conf, dists = classify_base(mfcc, protos)
        return best

    c, t, ar, bs = evaluate_condition('A', all_data, clf_a)
    acc = print_result('A: baseline (MFCC+코사인)', c, t, ar, bs)
    summary.append(('A: baseline', c, t, acc))

    # ── B: spectral tilt 보조 (오/우 예측 시) ──
    def clf_b(mfcc, spectrum, sr, protos):
        best, conf, dists = classify_base(mfcc, protos)
        if best in ('오', '우'):
            # 오/우 거리 차이가 작을 때만 개입
            margin = abs(dists['오'] - dists['우'])
            total_ou = dists['오'] + dists['우']
            if total_ou > 0 and margin / total_ou < 0.15:
                tilt = spectral_tilt(spectrum, sr, n_fft)
                # 오: tilt 높음 (저주파 우세), 우: tilt 낮음
                if tilt > 12.0:
                    return '오'
                elif tilt < 8.0:
                    return '우'
        return best

    c, t, ar, bs = evaluate_condition('B', all_data, clf_b)
    acc = print_result('B: + spectral tilt (오/우)', c, t, ar, bs)
    summary.append(('B: +spectral tilt', c, t, acc))

    # ── C: spectral tilt + 고주파 비율 ──
    def clf_c(mfcc, spectrum, sr, protos):
        best, conf, dists = classify_base(mfcc, protos)
        if best in ('오', '우'):
            margin = abs(dists['오'] - dists['우'])
            total_ou = dists['오'] + dists['우']
            if total_ou > 0 and margin / total_ou < 0.15:
                tilt = spectral_tilt(spectrum, sr, n_fft)
                hfr = high_freq_ratio(spectrum, sr, n_fft)
                # 우: 고주파 비율 낮음 (어두운 소리)
                # 오: 고주파 비율 상대적으로 높음
                score = tilt * 0.05 - hfr * 0.1  # 양수=오, 음수=우
                if score > 0.3:
                    return '오'
                elif score < -0.3:
                    return '우'
        return best

    c, t, ar, bs = evaluate_condition('C', all_data, clf_c)
    acc = print_result('C: + spectral tilt + HF ratio (오/우)', c, t, ar, bs)
    summary.append(('C: +tilt+HF', c, t, acc))

    # ── D: F3 추정 + spectral tilt ──
    def clf_d(mfcc, spectrum, sr, protos):
        best, conf, dists = classify_base(mfcc, protos)
        if best in ('오', '우'):
            margin = abs(dists['오'] - dists['우'])
            total_ou = dists['오'] + dists['우']
            if total_ou > 0 and margin / total_ou < 0.15:
                f3 = estimate_f3(spectrum, sr, n_fft)
                tilt = spectral_tilt(spectrum, sr, n_fft)
                # F3: 우 < 2600Hz, 오 > 2600Hz (일반적 경향)
                f3_score = (f3 - 2600.0) / 300.0
                f3_score = max(-1.0, min(1.0, f3_score))
                tilt_score = (tilt - 10.0) / 5.0
                tilt_score = max(-1.0, min(1.0, tilt_score))
                score = f3_score * 0.6 + tilt_score * 0.4
                if score > 0.2:
                    return '오'
                elif score < -0.2:
                    return '우'
        return best

    c, t, ar, bs = evaluate_condition('D', all_data, clf_d)
    acc = print_result('D: + F3 추정 + spectral tilt (오/우)', c, t, ar, bs)
    summary.append(('D: +F3+tilt', c, t, acc))

    # ── E: 으/오/우 삼각 보조 (spectral centroid) ──
    def clf_e(mfcc, spectrum, sr, protos):
        best, conf, dists = classify_base(mfcc, protos)
        if best in ('오', '우', '으'):
            # 삼각 판별: 으/오/우 거리가 모두 비슷할 때
            d_eu = dists.get('으', 999)
            d_oh = dists.get('오', 999)
            d_oo = dists.get('우', 999)
            top3 = sorted([(d_eu, '으'), (d_oh, '오'), (d_oo, '우')])

            margin = top3[1][0] - top3[0][0]
            if margin < 0.02:  # 1등과 2등 차이가 작을 때
                cent = spectral_centroid(spectrum, sr, n_fft)
                f3 = estimate_f3(spectrum, sr, n_fft)
                tilt = spectral_tilt(spectrum, sr, n_fft)

                # 으: centroid 높음 (F2 높음), 오: 중간, 우: 낮음
                if cent > 1200:
                    return '으'
                elif tilt > 12.0 and f3 > 2700:
                    return '오'
                elif tilt < 9.0 or f3 < 2500:
                    return '우'
        return best

    c, t, ar, bs = evaluate_condition('E', all_data, clf_e)
    acc = print_result('E: + 으/오/우 삼각 보조 (centroid+F3+tilt)', c, t, ar, bs)
    summary.append(('E: +삼각보조', c, t, acc))

    # ── 최종 비교 ──
    print(f'\n\n{"═"*70}')
    print(f'  최종 비교')
    print(f'{"═"*70}')
    print(f'\n  {"조건":<25s} {"전체":>12s}  {"오":>4s}  {"우":>4s}  {"으":>4s}')
    print(f'  {"─"*25} {"─"*12}  {"─"*4}  {"─"*4}  {"─"*4}')

    for name, c, t, acc in summary:
        # 각 조건의 all_results를 다시 계산하지 않고, 저장된 것 사용
        pass

    # 다시 돌리는 대신 summary에 all_results도 저장
    print('\n  (위 상세 결과 참조)')


if __name__ == '__main__':
    main()
