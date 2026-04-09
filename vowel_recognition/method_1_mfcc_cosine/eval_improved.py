"""개선 실험: log-mel 직접 사용, mel band 수, 주파수 상한, 프레임 선택.

조건별 비교:
  A: 기존 (MFCC 13, mel 26, 80-4000Hz)            — baseline 89.3%
  B: log-mel 직접 (mel 26, 80-4000Hz)              — DCT 제거
  C: log-mel 40 band (80-4000Hz)                   — band 수 증가
  D: log-mel 40 band (80-5500Hz)                   — 주파수 확장
  E: D + 에너지 상위 50% 프레임만                     — 프레임 선택
  F: E + 으/우 판별 시 F2 대역 보조                   — 으 전용 보완
"""
import sys, os, io, wave
import numpy as np
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

M6_BASE = os.path.join(os.path.dirname(__file__), '..', 'method_6_embedding')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']

SPEAKER_DIRS = [
    ('vowel-remote-001_hj', 'hj', 'HJ (여 60대)'),
    ('vowel-remote-001_mt', 'mt', 'MT (남 60대)'),
    ('vowel-remote-001_kdg0534 (1)', 'kdg0534', 'KDG (남)'),
    ('vowel-remote-001_lynn03 (1)', 'lynn03', 'Lynn (여)'),
    ('vowel-remote-001_아현 (1)', '아현', '아현 (여 20대)'),
]

FFT_SIZE = 2048
HOP_SIZE = 1024
MIN_RMS = 0.005


def load_audio(path):
    with wave.open(path, 'r') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        ch = wf.getnchannels()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return a, sr


def slice_frames(audio, frame_size=FFT_SIZE, hop_size=HOP_SIZE):
    frames = []
    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frames.append(audio[start:start + frame_size])
    return frames


# ── 특징 추출 ──

_fb_cache = {}


def build_mel_filterbank(sr, fft_size, num_bands, min_freq, max_freq):
    key = (sr, fft_size, num_bands, min_freq, max_freq)
    if key in _fb_cache:
        return _fb_cache[key]

    num_bins = fft_size // 2 + 1
    bin_freq_width = sr / fft_size
    min_mel = 2595 * np.log10(1 + min_freq / 700)
    max_mel = 2595 * np.log10(1 + max_freq / 700)

    mel_pts = []
    for i in range(num_bands + 2):
        mel = min_mel + (max_mel - min_mel) * i / (num_bands + 1)
        mel_pts.append(700 * (10 ** (mel / 2595) - 1))

    bins = [int(np.floor(f / bin_freq_width)) for f in mel_pts]

    fb = np.zeros((num_bands, num_bins), dtype=np.float32)
    for m in range(num_bands):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, min(center, num_bins)):
            if center != left:
                fb[m, k] = (k - left) / (center - left)
        for k in range(center, min(right + 1, num_bins)):
            if right != center:
                fb[m, k] = (right - k) / (right - center)

    _fb_cache[key] = fb
    return fb


_dct_cache = {}


def build_dct_matrix(num_coeffs, num_bands):
    key = (num_coeffs, num_bands)
    if key in _dct_cache:
        return _dct_cache[key]
    matrix = np.zeros((num_coeffs + 1, num_bands), dtype=np.float32)
    for k in range(num_coeffs + 1):
        for n in range(num_bands):
            matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / num_bands)
    _dct_cache[key] = matrix
    return matrix


def extract_features(audio_frame, sr, config):
    """설정에 따른 특징 추출."""
    n_fft = min(FFT_SIZE, len(audio_frame))
    windowed = audio_frame[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))

    num_bands = config['num_bands']
    max_freq = config['max_freq']
    fb = build_mel_filterbank(sr, n_fft, num_bands, 80, max_freq)

    mel_energies = np.zeros(num_bands, dtype=np.float32)
    for m in range(num_bands):
        mel_energies[m] = np.sum(spectrum[:len(fb[m])] ** 2 * fb[m])

    log_mel = np.log(np.maximum(mel_energies, 1e-10))

    if config['use_dct']:
        # MFCC: c1~c13
        num_coeffs = config.get('num_coeffs', 13)
        dct_matrix = build_dct_matrix(num_coeffs, num_bands)
        mfcc = np.zeros(num_coeffs, dtype=np.float32)
        for k in range(num_coeffs):
            mfcc[k] = np.sum(log_mel * dct_matrix[k + 1])
        return mfcc
    else:
        # log-mel 직접 반환
        return log_mel


def get_frame_energy(audio_frame):
    return np.sum(audio_frame ** 2)


def cosine_distance(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 2.0
    return 1 - dot / (na * nb)


def f2_band_energy(audio_frame, sr, low=1200, high=1800):
    """F2 대역(1200-1800Hz) 에너지 비율."""
    n_fft = min(FFT_SIZE, len(audio_frame))
    windowed = audio_frame[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = np.sum(spectrum) + 1e-10
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(spectrum[mask]) / total


# ── 평가 ──

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


def evaluate_condition(config, verbose=False):
    """하나의 설정으로 전체 화자 평가."""
    all_results = []

    for dirname, speaker_id, label in SPEAKER_DIRS:
        cal_files, test_files = collect_speaker_files(dirname)
        if not cal_files or not test_files:
            continue

        use_top50 = config.get('top50_frames', False)
        use_f2 = config.get('f2_assist', False)

        # ── 캘리브레이션 ──
        cal_features = {v: [] for v in VOWELS}
        cal_energies = {v: [] for v in VOWELS}

        for path, vowel, fname in cal_files:
            audio, sr = load_audio(path)
            frames = slice_frames(audio)
            for frame in frames:
                rms = np.sqrt(np.mean(frame ** 2))
                if rms < MIN_RMS:
                    continue
                feat = extract_features(frame, sr, config)
                cal_features[vowel].append(feat)
                cal_energies[vowel].append(get_frame_energy(frame))

        # 에너지 상위 50% 프레임만 (캘리브레이션)
        prototypes = {}
        proto_f2 = {}
        for v in VOWELS:
            feats = cal_features[v]
            if len(feats) < 10:
                continue

            if use_top50:
                energies = cal_energies[v]
                indices = np.argsort(energies)
                top_half = indices[len(indices) // 2:]
                feats = [feats[i] for i in top_half]

            prototypes[v] = np.mean(feats, axis=0).astype(np.float32)

        # F2 에너지 프로토타입 (으/우 보조)
        if use_f2:
            for v in ['으', '우', '오']:
                f2_vals = []
                for path, vowel, fname in cal_files:
                    if vowel != v:
                        continue
                    audio, sr = load_audio(path)
                    frames = slice_frames(audio)
                    for frame in frames:
                        rms = np.sqrt(np.mean(frame ** 2))
                        if rms < MIN_RMS:
                            continue
                        f2_vals.append(f2_band_energy(frame, sr))
                if f2_vals:
                    proto_f2[v] = np.mean(f2_vals)

        # ── 테스트 ──
        speaker_results = []
        for path, gt, fname in test_files:
            audio, sr = load_audio(path)
            frames = slice_frames(audio)

            frame_data = []  # (feat, energy, f2_energy)
            for frame in frames:
                rms = np.sqrt(np.mean(frame ** 2))
                if rms < MIN_RMS:
                    continue
                feat = extract_features(frame, sr, config)
                energy = get_frame_energy(frame)
                f2_e = f2_band_energy(frame, sr) if use_f2 else 0
                frame_data.append((feat, energy, f2_e))

            if not frame_data:
                speaker_results.append({'gt': gt, 'pred': '?', 'fname': fname})
                continue

            # 에너지 상위 50% (테스트)
            if use_top50 and len(frame_data) > 2:
                energies = [d[1] for d in frame_data]
                threshold = np.median(energies)
                frame_data = [d for d in frame_data if d[1] >= threshold]

            # 프레임별 예측 → 다수결
            frame_preds = []
            for feat, energy, f2_e in frame_data:
                pred, conf = None, 0
                min_dist = float('inf')
                for vowel, proto in prototypes.items():
                    dist = cosine_distance(feat, proto)
                    if dist < min_dist:
                        min_dist = dist
                        pred = vowel

                # F2 보조: 으/우 경계에서 재판단
                if use_f2 and pred in ['으', '우'] and proto_f2:
                    eu_f2 = proto_f2.get('으', 0)
                    oo_f2 = proto_f2.get('우', 0)
                    if eu_f2 > 0 and oo_f2 > 0:
                        mid = (eu_f2 + oo_f2) / 2
                        if f2_e > mid:
                            pred = '으'
                        else:
                            pred = '우'

                frame_preds.append(pred)

            vote = Counter(frame_preds)
            file_pred = vote.most_common(1)[0][0]
            vote_ratio = vote.most_common(1)[0][1] / len(frame_preds)

            speaker_results.append({
                'gt': gt, 'pred': file_pred,
                'vote_ratio': vote_ratio, 'fname': fname,
                'speaker': label
            })

        all_results.extend(speaker_results)

    # 집계
    correct = sum(1 for r in all_results if r['gt'] == r['pred'])
    total = len(all_results)
    acc = correct / total * 100 if total > 0 else 0

    # 화자별
    by_speaker = {}
    for r in all_results:
        spk = r.get('speaker', '?')
        by_speaker.setdefault(spk, []).append(r)

    # 모음별
    by_vowel = {}
    for v in VOWELS:
        vr = [r for r in all_results if r['gt'] == v]
        if vr:
            vc = sum(1 for r in vr if r['gt'] == r['pred'])
            by_vowel[v] = (vc, len(vr))

    return {
        'correct': correct, 'total': total, 'acc': acc,
        'by_speaker': by_speaker, 'by_vowel': by_vowel,
        'results': all_results
    }


def print_detail(result, label):
    """상세 출력."""
    print(f'\n  [{label}] 전체: {result["correct"]}/{result["total"]} ({result["acc"]:.1f}%)')

    # 화자별
    for spk, rs in sorted(result['by_speaker'].items()):
        sc = sum(1 for r in rs if r['gt'] == r['pred'])
        ou = [r for r in rs if r['gt'] in ['오', '우']]
        ou_c = sum(1 for r in ou if r['gt'] == r['pred'])
        print(f'    {spk:<16s}: {sc:2d}/{len(rs):2d} ({sc/len(rs)*100:5.1f}%)  오/우={ou_c}/{len(ou)}')

    # 모음별
    print(f'    모음별:')
    for v in VOWELS:
        if v in result['by_vowel']:
            vc, vt = result['by_vowel'][v]
            wrong = [r['pred'] for r in result['results'] if r['gt'] == v and r['pred'] != v]
            wrong_str = ''
            if wrong:
                wc = Counter(wrong)
                wrong_str = '  ' + ', '.join(f'→{k}({n})' for k, n in wc.most_common())
            print(f'      {v}: {vc:2d}/{vt:2d} ({vc/vt*100:5.1f}%){wrong_str}')


def main():
    print('=' * 70)
    print('  MFCC+코사인 개선 실험 (6개 조건 비교)')
    print('=' * 70)

    conditions = [
        ('A: baseline (MFCC13, mel26, 4kHz)', {
            'use_dct': True, 'num_coeffs': 13,
            'num_bands': 26, 'max_freq': 4000,
        }),
        ('B: log-mel 직접 (mel26, 4kHz)', {
            'use_dct': False,
            'num_bands': 26, 'max_freq': 4000,
        }),
        ('C: log-mel 40band (4kHz)', {
            'use_dct': False,
            'num_bands': 40, 'max_freq': 4000,
        }),
        ('D: log-mel 40band (5.5kHz)', {
            'use_dct': False,
            'num_bands': 40, 'max_freq': 5500,
        }),
        ('E: D + top50% 프레임', {
            'use_dct': False,
            'num_bands': 40, 'max_freq': 5500,
            'top50_frames': True,
        }),
        ('F: E + 으/우 F2 보조', {
            'use_dct': False,
            'num_bands': 40, 'max_freq': 5500,
            'top50_frames': True,
            'f2_assist': True,
        }),
    ]

    results = []
    for label, config in conditions:
        print(f'\n{"─"*70}')
        print(f'  실험: {label}')
        print(f'{"─"*70}')
        r = evaluate_condition(config, verbose=False)
        print_detail(r, label)
        results.append((label, r))

    # ── 최종 비교표 ──
    print(f'\n\n{"═"*70}')
    print(f'  최종 비교')
    print(f'{"═"*70}')
    print(f'\n  {"조건":<35s} {"전체":>12s}  {"으":>8s}  {"오/우":>8s}')
    print(f'  {"─"*35} {"─"*12}  {"─"*8}  {"─"*8}')

    for label, r in results:
        eu = r['by_vowel'].get('으', (0, 0))
        oh = r['by_vowel'].get('오', (0, 0))
        oo = r['by_vowel'].get('우', (0, 0))
        ou_c = oh[0] + oo[0]
        ou_t = oh[1] + oo[1]
        print(f'  {label:<35s} {r["correct"]:3d}/{r["total"]:3d} ({r["acc"]:5.1f}%)'
              f'  {eu[0]:2d}/{eu[1]:2d}'
              f'  {ou_c:2d}/{ou_t:2d}')

    # 모음별 상세 비교
    print(f'\n  모음별 정확도 비교:')
    header = f'  {"조건":<20s}'
    for v in VOWELS:
        header += f'  {v:>5s}'
    print(header)
    print(f'  {"─"*20}' + '  ─────' * 7)

    for label, r in results:
        short_label = label.split(':')[0] + ':' + label.split(':')[1][:12]
        row = f'  {short_label:<20s}'
        for v in VOWELS:
            vc, vt = r['by_vowel'].get(v, (0, 0))
            if vt > 0:
                row += f'  {vc/vt*100:4.0f}%'
            else:
                row += f'     -'
        row += f'  = {r["acc"]:.1f}%'
        print(row)


if __name__ == '__main__':
    main()
