"""HTML 버전 (Graph_ver_PSOLA_v2) 로직 그대로 재현 + 검증.

HTML 버전 특징:
  - MFCC 13계수 (c1~c13, c0 에너지 건너뜀)
  - FFT 2048, 멜 26밴드, 주파수 80~4000Hz
  - Pre-emphasis 없음
  - CMVN 없음
  - 캘리브레이션: VAD 활성 프레임의 평균 MFCC → 프로토타입
  - 분류: 코사인 거리 (최근접 프로토타입)
  - 디바운싱: 4프레임 연속

검증: remote 화자 5명 (순모음 캘리브 → 자음+모음 테스트)
"""
import sys, os, io, wave
import numpy as np
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

M6_BASE = os.path.join(os.path.dirname(__file__), '..', 'method_6_embedding')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']

SPEAKER_DIRS = [
    ('vowel-remote-001_hj', 'hj', 'HJ (여성 60대)'),
    ('vowel-remote-001_mt', 'mt', 'MT (남성 60대)'),
    ('vowel-remote-001_kdg0534 (1)', 'kdg0534', 'KDG (남성)'),
    ('vowel-remote-001_lynn03 (1)', 'lynn03', 'Lynn (여성)'),
    ('vowel-remote-001_아현 (1)', '아현', '아현 (여성 20대)'),
]

# ── HTML 버전과 동일한 파라미터 ──
FFT_SIZE = 2048
NUM_MEL_BANDS = 26
NUM_COEFFS = 13  # c1~c13 (c0 건너뜀)
MIN_FREQ = 80
MAX_FREQ = 4000
HOP_SIZE = 1024


def load_audio(path):
    with wave.open(path, 'r') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        ch = wf.getnchannels()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return a, sr


# ── HTML 버전 MFCC 구현 그대로 재현 ──

def build_mel_filterbank(sr, fft_size, num_bands, min_freq, max_freq):
    """HTML buildMelFilterbank() 그대로."""
    num_bins = fft_size // 2 + 1
    bin_freq_width = sr / fft_size

    min_mel = 2595 * np.log10(1 + min_freq / 700)
    max_mel = 2595 * np.log10(1 + max_freq / 700)

    mel_points = []
    for i in range(num_bands + 2):
        mel = min_mel + (max_mel - min_mel) * i / (num_bands + 1)
        freq = 700 * (10 ** (mel / 2595) - 1)
        mel_points.append(freq)

    bin_points = [int(np.floor(f / bin_freq_width)) for f in mel_points]

    filterbank = np.zeros((num_bands, num_bins), dtype=np.float32)
    for m in range(num_bands):
        left = bin_points[m]
        center = bin_points[m + 1]
        right = bin_points[m + 2]

        for k in range(left, min(center, num_bins)):
            if center != left:
                filterbank[m, k] = (k - left) / (center - left)
        for k in range(center, min(right + 1, num_bins)):
            if right != center:
                filterbank[m, k] = (right - k) / (right - center)

    return filterbank


def build_dct_matrix(num_coeffs, num_bands):
    """HTML buildDCTMatrix() 그대로. 정규화 없음."""
    matrix = np.zeros((num_coeffs + 1, num_bands), dtype=np.float32)
    for k in range(num_coeffs + 1):
        for n in range(num_bands):
            matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / num_bands)
    return matrix


_filterbank_cache = {}
_dct_cache = {}


def extract_mfcc_html(audio_frame, sr):
    """HTML extractMFCC() 재현.

    차이점: HTML은 AnalyserNode의 getFloatFrequencyData (magnitude)를 사용.
    여기서는 직접 FFT 후 magnitude를 구함.
    c0(에너지)를 건너뛰고 c1~c13 반환.
    """
    n_fft = min(FFT_SIZE, len(audio_frame))

    # HTML에서는 pre-emphasis 없음 — 바로 FFT
    # 윈도우도 AnalyserNode 기본값(Blackman) 사용하지만,
    # 재현을 위해 Hann 윈도우 적용 (Web Audio 기본)
    windowed = audio_frame[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))  # magnitude (not power)

    # 필터뱅크
    cache_key = (sr, n_fft)
    if cache_key not in _filterbank_cache:
        _filterbank_cache[cache_key] = build_mel_filterbank(sr, n_fft, NUM_MEL_BANDS, MIN_FREQ, MAX_FREQ)
        _dct_cache[cache_key] = build_dct_matrix(NUM_COEFFS, NUM_MEL_BANDS)

    filterbank = _filterbank_cache[cache_key]
    dct_matrix = _dct_cache[cache_key]

    # HTML: magnitude² * filter (power spectrum through filterbank)
    mel_energies = np.zeros(NUM_MEL_BANDS, dtype=np.float32)
    for m in range(NUM_MEL_BANDS):
        mel_energies[m] = np.sum(spectrum[:len(filterbank[m])] ** 2 * filterbank[m])

    # Log
    log_mel = np.log(np.maximum(mel_energies, 1e-10))

    # DCT: c1~c13 (c0 건너뜀)
    mfcc = np.zeros(NUM_COEFFS, dtype=np.float32)
    for k in range(NUM_COEFFS):
        mfcc[k] = np.sum(log_mel * dct_matrix[k + 1])  # k+1: c0 건너뜀

    return mfcc


def cosine_distance(a, b):
    """HTML cosineDistance() 그대로."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 2.0
    return 1 - dot / (norm_a * norm_b)


def classify_vowel(mfcc, prototypes):
    """HTML classifyVowel() 그대로."""
    min_dist = float('inf')
    second_min = float('inf')
    best = '아'

    for vowel, proto in prototypes.items():
        dist = cosine_distance(mfcc, proto)
        if dist < min_dist:
            second_min = min_dist
            min_dist = dist
            best = vowel
        elif dist < second_min:
            second_min = dist

    conf = 0
    if second_min > 0:
        conf = min(1.0, (second_min - min_dist) / second_min)

    return best, conf


def slice_frames(audio, frame_size=FFT_SIZE, hop_size=HOP_SIZE):
    frames = []
    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frames.append(audio[start:start + frame_size])
    return frames


def collect_speaker_files(dirname):
    d = os.path.join(M6_BASE, dirname)
    if not os.path.isdir(d):
        return [], []

    cal_files = []
    test_files = []

    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        parts = os.path.splitext(f)[0].split('_')
        if len(parts) < 4:
            continue
        vowel = parts[2]
        syllable = parts[3]
        if vowel not in VOWELS:
            continue
        path = os.path.join(d, f)
        if vowel == syllable:
            cal_files.append((path, vowel, f))
        else:
            test_files.append((path, vowel, f))

    return cal_files, test_files


def evaluate_speaker(dirname, speaker_id, label):
    cal_files, test_files = collect_speaker_files(dirname)
    if not cal_files or not test_files:
        print(f'  [경고] {label}: 파일 부족')
        return None

    print(f'\n  {label}')
    print(f'    캘리브레이션: {len(cal_files)}개 (순모음), 테스트: {len(test_files)}개 (자음+모음)')

    # ── 캘리브레이션: 모음별 평균 MFCC 프로토타입 ──
    cal_mfccs = {v: [] for v in VOWELS}

    for path, vowel, fname in cal_files:
        audio, sr = load_audio(path)
        frames = slice_frames(audio)
        for frame in frames:
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < 0.005:  # VAD: HTML의 MIN_RMS
                continue
            mfcc = extract_mfcc_html(frame, sr)
            cal_mfccs[vowel].append(mfcc)

    # 프로토타입: 평균 (HTML computeMeanMFCC 그대로)
    prototypes = {}
    for v in VOWELS:
        if len(cal_mfccs[v]) > 10:
            prototypes[v] = np.mean(cal_mfccs[v], axis=0).astype(np.float32)
        else:
            print(f'    [경고] {v}: 샘플 부족 ({len(cal_mfccs[v])}개)')

    cal_counts = {v: len(cal_mfccs[v]) for v in VOWELS}
    print(f'    프로토타입 벡터: {", ".join(f"{v}={n}" for v, n in cal_counts.items())}')

    # ── 테스트: 프레임별 코사인 거리 → 파일별 다수결 ──
    results = []

    for path, gt, fname in test_files:
        audio, sr = load_audio(path)
        frames = slice_frames(audio)
        frame_preds = []

        for frame in frames:
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < 0.005:
                continue
            mfcc = extract_mfcc_html(frame, sr)
            pred, conf = classify_vowel(mfcc, prototypes)
            frame_preds.append(pred)

        if frame_preds:
            vote = Counter(frame_preds)
            file_pred = vote.most_common(1)[0][0]
            vote_ratio = vote.most_common(1)[0][1] / len(frame_preds)
        else:
            file_pred = '?'
            vote_ratio = 0

        results.append({'gt': gt, 'pred': file_pred, 'vote_ratio': vote_ratio, 'fname': fname})

    # ── 출력 ──
    correct = sum(1 for r in results if r['gt'] == r['pred'])
    total = len(results)
    acc = correct / total * 100 if total > 0 else 0
    print(f'    결과: {correct}/{total} ({acc:.1f}%)')

    for v in VOWELS:
        vr = [r for r in results if r['gt'] == v]
        if not vr:
            continue
        vc = sum(1 for r in vr if r['gt'] == r['pred'])
        wrong = [r['pred'] for r in vr if r['gt'] != r['pred']]
        wrong_str = ''
        if wrong:
            wc = Counter(wrong)
            wrong_str = '  오류: ' + ', '.join(f'→{k}({n})' for k, n in wc.most_common())
        print(f'      {v}: {vc}/{len(vr)} ({vc/len(vr)*100:.0f}%){wrong_str}')

    wrong_items = [r for r in results if r['gt'] != r['pred']]
    if wrong_items:
        print(f'    오답 상세:')
        for r in wrong_items:
            print(f'      {r["fname"]:30s} {r["gt"]}→{r["pred"]} (vote={r["vote_ratio"]:.0%})')

    return {'speaker': label, 'correct': correct, 'total': total, 'acc': acc, 'results': results}


def main():
    print('=' * 65)
    print('  HTML 버전 (MFCC + 코사인 거리) 검증')
    print('  = Graph_ver_PSOLA_v2/index.html 로직 재현')
    print('  Remote 화자 5명 × 순모음 캘리브 → 자음+모음 테스트')
    print('=' * 65)

    all_results = []
    for dirname, speaker_id, label in SPEAKER_DIRS:
        result = evaluate_speaker(dirname, speaker_id, label)
        if result:
            all_results.append(result)

    if not all_results:
        return

    # ── 전체 요약 ──
    print(f'\n{"#"*65}')
    print(f'  전체 요약')
    print(f'{"#"*65}')

    total_c = sum(r['correct'] for r in all_results)
    total_n = sum(r['total'] for r in all_results)
    print(f'\n  전체: {total_c}/{total_n} ({total_c/total_n*100:.1f}%)')

    print(f'\n  화자별:')
    print(f'  {"화자":<20s} {"정확도":>10s} {"오/우":>8s}')
    print(f'  {"─"*20} {"─"*10} {"─"*8}')
    for r in all_results:
        ou = [x for x in r['results'] if x['gt'] in ['오', '우']]
        ou_c = sum(1 for x in ou if x['gt'] == x['pred'])
        print(f'  {r["speaker"]:<20s} {r["correct"]:2d}/{r["total"]:2d} ({r["acc"]:5.1f}%)'
              f'  {ou_c}/{len(ou)}')

    # 모음별
    print(f'\n  모음별 전체:')
    all_r = []
    for r in all_results:
        all_r.extend(r['results'])

    for v in VOWELS:
        vr = [r for r in all_r if r['gt'] == v]
        if not vr:
            continue
        vc = sum(1 for r in vr if r['gt'] == r['pred'])
        wrong = [r['pred'] for r in vr if r['gt'] != r['pred']]
        wrong_str = ''
        if wrong:
            wc = Counter(wrong)
            wrong_str = '  ' + ', '.join(f'→{k}({n})' for k, n in wc.most_common())
        print(f'    {v}: {vc:2d}/{len(vr):2d} ({vc/len(vr)*100:5.1f}%){wrong_str}')

    # ── 비교표 ──
    print(f'\n\n{"═"*65}')
    print(f'  방법 비교 (동일 테스트셋 기준)')
    print(f'{"═"*65}')
    print(f'  {"방법":<30s} {"전체":>12s} {"지연":>8s}')
    print(f'  {"─"*30} {"─"*12} {"─"*8}')
    print(f'  {"HTML (MFCC+코사인)":<30s} {total_c:3d}/{total_n:3d} ({total_c/total_n*100:5.1f}%)  {"<1ms":>8s}')
    print(f'  {"Method 3 (MFCC+CMVN+SVM)":<30s} {"87/140 (62.1%)":>12s}  {"<1ms":>8s}')
    print(f'  {"Method 6 (XLSR-53+SVM)":<30s} {"아현 85/90 (94.4%)":>18s}  {"360ms":>8s}')


if __name__ == '__main__':
    main()
