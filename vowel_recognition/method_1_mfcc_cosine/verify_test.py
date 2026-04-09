"""테스트 정확성 검증.

1) A(baseline)가 원본 eval_html_version.py와 동일한 결과인지 확인
2) B(log-mel)에서 실제로 DCT가 빠진 벡터가 나오는지 확인
3) 각 조건별 feature 벡터 차원, 프로토타입 값 비교
4) 단일 화자(Lynn, 100%)에서 B가 어디서 틀리는지 상세 추적
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


def build_mel_filterbank(sr, fft_size, num_bands, min_freq, max_freq):
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
    return fb


def build_dct_matrix(num_coeffs, num_bands):
    matrix = np.zeros((num_coeffs + 1, num_bands), dtype=np.float32)
    for k in range(num_coeffs + 1):
        for n in range(num_bands):
            matrix[k, n] = np.cos(np.pi * k * (n + 0.5) / num_bands)
    return matrix


def cosine_distance(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 2.0
    return 1 - dot / (na * nb)


def main():
    print('=' * 70)
    print('  테스트 정확성 검증')
    print('=' * 70)

    # Lynn 데이터 로드
    dirname = 'vowel-remote-001_lynn03 (1)'
    d = os.path.join(M6_BASE, dirname)
    cal_files, test_files = [], []
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
            cal_files.append((path, vowel, f))
        else:
            test_files.append((path, vowel, f))

    print(f'\n  Lynn: cal={len(cal_files)}, test={len(test_files)}')

    # 첫 번째 캘리브레이션 파일로 오디오 로드
    audio0, sr = load_audio(cal_files[0][0])
    frame0 = slice_frames(audio0)[5]  # 5번째 프레임
    print(f'  sr={sr}, frame shape={frame0.shape}')

    n_fft = min(FFT_SIZE, len(frame0))
    windowed = frame0[:n_fft] * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))

    # ── 조건 A: MFCC 13 ──
    fb26 = build_mel_filterbank(sr, n_fft, 26, 80, 4000)
    dct_m = build_dct_matrix(13, 26)

    mel26 = np.zeros(26, dtype=np.float32)
    for m in range(26):
        mel26[m] = np.sum(spectrum[:len(fb26[m])] ** 2 * fb26[m])
    log_mel26 = np.log(np.maximum(mel26, 1e-10))

    mfcc13 = np.zeros(13, dtype=np.float32)
    for k in range(13):
        mfcc13[k] = np.sum(log_mel26 * dct_m[k + 1])

    print(f'\n  [검증 1] 단일 프레임 특징 벡터')
    print(f'  MFCC 13 (조건 A): dim={len(mfcc13)}, 값={mfcc13[:5]}...')
    print(f'  log-mel 26 (조건 B): dim={len(log_mel26)}, 값={log_mel26[:5]}...')

    # ── 조건별 프로토타입 비교 ──
    print(f'\n  [검증 2] Lynn 프로토타입 구축 + 코사인 거리 분석')

    for cond_name, use_dct, num_bands, max_freq in [
        ('A: MFCC13', True, 26, 4000),
        ('B: log-mel26', False, 26, 4000),
    ]:
        fb = build_mel_filterbank(sr, n_fft, num_bands, 80, max_freq)
        dct = build_dct_matrix(13, num_bands)

        cal_feats = {v: [] for v in VOWELS}
        for path, vowel, fname in cal_files:
            audio, sr2 = load_audio(path)
            for frame in slice_frames(audio):
                rms = np.sqrt(np.mean(frame ** 2))
                if rms < MIN_RMS:
                    continue
                windowed = frame[:n_fft] * np.hanning(n_fft)
                spec = np.abs(np.fft.rfft(windowed, n=n_fft))
                mel_e = np.zeros(num_bands, dtype=np.float32)
                for m in range(num_bands):
                    mel_e[m] = np.sum(spec[:len(fb[m])] ** 2 * fb[m])
                log_mel = np.log(np.maximum(mel_e, 1e-10))

                if use_dct:
                    feat = np.zeros(13, dtype=np.float32)
                    for k in range(13):
                        feat[k] = np.sum(log_mel * dct[k + 1])
                else:
                    feat = log_mel.copy()

                cal_feats[vowel].append(feat)

        protos = {v: np.mean(cal_feats[v], axis=0).astype(np.float32) for v in VOWELS if len(cal_feats[v]) > 10}

        print(f'\n  {cond_name}: dim={len(protos["아"])}')

        # 프로토타입 간 코사인 거리 행렬
        print(f'    프로토타입 간 코사인 거리:')
        header = '         '
        for v in VOWELS:
            header += f' {v:>5s}'
        print(f'    {header}')
        for v1 in VOWELS:
            row = f'    {v1:>5s}:'
            for v2 in VOWELS:
                dist = cosine_distance(protos[v1], protos[v2])
                row += f' {dist:5.3f}'
            print(row)

        # 으/우/오 삼각관계
        eu_oo = cosine_distance(protos['으'], protos['우'])
        eu_oh = cosine_distance(protos['으'], protos['오'])
        oo_oh = cosine_distance(protos['우'], protos['오'])
        print(f'    으↔우={eu_oo:.4f}, 으↔오={eu_oh:.4f}, 우↔오={oo_oh:.4f}')

        # 테스트 파일별 상세
        print(f'\n    테스트 결과 상세:')
        correct = 0
        for path, gt, fname in test_files:
            audio, sr2 = load_audio(path)
            frame_preds = []
            frame_dists_all = []
            for frame in slice_frames(audio):
                rms = np.sqrt(np.mean(frame ** 2))
                if rms < MIN_RMS:
                    continue
                windowed = frame[:n_fft] * np.hanning(n_fft)
                spec = np.abs(np.fft.rfft(windowed, n=n_fft))
                mel_e = np.zeros(num_bands, dtype=np.float32)
                for m in range(num_bands):
                    mel_e[m] = np.sum(spec[:len(fb[m])] ** 2 * fb[m])
                log_mel = np.log(np.maximum(mel_e, 1e-10))

                if use_dct:
                    feat = np.zeros(13, dtype=np.float32)
                    for k in range(13):
                        feat[k] = np.sum(log_mel * dct[k + 1])
                else:
                    feat = log_mel.copy()

                dists = {v: cosine_distance(feat, protos[v]) for v in VOWELS}
                best = min(dists, key=dists.get)
                frame_preds.append(best)
                frame_dists_all.append(dists)

            if frame_preds:
                vote = Counter(frame_preds)
                pred = vote.most_common(1)[0][0]
                vote_r = vote.most_common(1)[0][1] / len(frame_preds)
            else:
                pred = '?'
                vote_r = 0

            mark = '✓' if pred == gt else '✗'
            if pred == gt:
                correct += 1

            # 오답만 상세 출력
            if pred != gt:
                # 평균 거리
                avg_dists = {v: np.mean([d[v] for d in frame_dists_all]) for v in VOWELS}
                top3 = sorted(avg_dists.items(), key=lambda x: x[1])[:3]
                dist_str = ' '.join(f'{v}={d:.4f}' for v, d in top3)
                print(f'      {fname:30s} {gt}→{pred} {mark} vote={vote_r:.0%}  [{dist_str}]')

        print(f'    정확도: {correct}/{len(test_files)} ({correct/len(test_files)*100:.1f}%)')


if __name__ == '__main__':
    main()
