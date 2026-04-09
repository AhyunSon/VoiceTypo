"""포먼트 + MFCC 앙상블 실험.

동일 테스트셋에서 두 분류기를 모두 돌리고,
다양한 결합 전략을 비교한다.

조건:
  A: MFCC+코사인 단독 (baseline, 89.3%)
  B: 포먼트 단독
  C: 앙상블 — 일치 시 채택, 불일치 시 confidence 비교
  D: 앙상블 — 모음별 가중치 (이/아/어=포먼트, 오/우/에=MFCC)
  E: 앙상블 — MFCC 기본 + 포먼트 확신 시 오버라이드
  F: 앙상블 — 양쪽 confidence 가중 합산
"""
import sys, os, io, wave
import numpy as np
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 프로젝트 루트 추가
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, ROOT)

from pitch_detection.yin import YinDetector
from vowel_recognition.formant_classifier import FormantVowelClassifier

M6_BASE = os.path.join(os.path.dirname(__file__), '..', 'method_6_embedding')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']
FFT_SIZE = 2048
HOP_SIZE = 1024
MIN_RMS = 0.005
NUM_MEL_BANDS = 26
NUM_COEFFS = 13
SR = 44100

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


# ── MFCC 추출 (eval_html_version.py와 동일) ──

_fb_cache = {}
_dct_cache = {}


def build_mel_filterbank(sr, fft_size):
    num_bins = fft_size // 2 + 1
    bfw = sr / fft_size
    min_mel = 2595 * np.log10(1 + 80 / 700)
    max_mel = 2595 * np.log10(1 + 4000 / 700)
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
    return mfcc


def cosine_distance(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 2.0
    return 1 - dot / (na * nb)


def classify_mfcc(mfcc, prototypes):
    """MFCC 코사인 거리 분류 → (vowel, confidence)."""
    dists = {}
    for v, proto in prototypes.items():
        dists[v] = cosine_distance(mfcc, proto)
    best = min(dists, key=dists.get)
    sorted_d = sorted(dists.values())
    conf = 0.0
    if sorted_d[1] > 0:
        conf = min(1.0, (sorted_d[1] - sorted_d[0]) / sorted_d[1])
    return best, conf


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


def build_mfcc_prototypes(cal_files):
    """캘리브레이션 파일에서 MFCC 프로토타입 구축."""
    cal_mfccs = {v: [] for v in VOWELS}
    for path, vowel, fname in cal_files:
        audio, sr = load_audio(path)
        for frame in slice_frames(audio):
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < MIN_RMS:
                continue
            mfcc = extract_mfcc(frame, sr)
            cal_mfccs[vowel].append(mfcc)
    prototypes = {}
    for v in VOWELS:
        if len(cal_mfccs[v]) > 10:
            prototypes[v] = np.mean(cal_mfccs[v], axis=0).astype(np.float32)
    return prototypes


def calibrate_formant(cal_files, formant_clf):
    """캘리브레이션 파일로 포먼트 분류기 캘리브레이션."""
    yin = YinDetector(SR)
    window = np.hanning(FFT_SIZE).astype(np.float32)

    # 모음별 F1/F2 수집
    samples = {v: [] for v in VOWELS}

    for path, vowel, fname in cal_files:
        audio, sr = load_audio(path)
        for frame in slice_frames(audio):
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < MIN_RMS:
                continue
            freq, _ = yin.detect(frame)
            windowed = frame * window
            fft_mag = np.abs(np.fft.rfft(windowed))
            fft_freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sr)

            v_pred, conf, f1, f2 = formant_clf.classify(
                frame, f0=freq, fft_mag=fft_mag, fft_freqs=fft_freqs)
            if f1 > 0 and f2 > 0:
                samples[vowel].append((f1, f2))

        # 파일 간 리셋
        formant_clf.reset()

    # 프로토타입 설정
    measured = {}
    for v, frames in samples.items():
        if len(frames) >= 10:
            f1s = [f[0] for f in frames]
            f2s = [f[1] for f in frames]
            measured[v] = (float(np.median(f1s)), float(np.median(f2s)))

    if measured:
        from vowel_recognition.formant_classifier import DEFAULT_PROTOTYPES
        new_protos = dict(DEFAULT_PROTOTYPES)
        new_protos.update(measured)
        formant_clf.set_prototypes(new_protos)
        print(f'    포먼트 캘리브레이션: {len(measured)}개 모음 측정')
    else:
        print(f'    포먼트 캘리브레이션 실패: 데이터 부족')


def evaluate_file(path, sr, yin, window, formant_clf, mfcc_protos):
    """단일 파일 평가 → 프레임별 (mfcc_pred, mfcc_conf, formant_pred, formant_conf)."""
    audio, sr = load_audio(path)
    frame_results = []

    for frame in slice_frames(audio):
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < MIN_RMS:
            continue

        # MFCC
        mfcc = extract_mfcc(frame, sr)
        m_pred, m_conf = classify_mfcc(mfcc, mfcc_protos)

        # 포먼트
        freq, _ = yin.detect(frame)
        windowed = frame * window
        fft_mag = np.abs(np.fft.rfft(windowed))
        fft_freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sr)
        f_result = formant_clf.classify(
            frame, f0=freq, fft_mag=fft_mag, fft_freqs=fft_freqs)
        f_pred = f_result[0] if f_result[0] is not None else '?'
        f_conf = f_result[1]

        frame_results.append((m_pred, m_conf, f_pred, f_conf))

    formant_clf.reset()
    return frame_results


def majority_vote(preds):
    """다수결."""
    if not preds:
        return '?'
    vote = Counter(preds)
    return vote.most_common(1)[0][0]


def main():
    print('=' * 70)
    print('  포먼트 + MFCC 앙상블 실험')
    print('=' * 70)

    yin = YinDetector(SR)
    window = np.hanning(FFT_SIZE).astype(np.float32)

    # 전체 결과 수집 (조건별)
    conditions = {}
    condition_names = [
        'A: MFCC 단독',
        'B: 포먼트 단독',
        'C: 일치→채택, 불일치→conf 비교',
        'D: 모음별 가중치',
        'E: MFCC 기본 + 포먼트 확신 오버라이드',
        'F: confidence 가중 합산',
    ]
    for name in condition_names:
        conditions[name] = []

    for dirname, speaker_id, label in SPEAKER_DIRS:
        cal_files, test_files = collect_speaker_files(dirname)
        if not cal_files or not test_files:
            continue

        # MFCC 프로토타입
        mfcc_protos = build_mfcc_prototypes(cal_files)

        # 포먼트 분류기 (화자별 새로 생성 + 캘리브레이션)
        formant_clf = FormantVowelClassifier(sample_rate=SR)
        calibrate_formant(cal_files, formant_clf)

        print(f'  {label}: test={len(test_files)} files')

        for path, gt, fname in test_files:
            frame_results = evaluate_file(
                path, SR, yin, window, formant_clf, mfcc_protos)

            if not frame_results:
                for name in condition_names:
                    conditions[name].append({
                        'gt': gt, 'pred': '?', 'speaker': label, 'fname': fname})
                continue

            m_preds = [r[0] for r in frame_results]
            m_confs = [r[1] for r in frame_results]
            f_preds = [r[2] for r in frame_results if r[2] != '?']
            f_confs = [r[3] for r in frame_results if r[2] != '?']

            # ── A: MFCC 단독 ──
            pred_a = majority_vote(m_preds)

            # ── B: 포먼트 단독 ──
            pred_b = majority_vote(f_preds) if f_preds else '?'

            # ── 프레임별 앙상블 → 파일별 다수결 ──
            ensemble_preds_c = []
            ensemble_preds_d = []
            ensemble_preds_e = []
            ensemble_preds_f = []

            # 모음별 가중치: 포먼트가 강한 모음 vs MFCC가 강한 모음
            FORMANT_STRONG = {'아', '이', '어'}  # 포먼트 신뢰
            MFCC_STRONG = {'오', '우', '에', '으'}  # MFCC 신뢰

            for m_pred, m_conf, f_pred, f_conf in frame_results:
                if f_pred == '?':
                    # 포먼트 실패 → MFCC만
                    ensemble_preds_c.append(m_pred)
                    ensemble_preds_d.append(m_pred)
                    ensemble_preds_e.append(m_pred)
                    ensemble_preds_f.append(m_pred)
                    continue

                # C: 일치→채택, 불일치→conf 비교
                if m_pred == f_pred:
                    ensemble_preds_c.append(m_pred)
                else:
                    ensemble_preds_c.append(
                        m_pred if m_conf >= f_conf else f_pred)

                # D: 모음별 가중치
                if m_pred == f_pred:
                    ensemble_preds_d.append(m_pred)
                else:
                    # 각 예측에 대해 해당 분류기의 신뢰 가중치
                    m_w = 0.6 if m_pred in MFCC_STRONG else 0.4
                    f_w = 0.6 if f_pred in FORMANT_STRONG else 0.4
                    m_score = m_conf * m_w
                    f_score = f_conf * f_w
                    ensemble_preds_d.append(
                        m_pred if m_score >= f_score else f_pred)

                # E: MFCC 기본 + 포먼트 확신 시 오버라이드
                if f_conf > 0.7 and f_pred in FORMANT_STRONG:
                    ensemble_preds_e.append(f_pred)
                else:
                    ensemble_preds_e.append(m_pred)

                # F: confidence 가중 합산 (모든 모음에 대해)
                # 간단히: 양쪽 예측에 confidence를 합산
                scores = {}
                scores[m_pred] = scores.get(m_pred, 0) + m_conf
                scores[f_pred] = scores.get(f_pred, 0) + f_conf
                ensemble_preds_f.append(
                    max(scores, key=scores.get))

            pred_c = majority_vote(ensemble_preds_c)
            pred_d = majority_vote(ensemble_preds_d)
            pred_e = majority_vote(ensemble_preds_e)
            pred_f = majority_vote(ensemble_preds_f)

            for name, pred in [
                ('A: MFCC 단독', pred_a),
                ('B: 포먼트 단독', pred_b),
                ('C: 일치→채택, 불일치→conf 비교', pred_c),
                ('D: 모음별 가중치', pred_d),
                ('E: MFCC 기본 + 포먼트 확신 오버라이드', pred_e),
                ('F: confidence 가중 합산', pred_f),
            ]:
                conditions[name].append({
                    'gt': gt, 'pred': pred, 'speaker': label, 'fname': fname})

    # ── 결과 출력 ──
    print(f'\n\n{"═"*70}')
    print(f'  결과 비교')
    print(f'{"═"*70}')

    summary = []
    for name in condition_names:
        results = conditions[name]
        correct = sum(1 for r in results if r['gt'] == r['pred'])
        total = len(results)
        acc = correct / total * 100 if total > 0 else 0

        print(f'\n  [{name}] 전체: {correct}/{total} ({acc:.1f}%)')

        # 화자별
        speakers = sorted(set(r['speaker'] for r in results))
        for sp in speakers:
            sr_ = [r for r in results if r['speaker'] == sp]
            sc = sum(1 for r in sr_ if r['gt'] == r['pred'])
            ou = [r for r in sr_ if r['gt'] in ['오', '우']]
            ou_c = sum(1 for r in ou if r['gt'] == r['pred'])
            eu = [r for r in sr_ if r['gt'] == '으']
            eu_c = sum(1 for r in eu if r['gt'] == r['pred'])
            print(f'    {sp:<16s}: {sc:2d}/{len(sr_):2d} ({sc/len(sr_)*100:5.1f}%)  '
                  f'오/우={ou_c}/{len(ou)}  으={eu_c}/{len(eu)}')

        # 모음별
        print(f'    모음별:')
        for v in VOWELS:
            vr = [r for r in results if r['gt'] == v]
            if not vr:
                continue
            vc = sum(1 for r in vr if r['gt'] == r['pred'])
            wrong = [r['pred'] for r in vr if r['gt'] != r['pred']]
            ws = ''
            if wrong:
                wc = Counter(wrong)
                ws = '  ' + ', '.join(f'→{k}({n})' for k, n in wc.most_common())
            print(f'      {v}: {vc:2d}/{len(vr):2d} ({vc/len(vr)*100:5.1f}%){ws}')

        summary.append((name, correct, total, acc))

    # ── 최종 요약 ──
    print(f'\n\n{"═"*70}')
    print(f'  최종 요약')
    print(f'{"═"*70}')
    print(f'  {"조건":<40s} {"정확도":>12s}  {"Δ":>6s}')
    print(f'  {"─"*40} {"─"*12}  {"─"*6}')
    baseline_acc = summary[0][3]
    for name, c, t, acc in summary:
        delta = acc - baseline_acc
        d_str = f'{delta:+.1f}' if delta != 0 else '  —'
        print(f'  {name:<40s} {c:3d}/{t:3d} ({acc:5.1f}%)  {d_str}')


if __name__ == '__main__':
    main()
