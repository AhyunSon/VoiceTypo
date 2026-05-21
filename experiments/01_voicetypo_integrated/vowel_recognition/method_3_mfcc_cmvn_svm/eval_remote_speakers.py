"""Method 3 (MFCC+CMVN+SVM) 검증: remote 화자 5명.

각 화자별로:
  - 캘리브레이션: 순모음 녹음 (파일 1-6, 모음별 6개 = 42개)
  - 테스트: 자음+모음 녹음 (파일 7-10, 모음별 4개 = 28개)

프레임 단위가 아닌 파일 단위 평가:
  각 WAV를 2048-sample 프레임으로 슬라이스 → 프레임별 예측 → 다수결로 파일 라벨 결정.
"""
import sys, os, io, wave
import numpy as np
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# method_3 모듈 임포트
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from method_3_mfcc_cmvn_svm.features import extract_mfcc, CMVN, N_MFCC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(__file__)
M6_BASE = os.path.join(BASE, '..', 'method_6_embedding')
VOWELS = ['아', '어', '오', '우', '으', '이', '에']

SPEAKER_DIRS = [
    ('vowel-remote-001_hj', 'hj', 'HJ (여성 60대)'),
    ('vowel-remote-001_mt', 'mt', 'MT (남성 60대)'),
    ('vowel-remote-001_kdg0534 (1)', 'kdg0534', 'KDG (남성)'),
    ('vowel-remote-001_lynn03 (1)', 'lynn03', 'Lynn (여성)'),
    ('vowel-remote-001_아현 (1)', '아현', '아현 (여성 20대)'),
]

FRAME_SIZE = 2048
HOP_SIZE = 1024  # 50% 오버랩


def load_audio(path):
    with wave.open(path, 'r') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        ch = wf.getnchannels()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch)[:, 0]
    return a, sr


def slice_frames(audio, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    """오디오를 프레임으로 슬라이스."""
    frames = []
    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frames.append(audio[start:start + frame_size])
    return frames


def collect_speaker_files(dirname, speaker_id):
    """화자 디렉토리에서 파일 수집 + 캘리브/테스트 분리."""
    d = os.path.join(M6_BASE, dirname)
    if not os.path.isdir(d):
        return [], []

    cal_files = []   # 순모음 (파일 인덱스 기준 1-6)
    test_files = []  # 자음+모음 (파일 인덱스 기준 7-10)

    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        parts = os.path.splitext(f)[0].split('_')
        if len(parts) < 4:
            continue

        idx = int(parts[1])  # 001, 002, ...
        vowel = parts[2]
        syllable = parts[3]  # 순모음이면 vowel==syllable

        if vowel not in VOWELS:
            continue

        path = os.path.join(d, f)
        is_pure = (vowel == syllable)  # 순모음 여부

        if is_pure:
            cal_files.append((path, vowel, f))
        else:
            test_files.append((path, vowel, f))

    return cal_files, test_files


def extract_file_mfccs(path, sr_target=None):
    """WAV 파일에서 프레임별 MFCC 추출."""
    audio, sr = load_audio(path)
    frames = slice_frames(audio)
    mfccs = []
    for frame in frames:
        mfcc = extract_mfcc(frame, sr)
        mfccs.append(mfcc)
    return mfccs, sr


def evaluate_speaker(dirname, speaker_id, label):
    """한 화자에 대해 캘리브레이션 → 학습 → 테스트."""
    cal_files, test_files = collect_speaker_files(dirname, speaker_id)

    if not cal_files or not test_files:
        print(f'  [경고] {label}: 파일 부족 (cal={len(cal_files)}, test={len(test_files)})')
        return None

    print(f'\n  {label}')
    print(f'    캘리브레이션: {len(cal_files)}개 (순모음)')
    print(f'    테스트: {len(test_files)}개 (자음+모음)')

    # ── 캘리브레이션 ──
    cmvn = CMVN(window_size=50)
    cal_data = {v: [] for v in VOWELS}

    for path, vowel, fname in cal_files:
        mfccs, sr = extract_file_mfccs(path)
        for mfcc in mfccs:
            cmvn.update(mfcc)
            normalized = cmvn.normalize(mfcc)
            cal_data[vowel].append(normalized)

    # 모음별 벡터 수
    cal_counts = {v: len(vecs) for v, vecs in cal_data.items() if vecs}
    print(f'    캘리브레이션 벡터: {sum(cal_counts.values())}개 '
          f'({", ".join(f"{v}={n}" for v, n in cal_counts.items())})')

    # ── 학습 ──
    X_train, y_train = [], []
    for vowel, vecs in cal_data.items():
        for v in vecs:
            X_train.append(v)
            y_train.append(vowel)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    svm = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale', random_state=42)
    svm.fit(X_scaled, y_train)

    # ── 테스트 (프레임 단위 예측 → 파일 단위 다수결) ──
    results = []

    for path, gt_vowel, fname in test_files:
        mfccs, sr = extract_file_mfccs(path)
        frame_preds = []
        frame_confs = []

        for mfcc in mfccs:
            cmvn.update(mfcc)
            normalized = cmvn.normalize(mfcc)
            scaled = scaler.transform(normalized.reshape(1, -1))
            pred = svm.predict(scaled)[0]
            proba = svm.predict_proba(scaled)[0]
            conf = float(np.max(proba))
            frame_preds.append(pred)
            frame_confs.append(conf)

        # 다수결
        if frame_preds:
            vote = Counter(frame_preds)
            file_pred = vote.most_common(1)[0][0]
            # 해당 예측의 평균 신뢰도
            pred_confs = [c for p, c in zip(frame_preds, frame_confs) if p == file_pred]
            file_conf = np.mean(pred_confs)
            # 다수결 비율
            vote_ratio = vote.most_common(1)[0][1] / len(frame_preds)
        else:
            file_pred = '?'
            file_conf = 0.0
            vote_ratio = 0.0

        mark = '✓' if file_pred == gt_vowel else '✗'
        results.append({
            'gt': gt_vowel, 'pred': file_pred, 'conf': file_conf,
            'vote_ratio': vote_ratio, 'fname': fname, 'n_frames': len(frame_preds)
        })

    # ── 결과 출력 ──
    correct = sum(1 for r in results if r['gt'] == r['pred'])
    total = len(results)
    acc = correct / total * 100 if total > 0 else 0

    print(f'    결과: {correct}/{total} ({acc:.1f}%)')

    # 모음별
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

    # 오답 상세
    wrong_items = [r for r in results if r['gt'] != r['pred']]
    if wrong_items:
        print(f'    오답 상세:')
        for r in wrong_items:
            print(f'      {r["fname"]:30s} {r["gt"]}→{r["pred"]} '
                  f'(conf={r["conf"]:.0%}, vote={r["vote_ratio"]:.0%}, frames={r["n_frames"]})')

    return {
        'speaker': label,
        'correct': correct,
        'total': total,
        'acc': acc,
        'results': results,
    }


def evaluate_speaker_full_loso(dirname, speaker_id, label):
    """한 화자의 전체 70개 파일로 평가 (50% 캘리브, 50% 테스트, 교차).

    홀수 인덱스 캘리브 → 짝수 테스트, 그리고 반대. 평균.
    """
    d = os.path.join(M6_BASE, dirname)
    if not os.path.isdir(d):
        return None

    all_files = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.wav'):
            continue
        parts = os.path.splitext(f)[0].split('_')
        if len(parts) < 4:
            continue
        vowel = parts[2]
        if vowel not in VOWELS:
            continue
        all_files.append((os.path.join(d, f), vowel, f))

    if not all_files:
        return None

    # 모음별로 파일 그룹화
    by_vowel = {v: [] for v in VOWELS}
    for path, vowel, fname in all_files:
        by_vowel[vowel].append((path, vowel, fname))

    fold_results = []

    for fold in range(2):
        cmvn = CMVN(window_size=50)
        cal_data = {v: [] for v in VOWELS}
        test_set = []

        for v in VOWELS:
            files = by_vowel[v]
            for i, item in enumerate(files):
                if i % 2 == fold:
                    # 캘리브레이션
                    mfccs, sr = extract_file_mfccs(item[0])
                    for mfcc in mfccs:
                        cmvn.update(mfcc)
                        normalized = cmvn.normalize(mfcc)
                        cal_data[v].append(normalized)
                else:
                    test_set.append(item)

        # 학습
        X_train, y_train = [], []
        for vowel, vecs in cal_data.items():
            for vec in vecs:
                X_train.append(vec)
                y_train.append(vowel)

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        svm_clf = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale', random_state=42)
        svm_clf.fit(X_scaled, y_train)

        # 테스트
        for path, gt, fname in test_set:
            mfccs, sr = extract_file_mfccs(path)
            preds = []
            for mfcc in mfccs:
                cmvn.update(mfcc)
                normalized = cmvn.normalize(mfcc)
                scaled = scaler.transform(normalized.reshape(1, -1))
                preds.append(svm_clf.predict(scaled)[0])
            if preds:
                file_pred = Counter(preds).most_common(1)[0][0]
            else:
                file_pred = '?'
            fold_results.append({'gt': gt, 'pred': file_pred, 'fname': fname})

    correct = sum(1 for r in fold_results if r['gt'] == r['pred'])
    total = len(fold_results)
    return correct, total


def main():
    print('=' * 65)
    print('  Method 3 (MFCC+CMVN+SVM) 검증')
    print('  Remote 화자 5명 × 캘리브레이션(순모음) → 테스트(자음+모음)')
    print('=' * 65)

    # ── 테스트 1: 순모음 캘리브 → 자음+모음 테스트 ──
    print(f'\n{"#"*65}')
    print(f'  테스트 1: 순모음 캘리브레이션 → 자음+모음 테스트')
    print(f'{"#"*65}')

    all_speaker_results = []
    for dirname, speaker_id, label in SPEAKER_DIRS:
        result = evaluate_speaker(dirname, speaker_id, label)
        if result:
            all_speaker_results.append(result)

    # 전체 요약
    if all_speaker_results:
        print(f'\n  {"─"*55}')
        total_c = sum(r['correct'] for r in all_speaker_results)
        total_n = sum(r['total'] for r in all_speaker_results)
        print(f'  전체: {total_c}/{total_n} ({total_c/total_n*100:.1f}%)')

        print(f'\n  화자별 요약:')
        print(f'  {"화자":<20s} {"정확도":>10s} {"오/우":>10s}')
        print(f'  {"─"*20} {"─"*10} {"─"*10}')
        for r in all_speaker_results:
            ou = [x for x in r['results'] if x['gt'] in ['오', '우']]
            ou_c = sum(1 for x in ou if x['gt'] == x['pred'])
            print(f'  {r["speaker"]:<20s} {r["correct"]:2d}/{r["total"]:2d} ({r["acc"]:5.1f}%)'
                  f'  {ou_c}/{len(ou)}')

        # 모음별 전체
        print(f'\n  모음별 전체:')
        all_results = []
        for r in all_speaker_results:
            all_results.extend(r['results'])

        for v in VOWELS:
            vr = [r for r in all_results if r['gt'] == v]
            if not vr:
                continue
            vc = sum(1 for r in vr if r['gt'] == r['pred'])
            wrong = [r['pred'] for r in vr if r['gt'] != r['pred']]
            wrong_str = ''
            if wrong:
                wc = Counter(wrong)
                wrong_str = '  오류: ' + ', '.join(f'→{k}({n})' for k, n in wc.most_common())
            print(f'    {v}: {vc}/{len(vr)} ({vc/len(vr)*100:.0f}%){wrong_str}')

    # ── 테스트 2: 2-fold 교차 검증 (전체 파일 사용) ──
    print(f'\n\n{"#"*65}')
    print(f'  테스트 2: 2-Fold 교차 검증 (전체 70개, 반반 분할)')
    print(f'{"#"*65}')

    total_c2, total_n2 = 0, 0
    for dirname, speaker_id, label in SPEAKER_DIRS:
        result = evaluate_speaker_full_loso(dirname, speaker_id, label)
        if result:
            c, n = result
            print(f'    {label:<20s}: {c}/{n} ({c/n*100:.1f}%)')
            total_c2 += c
            total_n2 += n

    if total_n2 > 0:
        print(f'    {"전체":<20s}: {total_c2}/{total_n2} ({total_c2/total_n2*100:.1f}%)')

    # ── Method 6 비교 참조 ──
    print(f'\n\n{"═"*65}')
    print(f'  Method 6 (XLSR-53+SVM) 참고 수치:')
    print(f'    아현 테스트 (이전, HJ/MT 없이):   85/90 (94.4%)')
    print(f'    아현 테스트 (이후, HJ/MT 포함):   83/90 (92.2%)')
    print(f'{"═"*65}')


if __name__ == '__main__':
    main()
