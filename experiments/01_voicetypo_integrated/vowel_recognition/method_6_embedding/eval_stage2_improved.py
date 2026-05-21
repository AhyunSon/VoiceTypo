"""Stage 2 개선 실험: L0 + H1-H2 로 오/우 이진 분류기 평가."""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pickle, wave, torch, hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── H1-H2 추출 함수 ──
def extract_h1h2_from_signal(y, sr):
    frame_size = int(0.03 * sr)
    min_lag = sr // 400
    max_lag = sr // 60
    f0s = []
    for s in range(0, len(y) - frame_size, frame_size // 2):
        frame = y[s:s+frame_size]
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        if max_lag >= len(corr):
            continue
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            continue
        peak = np.argmax(search) + min_lag
        if corr[peak] > 0.3 * corr[0]:
            f0s.append(sr / peak)
    if not f0s:
        return 0.0
    f0_mean = np.mean(f0s)

    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_frames = (len(y) - frame_len) // hop + 1
    energy = np.array([np.sum(y[i*hop:i*hop+frame_len]**2) for i in range(n_frames)])
    threshold = np.percentile(energy, 50)
    h1h2_list = []
    n_fft = 2048
    for i in range(n_frames):
        if energy[i] <= threshold:
            continue
        frame = y[i*hop:i*hop+frame_len]
        w = np.hanning(len(frame))
        S = np.abs(np.fft.rfft(frame * w, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        h1_idx = np.argmin(np.abs(freqs - f0_mean))
        h2_idx = np.argmin(np.abs(freqs - 2*f0_mean))
        h1_db = 20 * np.log10(S[h1_idx] + 1e-10)
        h2_db = 20 * np.log10(S[h2_idx] + 1e-10)
        h1h2_list.append(h1_db - h2_db)
    return np.mean(h1h2_list) if h1h2_list else 0.0


def read_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        d = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return d, sr


def pool(emb):
    norms = np.linalg.norm(emb, axis=1)
    mask = norms > np.percentile(norms, 50)
    if mask.sum() < 1:
        mask = np.ones(len(norms), dtype=bool)
    return emb[mask].mean(axis=0)


def main():
    out_dir = os.path.dirname(__file__)
    audio_dir = os.path.join(out_dir, '..', 'dataset')

    # ── 1. 학습 데이터 준비 ──
    print("=" * 60)
    print("  Stage 2 개선: L0 + H1-H2 오/우 이진 분류기")
    print("=" * 60)

    cache_key = "allayers_facebook/wav2vec2-large-xlsr-53_%s" % os.path.abspath(audio_dir)
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(out_dir, "layer_cache_%s.npz" % cache_hash)

    data = np.load(cache_path, allow_pickle=True)
    cached_files = list(data['filenames'])

    VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]
    _MEDIAL_TO_VOWEL = {0: '아', 1: '애', 4: '어', 5: '에', 8: '오', 13: '우', 18: '으', 20: '이'}

    def syllable_to_vowel(ch):
        code = ord(ch) - 0xAC00
        if code < 0 or code > 11171:
            return None
        return _MEDIAL_TO_VOWEL.get((code % (28 * 21)) // 28)

    def parse_vowel(fn):
        stem = os.path.splitext(fn)[0]
        first = stem.split('_')[0]
        if first in VOWELS:
            return first
        if len(first) == 1:
            return syllable_to_vowel(first)
        return None

    # 오/우만 추출
    ou_indices = []
    ou_labels = []
    ou_speakers = []
    for i, f in enumerate(cached_files):
        v = parse_vowel(f)
        if v in ['오', '우']:
            ou_indices.append(i)
            ou_labels.append(v)
            parts = f.split('_')
            ou_speakers.append(parts[2] if len(parts) >= 3 else 'unknown')

    ou_indices = np.array(ou_indices)
    ou_labels = np.array(ou_labels)
    ou_speakers = np.array(ou_speakers)
    print("학습 데이터: 오 %d개, 우 %d개" % ((ou_labels == '오').sum(), (ou_labels == '우').sum()))
    print("화자: %s" % list(np.unique(ou_speakers)))

    # L0, L5-7 임베딩
    X_l0 = data['layer_0'][ou_indices]
    X_l567 = np.mean([data['layer_%d' % l][ou_indices] for l in [5, 6, 7]], axis=0)

    # H1-H2 for training data
    print("\n학습 데이터 H1-H2 추출 중...")
    from pydub import AudioSegment
    h1h2_train = []
    for idx, i in enumerate(ou_indices):
        fn = cached_files[i]
        filepath = os.path.join(audio_dir, fn)
        try:
            seg = AudioSegment.from_file(filepath)
            seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
            h = extract_h1h2_from_signal(samples, 16000)
        except Exception as e:
            h = 0.0
        h1h2_train.append(h)
        if (idx + 1) % 50 == 0:
            print("  %d/%d..." % (idx + 1, len(ou_indices)))

    h1h2_train = np.array(h1h2_train).reshape(-1, 1)
    print("  오 H1-H2 평균: %.1f dB" % np.mean(h1h2_train[ou_labels == '오']))
    print("  우 H1-H2 평균: %.1f dB" % np.mean(h1h2_train[ou_labels == '우']))

    # ── 2. LOSO 평가 ──
    configs = [
        ("L5-7 (현재)", X_l567),
        ("L0", X_l0),
        ("L0 + H1-H2", np.hstack([X_l0, h1h2_train])),
    ]

    print("\n" + "=" * 60)
    print("  학습 데이터 LOSO 평가")
    print("=" * 60)

    for name, X in configs:
        correct = 0
        total = 0
        for spk in np.unique(ou_speakers):
            test_mask = ou_speakers == spk
            train_mask = ~test_mask
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_mask])
            X_te = scaler.transform(X[test_mask])
            clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
            clf.fit(X_tr, ou_labels[train_mask])
            preds = clf.predict(X_te)
            correct += (preds == ou_labels[test_mask]).sum()
            total += test_mask.sum()
        acc = correct * 100.0 / total
        print("  %s: LOSO = %d/%d = %.1f%%" % (name, correct, total, acc))

    # ── 3. 실제 녹음 테스트 ──
    print("\n" + "=" * 60)
    print("  실제 녹음 파일 테스트 (25개)")
    print("=" * 60)

    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    feat_ext_m = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    xlsr = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    xlsr.eval()
    print("XLSR-53 로드 완료\n")

    base = os.path.join(out_dir, 'live_recordings')
    test_files = []
    for speaker, path, vowels in [
        ('서울여성', base + '/session_20260310_145230', ['오', '우']),
        ('경상도여성', base + '/speaker_F_20s_gyeongsang', ['오', '우']),
        ('20대남성', base + '/speaker_M_20s', ['우']),
    ]:
        for v in vowels:
            for i in range(1, 6):
                wav = os.path.join(path, '%s_%02d.wav' % (v, i))
                if os.path.exists(wav):
                    test_files.append((wav, v, speaker))

    print("테스트 파일: %d개" % len(test_files))

    # Extract test embeddings + H1-H2
    test_l0 = []
    test_l567 = []
    test_h1h2 = []
    test_labels = []
    test_spks = []

    for wav, v, spk in test_files:
        audio, sr = read_wav(wav)
        if sr != 16000:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * 16000 / sr))

        inp = feat_ext_m(audio, sampling_rate=16000, return_tensors='pt', padding=True)
        with torch.no_grad():
            out = xlsr(**inp, output_hidden_states=True)
        h = out.hidden_states

        e0 = pool(h[0].squeeze(0).detach().numpy())
        e567 = (pool(h[5].squeeze(0).detach().numpy()) +
                pool(h[6].squeeze(0).detach().numpy()) +
                pool(h[7].squeeze(0).detach().numpy())) / 3.0
        h1h2_val = extract_h1h2_from_signal(audio, 16000)

        test_l0.append(e0)
        test_l567.append(e567)
        test_h1h2.append(h1h2_val)
        test_labels.append(v)
        test_spks.append(spk)
        print("  %s %s H1-H2=%.1f" % (spk, os.path.basename(wav), h1h2_val))

    test_l0 = np.array(test_l0)
    test_l567 = np.array(test_l567)
    test_h1h2 = np.array(test_h1h2).reshape(-1, 1)
    test_labels = np.array(test_labels)
    test_spks = np.array(test_spks)

    test_X_map = {
        "L5-7 (현재)": test_l567,
        "L0": test_l0,
        "L0 + H1-H2": np.hstack([test_l0, test_h1h2]),
    }

    # Train on full training data, test on recordings
    print("\n" + "=" * 60)
    print("  결과 비교")
    print("=" * 60)

    for name, X_train in configs:
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        clf.fit(X_tr_scaled, ou_labels)

        X_test = test_X_map[name]
        X_te_scaled = scaler.transform(X_test)
        preds = clf.predict(X_te_scaled)
        proba = clf.predict_proba(X_te_scaled)

        print("\n  [%s]" % name)
        print("  " + "-" * 55)

        for spk in ['서울여성', '경상도여성', '20대남성']:
            mask = test_spks == spk
            if not mask.any():
                continue
            for v in ['오', '우']:
                v_mask = mask & (test_labels == v)
                if not v_mask.any():
                    continue
                v_preds = preds[v_mask]
                acc = (v_preds == v).sum() * 100.0 / len(v_preds)
                print("    %s %s: %d/%d = %3.0f%%  %s" % (
                    spk, v, (v_preds == v).sum(), len(v_preds), acc,
                    list(v_preds)))

        o_mask = test_labels == '오'
        u_mask = test_labels == '우'
        o_acc = (preds[o_mask] == '오').sum() * 100.0 / o_mask.sum() if o_mask.sum() > 0 else 0
        u_acc = (preds[u_mask] == '우').sum() * 100.0 / u_mask.sum() if u_mask.sum() > 0 else 0
        total_acc = (preds == test_labels).sum() * 100.0 / len(test_labels)
        print("    -- 합계 --")
        print("    오: %.0f%%  우: %.0f%%  전체: %.0f%%" % (o_acc, u_acc, total_acc))


if __name__ == '__main__':
    main()
