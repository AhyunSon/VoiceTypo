"""풀링 방식 비교: energy_top50 vs middle_50.

학습 + LOSO + 실제 녹음 테스트 전부 포함.
"""
import sys, os, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import hashlib, pickle, wave, torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pydub import AudioSegment

out_dir = os.path.dirname(__file__)
audio_dir = os.path.join(out_dir, '..', 'dataset')

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


def pool_energy_top50(frames):
    energy = frames.norm(dim=1)
    k = max(1, len(energy) // 2)
    top_idx = torch.topk(energy, k).indices
    return frames[top_idx].mean(dim=0).numpy().astype(np.float32)


def pool_middle50(frames):
    T = len(frames)
    start = T // 4
    end = start + T // 2
    if end <= start:
        end = start + 1
    return frames[start:end].mean(dim=0).numpy().astype(np.float32)


def read_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        d = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return d, sr


def main():
    # ── 파일 목록 ──
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    all_files = sorted([f for f in os.listdir(audio_dir)
                        if os.path.splitext(f)[1].lower() in audio_exts])

    samples = []
    for f in all_files:
        v = parse_vowel(f)
        if v and v != '애':
            samples.append((f, v))

    filenames = [f for f, _ in samples]
    labels = np.array([v for _, v in samples])
    speakers = []
    for f, _ in samples:
        parts = f.split('_')
        speakers.append(parts[2] if len(parts) >= 3 else 'unknown')
    speakers = np.array(speakers)
    unique_speakers = np.unique(speakers)

    print("Data: %d samples, speakers: %s" % (len(samples), list(unique_speakers)))

    # ── XLSR-53 ──
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    feat_ext = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    xlsr = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
    xlsr.eval()
    print('XLSR-53 loaded')

    # ── 기존 energy_top50 캐시 로드 ──
    cache_key = "allayers_facebook/wav2vec2-large-xlsr-53_%s" % os.path.abspath(audio_dir)
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path_e = os.path.join(out_dir, "layer_cache_%s.npz" % cache_hash)
    edata = np.load(cache_path_e, allow_pickle=True)
    cached_files_e = list(edata['filenames'])
    idx_map_e = {f: i for i, f in enumerate(cached_files_e)}
    file_indices_e = [idx_map_e[f] for f in filenames]

    X_e_l16 = edata['layer_16'][file_indices_e]
    X_e_l567 = np.mean([edata['layer_%d' % l][file_indices_e] for l in [5, 6, 7]], axis=0)

    # ── middle50 캐시 ──
    cache_path_mid = os.path.join(out_dir, 'layer_cache_middle50.npz')

    if os.path.exists(cache_path_mid):
        print('Middle50 cache found')
        mdata = np.load(cache_path_mid, allow_pickle=True)
        cached_fnames_m = list(mdata['filenames'])
        idx_map_m = {f: i for i, f in enumerate(cached_fnames_m)}
        order_m = [idx_map_m[f] for f in filenames]
        X_mid_l16 = mdata['layer_16'][order_m]
        X_mid_l5 = mdata['layer_5'][order_m]
        X_mid_l6 = mdata['layer_6'][order_m]
        X_mid_l7 = mdata['layer_7'][order_m]
    else:
        print('Extracting middle50 for %d files...' % len(samples))
        X_mid_l16 = np.zeros((len(samples), 1024), dtype=np.float32)
        X_mid_l5 = np.zeros((len(samples), 1024), dtype=np.float32)
        X_mid_l6 = np.zeros((len(samples), 1024), dtype=np.float32)
        X_mid_l7 = np.zeros((len(samples), 1024), dtype=np.float32)

        t0 = time.time()
        for i, (fn, v) in enumerate(samples):
            filepath = os.path.join(audio_dir, fn)
            seg = AudioSegment.from_file(filepath)
            seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0

            inp = feat_ext(audio, sampling_rate=16000, return_tensors='pt', padding=False)
            with torch.no_grad():
                out = xlsr(**inp, output_hidden_states=True)
            h = out.hidden_states

            X_mid_l16[i] = pool_middle50(h[16].squeeze(0))
            X_mid_l5[i] = pool_middle50(h[5].squeeze(0))
            X_mid_l6[i] = pool_middle50(h[6].squeeze(0))
            X_mid_l7[i] = pool_middle50(h[7].squeeze(0))

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(samples) - i - 1)
                print('  %d/%d (%.0fs, ETA %.0fs)' % (i + 1, len(samples), elapsed, eta))

        np.savez(cache_path_mid,
                 layer_16=X_mid_l16, layer_5=X_mid_l5,
                 layer_6=X_mid_l6, layer_7=X_mid_l7,
                 filenames=np.array(filenames))
        print('Cache saved')

    X_mid_l567 = (X_mid_l5 + X_mid_l6 + X_mid_l7) / 3.0

    # ── LOSO 2단계 평가 ──
    def loso_twostage(X_s1, X_s2):
        preds_all = np.empty(len(labels), dtype=object)
        for spk in unique_speakers:
            te = speakers == spk
            tr = ~te

            sc1 = StandardScaler()
            clf1 = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
            clf1.fit(sc1.fit_transform(X_s1[tr]), labels[tr])
            p1 = clf1.predict(sc1.transform(X_s1[te]))

            ou_tr = np.isin(labels[tr], ['오', '우'])
            sc2 = StandardScaler()
            clf2 = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
            clf2.fit(sc2.fit_transform(X_s2[tr][ou_tr]), labels[tr][ou_tr])

            test_idx = np.where(te)[0]
            for j, ti in enumerate(test_idx):
                if p1[j] in ['오', '우']:
                    preds_all[ti] = clf2.predict(sc2.transform(X_s2[ti:ti + 1]))[0]
                else:
                    preds_all[ti] = p1[j]
        return preds_all

    print('\n' + '=' * 65)
    print('  LOSO 2단계 비교')
    print('=' * 65)

    for name, X1, X2 in [
        ('energy_top50', X_e_l16, X_e_l567),
        ('middle_50', X_mid_l16, X_mid_l567),
    ]:
        preds = loso_twostage(X1, X2)
        total_acc = np.mean(preds == labels) * 100
        print('\n  [%s]' % name)
        print('  ' + '-' * 55)
        for v in ["아", "어", "오", "우", "으", "이", "에"]:
            mask = labels == v
            if mask.sum() == 0:
                continue
            acc = np.mean(preds[mask] == v) * 100
            n_c = (preds[mask] == v).sum()
            print('    %s: %2d/%2d = %5.1f%%' % (v, n_c, mask.sum(), acc))
        print('    ----------')
        print('    total: %d/%d = %.1f%%' % ((preds == labels).sum(), len(labels), total_acc))

    # ── 실제 녹음 테스트 ──
    print('\n' + '=' * 65)
    print('  실제 녹음 테스트 (25개)')
    print('=' * 65)

    base = os.path.join(out_dir, 'live_recordings')
    test_files = []
    for speaker, path, vowels in [
        ('Seoul_F', base + '/session_20260310_145230', ['오', '우']),
        ('Gyeong_F', base + '/speaker_F_20s_gyeongsang', ['오', '우']),
        ('Male_20s', base + '/speaker_M_20s', ['우']),
    ]:
        for v in vowels:
            for i in range(1, 6):
                wav = os.path.join(path, '%s_%02d.wav' % (v, i))
                if os.path.exists(wav):
                    test_files.append((wav, v, speaker))

    for pool_name, pool_fn, X1_full, X2_full in [
        ('energy_top50', pool_energy_top50, X_e_l16, X_e_l567),
        ('middle_50', pool_middle50, X_mid_l16, X_mid_l567),
    ]:
        # Train on all training data
        sc1 = StandardScaler()
        clf1 = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        clf1.fit(sc1.fit_transform(X1_full), labels)

        ou_mask = np.isin(labels, ['오', '우'])
        sc2 = StandardScaler()
        clf2 = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
        clf2.fit(sc2.fit_transform(X2_full[ou_mask]), labels[ou_mask])

        correct_o, total_o = 0, 0
        correct_u, total_u = 0, 0

        for wav, v, spk in test_files:
            audio, sr = read_wav(wav)
            if sr != 16000:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * 16000 / sr))
            inp = feat_ext(audio, sampling_rate=16000, return_tensors='pt', padding=True)
            with torch.no_grad():
                out = xlsr(**inp, output_hidden_states=True)
            h = out.hidden_states

            e16 = pool_fn(h[16].squeeze(0)).reshape(1, -1)
            p1 = clf1.predict(sc1.transform(e16))[0]

            if p1 in ['오', '우']:
                e567 = (pool_fn(h[5].squeeze(0)) + pool_fn(h[6].squeeze(0)) + pool_fn(h[7].squeeze(0))) / 3.0
                final = clf2.predict(sc2.transform(e567.reshape(1, -1)))[0]
            else:
                final = p1

            if v == '오':
                total_o += 1
                if final == '오':
                    correct_o += 1
            else:
                total_u += 1
                if final == '우':
                    correct_u += 1

        print('\n  [%s]' % pool_name)
        if total_o > 0:
            print('    오: %d/%d = %d%%' % (correct_o, total_o, correct_o * 100 // total_o))
        print('    우: %d/%d = %d%%' % (correct_u, total_u, correct_u * 100 // total_u))
        tot = correct_o + correct_u
        print('    전체: %d/%d = %d%%' % (tot, total_o + total_u, tot * 100 // (total_o + total_u)))


if __name__ == '__main__':
    main()
