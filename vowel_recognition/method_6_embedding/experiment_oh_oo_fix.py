"""오/우 혼동 개선 실험.

방법 2: F0(피치) + F1/F2(포먼트) 보조 특성 추가
방법 3: 다른 레이어 조합 탐색
방법 1: 2단계 분류기 (전체 → 오/우 재판정)

사용법:
  python -m vowel_recognition.method_6_embedding.experiment_oh_oo_fix \
    --audio_dir vowel_recognition/dataset
"""

import sys
import os
import argparse
import hashlib
import time
import wave
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

VOWELS_ALL = ["아", "어", "오", "우", "으", "이", "에", "애"]
TARGET_VOWELS = ["오", "우"]

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}


def syllable_to_vowel(ch):
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    medial = (code % (28 * 21)) // 28
    return _MEDIAL_TO_VOWEL.get(medial)


def parse_vowel_from_filename(filename):
    stem = os.path.splitext(filename)[0]
    first = stem.split('_')[0]
    if first in VOWELS_ALL:
        return first
    if len(first) == 1:
        return syllable_to_vowel(first)
    return None


def parse_metadata(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    meta = {}
    if len(parts) >= 1: meta['syllable'] = parts[0]
    if len(parts) >= 2: meta['gender'] = parts[1]
    if len(parts) >= 3: meta['speaker'] = parts[2]
    if len(parts) >= 4: meta['number'] = parts[3]
    if len(parts) >= 5: meta['condition'] = parts[4]
    if len(parts) >= 6: meta['pitch'] = parts[5]
    return meta


def load_audio(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        with wave.open(path, 'r') as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            ch = wf.getnchannels()
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            audio = audio.reshape(-1, ch)[:, 0]
        return audio, sr
    else:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path)
        seg = seg.set_channels(1)
        sr = seg.frame_rate
        raw = seg.raw_data
        sw = seg.sample_width
        if sw == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        return audio, sr


def get_cache_path(model_name, layers, audio_dir):
    key = f"{model_name}_{layers}_{os.path.abspath(audio_dir)}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    cache_dir = os.path.dirname(__file__)
    return os.path.join(cache_dir, f"embeddings_cache_{h}.npz")


def load_embeddings_from_cache(cache_path, filenames):
    if not os.path.exists(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=True)
    cached_files = list(data['filenames'])
    cached_embeddings = data['embeddings']
    if set(filenames) <= set(cached_files):
        idx_map = {f: i for i, f in enumerate(cached_files)}
        return np.array([cached_embeddings[idx_map[f]] for f in filenames])
    return None


# ─────────────────────────────────────
# F0 / F1 / F2 추출
# ─────────────────────────────────────
def extract_f0_windowed(audio, sr):
    """YIN 기반 F0 추출 (윈도우별 중간값)."""
    from pitch_detection.yin import YinDetector

    detector = YinDetector(sample_rate=sr)
    frame_size = int(0.03 * sr)  # 30ms
    hop = frame_size // 2
    f0_list = []

    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start:start + frame_size]
        freq, rms = detector.detect(frame)
        if freq > 0:
            f0_list.append(freq)

    if f0_list:
        return np.median(f0_list), np.std(f0_list), np.min(f0_list), np.max(f0_list)
    return 0.0, 0.0, 0.0, 0.0


def extract_formants_windowed(audio, sr):
    """LPC 기반 F1/F2 추출 (윈도우별 중간값)."""
    from vowel_recognition.method_2_formant_lpc.formant import extract_formants

    frame_size = int(0.03 * sr)
    hop = frame_size // 2
    f1_list, f2_list = [], []

    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.01:
            continue
        formants = extract_formants(frame, sr, 2)
        if len(formants) >= 2 and formants[0] > 0 and formants[1] > 0:
            f1_list.append(formants[0])
            f2_list.append(formants[1])

    if f1_list:
        return (np.median(f1_list), np.std(f1_list),
                np.median(f2_list), np.std(f2_list))
    return 0.0, 0.0, 0.0, 0.0


def extract_acoustic_features(audio_dir, filenames):
    """모든 파일에서 F0 + F1/F2 음향 특성 추출."""
    features = []
    print(f"\nF0+F1/F2 음향 특성 추출 ({len(filenames)}개)...")
    t0 = time.perf_counter()

    for i, fname in enumerate(filenames):
        audio, sr = load_audio(os.path.join(audio_dir, fname))

        f0_med, f0_std, f0_min, f0_max = extract_f0_windowed(audio, sr)
        f1_med, f1_std, f2_med, f2_std = extract_formants_windowed(audio, sr)

        # F2-F1 차이 (오/우 구분에 유용할 수 있음)
        f2_f1_diff = f2_med - f1_med if f1_med > 0 and f2_med > 0 else 0.0
        # F1/F2 비율
        f1_f2_ratio = f1_med / f2_med if f2_med > 0 else 0.0

        feat = [
            f0_med / 500.0,      # F0 정규화 (대략 0~1)
            f0_std / 100.0,      # F0 변동성
            f1_med / 1000.0,     # F1 정규화
            f1_std / 200.0,      # F1 안정성
            f2_med / 3000.0,     # F2 정규화
            f2_std / 500.0,      # F2 안정성
            f2_f1_diff / 2000.0, # F2-F1 차이
            f1_f2_ratio,         # F1/F2 비율
        ]
        features.append(feat)

        if (i + 1) % 40 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(filenames) - i - 1)
            print(f"  [{i+1}/{len(filenames)}] ETA: {eta:.0f}s")

    total = time.perf_counter() - t0
    print(f"완료: {total:.1f}초")
    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────
# LOSO 평가 공통 함수
# ─────────────────────────────────────
def loso_binary_eval(X, vowels, speakers, label=""):
    """오/우 이진 LOSO 평가. 결과 딕셔너리 반환."""
    unique_speakers = sorted(set(speakers))
    all_gt, all_pred = [], []
    per_speaker = {}

    for held_out in unique_speakers:
        test_mask = speakers == held_out
        train_mask = ~test_mask

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        clf = SVC(kernel='rbf', C=10, gamma='scale')
        clf.fit(X_train, vowels[train_mask])
        preds = clf.predict(X_test)

        y_test = vowels[test_mask]
        all_gt.extend(y_test)
        all_pred.extend(preds)

        acc = np.mean(preds == y_test) * 100
        oh_m = y_test == '오'
        oo_m = y_test == '우'
        oh_acc = np.mean(preds[oh_m] == '오') * 100 if any(oh_m) else 0
        oo_acc = np.mean(preds[oo_m] == '우') * 100 if any(oo_m) else 0
        per_speaker[held_out] = (acc, oh_acc, oo_acc, sum(oh_m), sum(oo_m))

    all_gt = np.array(all_gt)
    all_pred = np.array(all_pred)
    total = np.mean(all_gt == all_pred) * 100
    oh_total = np.mean(all_pred[all_gt == '오'] == '오') * 100
    oo_total = np.mean(all_pred[all_gt == '우'] == '우') * 100

    cm = confusion_matrix(all_gt, all_pred, labels=['오', '우'])

    return {
        'label': label,
        'total': total, 'oh': oh_total, 'oo': oo_total,
        'cm': cm, 'per_speaker': per_speaker,
    }


def print_result(r):
    """결과 출력."""
    print(f"\n  [{r['label']}]")
    print(f"    전체: {r['total']:.1f}%  |  오: {r['oh']:.1f}%  |  우: {r['oo']:.1f}%")
    cm = r['cm']
    print(f"    혼동행렬:  오→오:{cm[0,0]}  오→우:{cm[0,1]}  |  우→오:{cm[1,0]}  우→우:{cm[1,1]}")
    for spk, (acc, oh_a, oo_a, n_oh, n_oo) in sorted(r['per_speaker'].items()):
        print(f"    {spk:8s}: {acc:5.1f}% (오:{oh_a:.0f}% [{n_oh}], 우:{oo_a:.0f}% [{n_oo}])")


# ─────────────────────────────────────
# 방법 2: F0 + 포먼트 보조 특성
# ─────────────────────────────────────
def experiment_acoustic_features(X_emb, acoustic_feat, vowels, speakers):
    """임베딩 + 음향 특성 결합 실험."""
    print("\n" + "=" * 60)
    print("  방법 2: F0 + F1/F2 보조 특성 추가")
    print("=" * 60)

    results = []

    # 베이스라인: 임베딩만
    r = loso_binary_eval(X_emb, vowels, speakers, "베이스라인 (임베딩만, 1024d)")
    results.append(r)
    print_result(r)

    # 음향 특성만
    r = loso_binary_eval(acoustic_feat, vowels, speakers, "음향 특성만 (F0+F1/F2, 8d)")
    results.append(r)
    print_result(r)

    # 임베딩 + 음향 (단순 결합)
    X_combined = np.hstack([X_emb, acoustic_feat])
    r = loso_binary_eval(X_combined, vowels, speakers,
                         f"임베딩 + 음향 (concat, {X_combined.shape[1]}d)")
    results.append(r)
    print_result(r)

    # 임베딩 + 음향 (음향에 가중치)
    for weight in [5, 10, 20, 50]:
        X_weighted = np.hstack([X_emb, acoustic_feat * weight])
        r = loso_binary_eval(X_weighted, vowels, speakers,
                             f"임베딩 + 음향*{weight} ({X_weighted.shape[1]}d)")
        results.append(r)
        print_result(r)

    # 음향 특성 서브셋 실험
    # F1만 (오: F1 높음 ~500Hz, 우: F1 낮음 ~300Hz)
    f1_only = acoustic_feat[:, [2, 3]]  # F1 median, F1 std
    X_f1 = np.hstack([X_emb, f1_only * 20])
    r = loso_binary_eval(X_f1, vowels, speakers, "임베딩 + F1*20 (2d)")
    results.append(r)
    print_result(r)

    # F2-F1 차이만
    diff_only = acoustic_feat[:, [6, 7]]  # F2-F1 diff, F1/F2 ratio
    X_diff = np.hstack([X_emb, diff_only * 20])
    r = loso_binary_eval(X_diff, vowels, speakers, "임베딩 + (F2-F1,비율)*20 (2d)")
    results.append(r)
    print_result(r)

    # F0 + F1 + F2-F1
    f0_f1_diff = acoustic_feat[:, [0, 2, 6]]
    X_sel = np.hstack([X_emb, f0_f1_diff * 20])
    r = loso_binary_eval(X_sel, vowels, speakers, "임베딩 + (F0,F1,F2-F1)*20 (3d)")
    results.append(r)
    print_result(r)

    return results


# ─────────────────────────────────────
# 방법 3: 레이어 조합 탐색
# ─────────────────────────────────────
def experiment_layer_combinations(audio_dir, all_audio_files, filenames, vowels, speakers,
                                  acoustic_feat, model_name):
    """다른 레이어/레이어 조합으로 오/우 성능 탐색."""
    print("\n" + "=" * 60)
    print("  방법 3: 레이어 조합 탐색 (오/우 이진)")
    print("=" * 60)

    # layer_cache에서 모든 레이어 로드
    import glob
    cache_dir = os.path.dirname(__file__)
    layer_caches = glob.glob(os.path.join(cache_dir, "layer_cache_*.npz"))

    # XLSR-53 layer cache 찾기
    target_cache = None
    for cp in layer_caches:
        data = np.load(cp, allow_pickle=True)
        if 'model' in data and 'xlsr' in str(data['model']).lower():
            target_cache = cp
            break
        # shape check: XLSR-53 has 25 layers, 1024d
        if 'all_layers' in data:
            shape = data['all_layers'].shape
            if len(shape) == 3 and shape[2] == 1024:
                target_cache = cp
                break

    if target_cache is None:
        # 캐시 키로 직접 탐색
        for cp in layer_caches:
            data = np.load(cp, allow_pickle=True)
            keys = list(data.keys())
            if 'all_layers' in keys:
                shape = data['all_layers'].shape
                print(f"  캐시 발견: {os.path.basename(cp)}, shape={shape}, keys={keys}")
                if shape[2] == 1024:  # XLSR-53 hidden dim
                    target_cache = cp
                    break

    if target_cache is None:
        print("  XLSR-53 layer cache를 찾을 수 없습니다.")
        print("  layer_probing.py를 먼저 실행하세요.")

        # 대안: 개별 레이어 캐시에서 탐색
        print("\n  개별 레이어 캐시에서 탐색...")
        layer_results = []
        for layer_num in [4, 8, 10, 12, 14, 16, 18, 20]:
            layers = (layer_num,)
            cache_key = f"{model_name}_mean_f0"
            cache_path = get_cache_path(cache_key, layers, audio_dir)
            emb = load_embeddings_from_cache(cache_path, all_audio_files)
            if emb is not None:
                idx_map = {f: i for i, f in enumerate(all_audio_files)}
                X = np.array([emb[idx_map[f]] for f in filenames])
                r = loso_binary_eval(X, vowels, speakers, f"Layer {layer_num}")
                layer_results.append((layer_num, r))
                print(f"    Layer {layer_num}: {r['total']:.1f}% (오:{r['oh']:.1f}%, 우:{r['oo']:.1f}%)")
        return layer_results

    # layer cache 로드
    print(f"  캐시 로드: {os.path.basename(target_cache)}")
    data = np.load(target_cache, allow_pickle=True)
    all_layers = data['all_layers']  # (n_files, n_layers, hidden_dim)
    cached_files = list(data['filenames'])

    print(f"  shape: {all_layers.shape}")

    # 오/우 파일만 인덱싱
    idx_map = {f: i for i, f in enumerate(cached_files)}
    missing = [f for f in filenames if f not in idx_map]
    if missing:
        print(f"  경고: {len(missing)}개 파일이 캐시에 없음")
        # 있는 파일만 사용
        valid_mask = np.array([f in idx_map for f in filenames])
        filenames_valid = [f for f in filenames if f in idx_map]
        vowels_valid = vowels[valid_mask]
        speakers_valid = speakers[valid_mask]
        acoustic_valid = acoustic_feat[valid_mask]
    else:
        filenames_valid = filenames
        vowels_valid = vowels
        speakers_valid = speakers
        acoustic_valid = acoustic_feat

    file_indices = [idx_map[f] for f in filenames_valid]

    n_layers = all_layers.shape[1]
    print(f"  총 레이어 수: {n_layers}")

    # 단일 레이어 탐색
    print(f"\n  ── 단일 레이어 (오/우 이진 LOSO) ──")
    layer_results = []
    best_layer = -1
    best_acc = 0

    for layer_idx in range(n_layers):
        X = all_layers[file_indices, layer_idx, :]
        r = loso_binary_eval(X, vowels_valid, speakers_valid, f"L{layer_idx}")
        layer_results.append((layer_idx, r))
        marker = " ★" if r['total'] > best_acc else ""
        if r['total'] > best_acc:
            best_acc = r['total']
            best_layer = layer_idx
        print(f"    Layer {layer_idx:2d}: {r['total']:5.1f}% (오:{r['oh']:5.1f}%, 우:{r['oo']:5.1f}%){marker}")

    print(f"\n  최적 단일 레이어: Layer {best_layer} ({best_acc:.1f}%)")

    # 최적 레이어 + 음향 특성
    X_best = all_layers[file_indices, best_layer, :]
    X_best_acoustic = np.hstack([X_best, acoustic_valid * 20])
    r = loso_binary_eval(X_best_acoustic, vowels_valid, speakers_valid,
                         f"L{best_layer} + 음향*20")
    print(f"\n  Layer {best_layer} + 음향*20: {r['total']:.1f}% (오:{r['oh']:.1f}%, 우:{r['oo']:.1f}%)")

    # 레이어 범위 조합 (상위 3개 레이어 중심)
    print(f"\n  ── 레이어 범위 조합 ──")
    sorted_layers = sorted(layer_results, key=lambda x: x[1]['total'], reverse=True)
    top3 = [l for l, _ in sorted_layers[:3]]
    print(f"  상위 3개 레이어: {top3}")

    # 인접 레이어 범위
    range_results = []
    for start in range(max(0, best_layer - 4), min(n_layers - 1, best_layer + 3)):
        for end in range(start + 1, min(n_layers, start + 5)):
            X_range = all_layers[file_indices, start:end+1, :].mean(axis=1)
            r = loso_binary_eval(X_range, vowels_valid, speakers_valid, f"L{start}-{end}")
            range_results.append((start, end, r))

    range_results.sort(key=lambda x: x[2]['total'], reverse=True)
    print(f"\n  레이어 범위 상위 5:")
    for start, end, r in range_results[:5]:
        print(f"    L{start}-{end}: {r['total']:.1f}% (오:{r['oh']:.1f}%, 우:{r['oo']:.1f}%)")

    # 최적 범위 + 음향 특성
    if range_results:
        bs, be, br = range_results[0]
        X_br = all_layers[file_indices, bs:be+1, :].mean(axis=1)
        X_br_acoustic = np.hstack([X_br, acoustic_valid * 20])
        r = loso_binary_eval(X_br_acoustic, vowels_valid, speakers_valid,
                             f"L{bs}-{be} + 음향*20")
        print(f"\n  L{bs}-{be} + 음향*20: {r['total']:.1f}% (오:{r['oh']:.1f}%, 우:{r['oo']:.1f}%)")

    return layer_results


# ─────────────────────────────────────
# 방법 1: 2단계 분류기
# ─────────────────────────────────────
def experiment_two_stage(X_emb_all, vowels_all, speakers_all,
                         acoustic_feat_all, audio_dir):
    """전체 7모음 분류 후, 오/우 판정만 2단계로 재판정."""
    print("\n" + "=" * 60)
    print("  방법 1: 2단계 분류기 (전체→오/우 재판정)")
    print("=" * 60)

    unique_speakers = sorted(set(speakers_all))
    all_gt, all_pred_baseline, all_pred_twostage = [], [], []

    for held_out in unique_speakers:
        test_mask = speakers_all == held_out
        train_mask = ~test_mask

        # 1단계: 전체 7모음 분류
        scaler1 = StandardScaler()
        X_train_1 = scaler1.fit_transform(X_emb_all[train_mask])
        X_test_1 = scaler1.transform(X_emb_all[test_mask])

        clf1 = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        clf1.fit(X_train_1, vowels_all[train_mask])
        preds1 = clf1.predict(X_test_1)
        probs1 = clf1.predict_proba(X_test_1)

        # 2단계: 1단계에서 오 또는 우로 판정된 것만 재판정
        oh_oo_mask_train = np.isin(vowels_all[train_mask], TARGET_VOWELS)
        oh_oo_pred_mask = np.isin(preds1, TARGET_VOWELS)

        # 2단계 학습: 오/우 훈련 데이터 + 음향 특성
        X_train_emb_ou = X_emb_all[train_mask][oh_oo_mask_train]
        X_train_acoustic_ou = acoustic_feat_all[train_mask][oh_oo_mask_train]
        X_train_2 = np.hstack([X_train_emb_ou, X_train_acoustic_ou * 20])
        y_train_2 = vowels_all[train_mask][oh_oo_mask_train]

        if len(set(y_train_2)) < 2:
            preds2 = preds1.copy()
        else:
            scaler2 = StandardScaler()
            X_train_2s = scaler2.fit_transform(X_train_2)

            clf2 = SVC(kernel='rbf', C=10, gamma='scale')
            clf2.fit(X_train_2s, y_train_2)

            # 2단계 예측: 오/우로 판정된 테스트 샘플만
            preds2 = preds1.copy()
            if any(oh_oo_pred_mask):
                X_test_emb_ou = X_emb_all[test_mask][oh_oo_pred_mask]
                X_test_acoustic_ou = acoustic_feat_all[test_mask][oh_oo_pred_mask]
                X_test_2 = np.hstack([X_test_emb_ou, X_test_acoustic_ou * 20])
                X_test_2s = scaler2.transform(X_test_2)
                preds2[oh_oo_pred_mask] = clf2.predict(X_test_2s)

        y_test = vowels_all[test_mask]
        all_gt.extend(y_test)
        all_pred_baseline.extend(preds1)
        all_pred_twostage.extend(preds2)

    all_gt = np.array(all_gt)
    all_pred_baseline = np.array(all_pred_baseline)
    all_pred_twostage = np.array(all_pred_twostage)

    # 전체 7모음 정확도
    baseline_acc = np.mean(all_gt == all_pred_baseline) * 100
    twostage_acc = np.mean(all_gt == all_pred_twostage) * 100

    print(f"\n  전체 7모음 정확도:")
    print(f"    1단계만 (베이스라인): {baseline_acc:.1f}%")
    print(f"    2단계 (오/우 재판정): {twostage_acc:.1f}%  ({twostage_acc - baseline_acc:+.1f}%)")

    # 오/우만
    ou_mask = np.isin(all_gt, TARGET_VOWELS)
    if any(ou_mask):
        bl_ou = np.mean(all_pred_baseline[ou_mask] == all_gt[ou_mask]) * 100
        ts_ou = np.mean(all_pred_twostage[ou_mask] == all_gt[ou_mask]) * 100
        print(f"\n  오/우만 정확도:")
        print(f"    1단계만: {bl_ou:.1f}%")
        print(f"    2단계:   {ts_ou:.1f}%  ({ts_ou - bl_ou:+.1f}%)")

    # 모음별 상세
    print(f"\n  모음별 변화:")
    for v in VOWELS_ALL:
        mask = all_gt == v
        if not any(mask):
            continue
        bl = np.mean(all_pred_baseline[mask] == v) * 100
        ts = np.mean(all_pred_twostage[mask] == v) * 100
        diff = ts - bl
        marker = " ◀" if abs(diff) > 1 else ""
        print(f"    {v}: {bl:.1f}% → {ts:.1f}%  ({diff:+.1f}%){marker}")


# ═══════════════════════════════════════
# 메인
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="오/우 혼동 개선 실험")
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--model', default='facebook/wav2vec2-large-xlsr-53')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--method', default='all', choices=['2', '3', '1', 'all'],
                        help='실험 방법 (2: 음향특성, 3: 레이어, 1: 2단계, all: 전부)')
    args = parser.parse_args()

    audio_dir = args.audio_dir
    model_name = args.model
    layers = (args.layer,)

    # ── 데이터 로드 ──
    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    all_files = sorted([f for f in os.listdir(audio_dir)
                        if os.path.splitext(f)[1].lower() in audio_exts])

    # 전체 데이터
    all_samples = []
    for f in all_files:
        vowel = parse_vowel_from_filename(f)
        if vowel is None:
            continue
        meta = parse_metadata(f)
        all_samples.append((f, vowel, meta.get('speaker', '?')))

    all_filenames = [f for f, _, _ in all_samples]
    all_vowels = np.array([v for _, v, _ in all_samples])
    all_speakers = np.array([s for _, _, s in all_samples])

    # 오/우만
    ou_mask = np.isin(all_vowels, TARGET_VOWELS)
    ou_filenames = [all_filenames[i] for i in range(len(all_filenames)) if ou_mask[i]]
    ou_vowels = all_vowels[ou_mask]
    ou_speakers = all_speakers[ou_mask]

    print(f"전체 데이터: {len(all_samples)}개")
    print(f"오/우 데이터: {len(ou_filenames)}개 (오:{sum(ou_vowels=='오')}, 우:{sum(ou_vowels=='우')})")

    # ── 임베딩 로드 ──
    cache_key = f"{model_name}_mean_f0"
    cache_path = get_cache_path(cache_key, layers, audio_dir)

    all_embeddings = load_embeddings_from_cache(cache_path, all_filenames)
    if all_embeddings is None:
        print(f"캐시 없음: {cache_path}")
        print("train_and_eval.py를 먼저 실행하세요.")
        sys.exit(1)

    print(f"임베딩 로드: {all_embeddings.shape}")
    ou_embeddings = all_embeddings[ou_mask]

    # ── 음향 특성 추출 ──
    acoustic_cache = os.path.join(os.path.dirname(__file__), 'acoustic_features_cache.npz')
    if os.path.exists(acoustic_cache):
        data = np.load(acoustic_cache, allow_pickle=True)
        cached_files = list(data['filenames'])
        if set(all_filenames) <= set(cached_files):
            idx_map = {f: i for i, f in enumerate(cached_files)}
            all_acoustic = np.array([data['features'][idx_map[f]] for f in all_filenames])
            print(f"음향 특성 캐시 로드: {all_acoustic.shape}")
        else:
            all_acoustic = extract_acoustic_features(audio_dir, all_filenames)
            np.savez(acoustic_cache,
                     filenames=np.array(all_filenames, dtype=object),
                     features=all_acoustic)
    else:
        all_acoustic = extract_acoustic_features(audio_dir, all_filenames)
        np.savez(acoustic_cache,
                 filenames=np.array(all_filenames, dtype=object),
                 features=all_acoustic)
        print(f"음향 특성 캐시 저장: {acoustic_cache}")

    ou_acoustic = all_acoustic[ou_mask]

    # 음향 특성 통계 출력
    print(f"\n오/우 음향 특성 비교:")
    feat_names = ['F0_med', 'F0_std', 'F1_med', 'F1_std', 'F2_med', 'F2_std', 'F2-F1', 'F1/F2']
    denorm = [500, 100, 1000, 200, 3000, 500, 2000, 1]
    for i, (name, d) in enumerate(zip(feat_names, denorm)):
        oh_vals = ou_acoustic[ou_vowels == '오', i] * d
        oo_vals = ou_acoustic[ou_vowels == '우', i] * d
        unit = "Hz" if 'F' in name and '/' not in name else ""
        print(f"  {name:8s}: 오={np.mean(oh_vals):7.1f}±{np.std(oh_vals):5.1f}{unit}"
              f"  우={np.mean(oo_vals):7.1f}±{np.std(oo_vals):5.1f}{unit}"
              f"  차이={np.mean(oh_vals)-np.mean(oo_vals):+7.1f}")

    # ── 실험 실행 ──
    run_2 = args.method in ('2', 'all')
    run_3 = args.method in ('3', 'all')
    run_1 = args.method in ('1', 'all')

    if run_2:
        experiment_acoustic_features(ou_embeddings, ou_acoustic, ou_vowels, ou_speakers)

    if run_3:
        experiment_layer_combinations(audio_dir, all_filenames, ou_filenames,
                                      ou_vowels, ou_speakers, ou_acoustic, model_name)

    if run_1:
        experiment_two_stage(all_embeddings, all_vowels, all_speakers,
                             all_acoustic, audio_dir)

    print("\n" + "=" * 60)
    print("  실험 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
