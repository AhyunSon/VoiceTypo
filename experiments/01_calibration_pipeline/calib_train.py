"""
calib_train.py - 캘리브레이션 학습 스크립트
calib_wavs/<이름>/ 의 7모음 x 3테이크로 화자별 SVM을 학습한다.
1) leave-one-take-out 품질 체크 (피치 일반화 확인)
2) 전체 테이크로 최종 학습
3) 마진 임계값(정답 프레임 마진의 5퍼센타일) 자동 계산
저장: calib_model_<이름>.pkl
"""
import os
import sys
import glob
import pickle
import numpy as np
import soundfile as sf
from sklearn.svm import SVC
from feature_utils import (
    SAMPLE_RATE, FRAME_LEN, HOP_LEN, VOWELS, mfcc_frame
)

RMS_THRESH = 0.003     # 이 미만 프레임은 무음으로 보고 버림
RMS_TARGET = 0.1       # 프레임별 정규화 목표 RMS
MARGIN_PCT = 5         # 마진 임계값 퍼센타일

def wav_to_feats(path):
    """wav 하나 -> (프레임 수, 26) 특징 행렬"""
    sig, sr = sf.read(path)
    if sig.ndim > 1:
        sig = sig[:, 0]
    mfccs = []
    for s in range(0, len(sig) - FRAME_LEN, HOP_LEN):
        frame = sig[s:s + FRAME_LEN]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < RMS_THRESH:
            continue
        frame = frame * (RMS_TARGET / rms)   # 프레임별 RMS 정규화
        mfccs.append(mfcc_frame(frame))
    if len(mfccs) < 3:
        return None
    mfccs = np.array(mfccs)                  # (T, 13)
    delta = np.zeros_like(mfccs)             # 델타: 앞뒤 프레임 차이
    delta[1:-1] = (mfccs[2:] - mfccs[:-2]) / 2.0
    delta[0], delta[-1] = delta[1], delta[-2]
    return np.hstack([mfccs, delta])         # (T, 26)

def load_speaker(name):
    """{테이크번호: (X, y)} 형태로 로드"""
    out_dir = os.path.join("calib_wavs", name)
    data = {1: ([], []), 2: ([], []), 3: ([], []), 4: ([], [])}
    for vi, vowel in enumerate(VOWELS):
        for take in (1, 2, 3, 4):
            path = os.path.join(out_dir, f"{vowel}_{take}.wav")
            if not os.path.exists(path):
                print(f"경고: {path} 없음, 건너뜀")
                continue
            feats = wav_to_feats(path)
            if feats is None:
                print(f"경고: {path} 유효 프레임 부족, 건너뜀")
                continue
            data[take][0].append(feats)
            data[take][1].append(np.full(len(feats), vi))
    result = {}
    for take, (Xs, ys) in data.items():
        if Xs:
            result[take] = (np.vstack(Xs), np.concatenate(ys))
    return result

def train_svm(X, y):
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X, y)
    return clf

def main():
    name = input("화자 이름: ").strip()
    data = load_speaker(name)
    if len(data) < 3:
        print("테이크 3개가 모두 필요합니다. 종료.")
        sys.exit(1)

    # ---- 1) leave-one-take-out ----
    print("\n=== Leave-one-take-out 품질 체크 ===")
    accs = []
    for test_take in (1, 2, 3):
        train_takes = [t for t in (1, 2, 3) if t != test_take]
        Xtr = np.vstack([data[t][0] for t in train_takes])
        ytr = np.concatenate([data[t][1] for t in train_takes])
        Xte, yte = data[test_take]
        clf = train_svm(Xtr, ytr)
        acc = clf.score(Xte, yte)
        accs.append(acc)
        print(f"  테이크 {test_take} 홀드아웃: {acc*100:.1f}%")
    mean_acc = np.mean(accs)
    print(f"  평균: {mean_acc*100:.1f}%")
    if mean_acc < 0.7:
        print("  ⚠ 70% 미만 - 재녹음 권장 (피치 간 발성 차이가 클 가능성)")

    # ---- 2) 최종 학습 ----
    # ---- 2) 최종 학습 ----
    takes_final = [t for t in (1, 2, 3, 4) if t in data]
    X = np.vstack([data[t][0] for t in takes_final])
    y = np.concatenate([data[t][1] for t in takes_final])

    # ---- 3) 마진 임계값 ----
    dec = clf.decision_function(X)           # (N, 7) ovr 점수
    pred = clf.predict(X)
    correct = pred == y
    top2 = np.sort(dec[correct], axis=1)[:, -2:]
    margins = top2[:, 1] - top2[:, 0]        # 1등 - 2등 점수 차
    thresh = np.percentile(margins, MARGIN_PCT)
    print(f"마진 임계값({MARGIN_PCT}퍼센타일): {thresh:.4f}")

    # ---- 저장 ----
    out = f"calib_model_{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump({"clf": clf, "margin_thresh": thresh,
                     "vowels": VOWELS, "loto_acc": mean_acc}, f)
    print(f"저장 완료: {out}")

if __name__ == "__main__":
    main()