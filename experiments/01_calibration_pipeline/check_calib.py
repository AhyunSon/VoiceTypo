"""
check_calib.py - 캘리브레이션 wav를 저장된 모델로 오프라인 채점
라이브와 오프라인 결과가 다르면 파이프라인 불일치, 같이 나쁘면 데이터/모델 문제.
"""
import os
import pickle
import numpy as np
from feature_utils import VOWELS
from calib_train import wav_to_feats

name = input("화자 이름: ").strip()
with open(f"calib_model_{name}.pkl", "rb") as f:
    model = pickle.load(f)
clf = model["clf"]

print(f"\n{'모음':<4} {'정확도':<8} 오판 분포")
for vi, vowel in enumerate(VOWELS):
    preds = []
    for take in (1, 2, 3):
        path = os.path.join("calib_wavs", name, f"{vowel}_{take}.wav")
        feats = wav_to_feats(path)
        if feats is None:
            continue
        preds.extend(clf.predict(feats))
    preds = np.array(preds)
    acc = np.mean(preds == vi) * 100
    wrong = preds[preds != vi]
    dist = {VOWELS[i]: int(n) for i, n in
            zip(*np.unique(wrong, return_counts=True))} if len(wrong) else {}
    print(f"{vowel:<4} {acc:6.1f}%  {dist}")