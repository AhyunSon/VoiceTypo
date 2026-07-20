"""
diag_live.py - 라이브 '우' 진단
record()로 우를 녹음(캘리브레이션과 동일 경로) -> 저장 ->
오프라인 특징 추출(wav_to_feats)로 채점.
"""
import pickle
import numpy as np
import soundfile as sf
from feature_utils import SAMPLE_RATE, VOWELS, record, trim_silence
from calib_train import wav_to_feats

name = input("화자 이름: ").strip()
with open(f"calib_model_{name}.pkl", "rb") as f:
    model = pickle.load(f)
clf = model["clf"]

input("'우~' 발성 준비되면 엔터 >")
sig = record(1.5)
trimmed = trim_silence(sig)
if trimmed is None:
    print("소리 감지 안 됨"); exit()
sf.write("diag_u.wav", trimmed, SAMPLE_RATE)

feats = wav_to_feats("diag_u.wav")
preds = clf.predict(feats)
vals, counts = np.unique(preds, return_counts=True)
print("\n프레임 판정 분포:")
for v, c in zip(vals, counts):
    print(f"  {VOWELS[v]}: {c}프레임 ({c/len(preds)*100:.0f}%)")