"""
diag_compare.py - 캘리브레이션 '우' vs 라이브 '우' MFCC 비교
어느 계수가 벌어졌는지 확인 (C1 위주면 스펙트럼 기울기 = 거리/근접효과 의심)
"""
import os
import numpy as np
from calib_train import wav_to_feats

name = input("화자 이름: ").strip()
calib = np.vstack([wav_to_feats(os.path.join("calib_wavs", name, f"우_{t}.wav"))
                   for t in (1, 2, 3)])
live = wav_to_feats("diag_u.wav")

cm, lm = calib.mean(0), live.mean(0)
cs = calib.std(0) + 1e-6
diff_z = (lm - cm) / cs   # 캘리브 분포 기준 표준화 차이

print("\nMFCC 계수별 차이 (|z|>2면 유의미하게 밀린 것):")
for i in range(13):
    flag = " ◀◀" if abs(diff_z[i]) > 2 else ""
    print(f"  C{i:<2}: z={diff_z[i]:+.2f}{flag}")