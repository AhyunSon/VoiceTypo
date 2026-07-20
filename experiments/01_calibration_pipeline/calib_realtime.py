"""
calib_realtime.py - 캘리브레이션 모델 실시간 특징 스트림 프로토타입
프레임마다 {모음(smooth), margin, F0, RMS(raw)} 를 출력한다.
- 유성음 게이트: F0 없으면 판정만 차단 (MFCC 축적은 유지 -> 델타 연속성 보존)
- 히스테리시스: 확실한 모음이 창의 60% 이상일 때만 표시 라벨 교체
  ('?'는 라벨을 못 뺏음, 무음이 오면 라벨 백지화)
- F0: 간이 YIN (numpy 벡터화)
"""
import sys
import pickle
import numpy as np
import sounddevice as sd
from collections import deque, Counter
from feature_utils import (
    SAMPLE_RATE, FRAME_LEN, HOP_LEN, mfcc_frame
)

RMS_THRESH = 0.003
RMS_TARGET = 0.1
SMOOTH_N   = 10        # 라벨 다수결 창 (150ms)
SWITCH_RATIO = 0.6     # 라벨 교체 문턱: 창의 60% 이상 점유해야 교체
F0_MIN, F0_MAX = 70, 400
YIN_THRESHOLD = 0.15   # 골 인정 기준 (낮을수록 엄격)
PENDING_N = 5          # 교체 후보가 연속 5프레임(75ms) 유지돼야 확정

def estimate_f0(sig, fmin=F0_MIN, fmax=F0_MAX, threshold=YIN_THRESHOLD):
    """간이 YIN, numpy 벡터화 버전. F0(Hz) 또는 None."""
    x = sig - np.mean(sig)
    if np.sqrt(np.mean(x**2)) < RMS_THRESH:
        return None
    n = len(x)
    tau_min = int(SAMPLE_RATE / fmax)
    tau_max = min(int(SAMPLE_RATE / fmin), n - 1)

    # ① 차이함수 (벡터화):
    # d(tau) = sum(x[:n-tau]^2) + sum(x[tau:]^2) - 2*autocorr(tau)
    nfft = 1 << (2 * n - 1).bit_length()          # FFT 크기 (2의 거듭제곱)
    fx = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(fx * np.conj(fx))[: tau_max + 1]   # 자기상관
    csum = np.concatenate(([0.0], np.cumsum(x * x)))       # 누적 에너지
    taus = np.arange(tau_max + 1)
    e_head = csum[n - taus] - csum[0]      # sum(x[:n-tau]^2)
    e_tail = csum[n] - csum[taus]          # sum(x[tau:]^2)
    diff = e_head + e_tail - 2.0 * acf
    diff[0] = 0.0

    # ② 누적평균정규화 (옥타브 에러 억제)
    cmnd = np.ones(tau_max + 1)
    running = np.cumsum(diff[1:])
    valid = running > 0
    cmnd[1:][valid] = diff[1:][valid] * taus[1:][valid] / running[valid]

    # ③ threshold 밑 첫 골 -> 골 바닥 확정
    below = np.where(cmnd[tau_min: tau_max + 1] < threshold)[0]
    if len(below) == 0:
        return None   # 주기 없음 = 무성음
    tau = tau_min + below[0]
    while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1
    return SAMPLE_RATE / tau

def main():
    name = input("화자 이름: ").strip()
    with open(f"calib_model_{name}.pkl", "rb") as f:
        model = pickle.load(f)
    clf, thresh, vowels = model["clf"], model["margin_thresh"], model["vowels"]
    print(f"모델 로드 완료. 마진 임계값 {thresh:.4f}")
    print("발성해보세요. Ctrl+C로 종료.\n")

    sigbuf = np.zeros(0, dtype=np.float32)
    mfcc_hist = deque(maxlen=3)   # 델타용 [t-1, t, t+1]
    recent = deque(maxlen=SMOOTH_N)
    cur_label = "?"               # 히스테리시스: 현재 유지 중인 표시 라벨
    pending_label, pending_cnt = None, 0
    f0_win = deque(maxlen=4)      # F0용 60ms 창

    def callback(indata, frames, time_info, status):
        nonlocal sigbuf
        sigbuf = np.concatenate([sigbuf, indata[:, 0]])

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=HOP_LEN, callback=callback):
        while True:
            sd.sleep(5)
            while len(sigbuf) >= FRAME_LEN:
                frame = sigbuf[:FRAME_LEN]
                sigbuf = sigbuf[HOP_LEN:]

                rms = np.sqrt(np.mean(frame ** 2))   # raw RMS (버블 크기용)
                f0_win.append(frame[:HOP_LEN])

                if rms < RMS_THRESH:
                    mfcc_hist.clear()
                    recent.clear()
                    cur_label = "?"    # 발성 종료 -> 표시 라벨 백지화
                    pending_label, pending_cnt = None, 0
                    print("\r(무음)                                                  ", end="")
                    continue

                # F0: 60ms 창으로 추정 (버블 상하 움직임용)
                f0 = estimate_f0(np.concatenate(f0_win)) if len(f0_win) == 4 else None

                # MFCC는 항상 축적 (델타 연속성 유지)
                norm = frame * (RMS_TARGET / rms)
                mfcc_hist.append(mfcc_frame(norm))
                if len(mfcc_hist) < 3:
                    continue
                prev, cur, nxt = mfcc_hist
                delta = (nxt - prev) / 2.0
                feat = np.hstack([cur, delta]).reshape(1, -1)

                # 유성음 게이트: 판정만 차단, 축적은 이미 끝남
                if f0 is None:
                    raw, second, margin = "?", "-", 0.0
                else:
                    dec = clf.decision_function(feat)[0]
                    order = np.argsort(dec)
                    margin = dec[order[-1]] - dec[order[-2]]
                    second = vowels[order[-2]]
                    raw = "?" if margin < thresh else vowels[order[-1]]

                recent.append(raw)
                # 히스테리시스: 확실한 모음이 문턱(60%)을 '연속 유지'해야 교체
                top, cnt = Counter(recent).most_common(1)[0]
                if top != cur_label and top != "?" and cnt >= SMOOTH_N * SWITCH_RATIO:
                    if top == pending_label:
                        pending_cnt += 1
                    else:
                        pending_label, pending_cnt = top, 1
                    if pending_cnt >= PENDING_N:
                        cur_label = top
                        pending_label, pending_cnt = None, 0
                else:
                    pending_label, pending_cnt = None, 0
                smooth = cur_label

                # ---- 특징 묶음 (여기가 나중에 OSC로 나갈 페이로드) ----
                features = {
                    "vowel": smooth,          # 표시 글자
                    "margin": round(float(margin), 3),  # 신뢰도
                    "f0": round(f0, 1) if f0 else None, # 버블 상하
                    "rms": round(float(rms), 4),        # 버블 크기
                }
                f0_str = f"{features['f0']:.0f}Hz" if features["f0"] else "---"
                print(f"\rraw:{raw}(2등:{second})  vowel:{features['vowel']}  "
                      f"margin:{features['margin']:.2f}  f0:{f0_str}  "
                      f"rms:{features['rms']:.4f}     ", end="")

if __name__ == "__main__":
    main()