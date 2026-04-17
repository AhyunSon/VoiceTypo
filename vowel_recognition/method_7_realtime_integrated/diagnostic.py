"""
diagnostic.py — 마이크 + 분석 파이프라인 단계별 진단
실행: python diagnostic.py
"""
import sys, time, threading, collections, traceback
import numpy as np

# ── 1. sounddevice 마이크 테스트 ──────────────────────────────
print("=" * 60)
print("[1] sounddevice 마이크 테스트 (3초)")
print("=" * 60)
try:
    import sounddevice as sd

    buf = collections.deque(maxlen=44100 * 2)
    lock = threading.Lock()

    def _cb(indata, frames, ti, status):
        with lock:
            buf.extend(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=44100, channels=1,
                            blocksize=441, dtype="float32", callback=_cb)
    stream.start()

    for _ in range(30):          # 3초
        time.sleep(0.1)
        with lock:
            if len(buf) == 0:
                print("  [경고] 오디오 버퍼가 비어 있음!")
                continue
            arr = np.array(list(buf)[-4410:], dtype=np.float64)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        bar = "#" * min(40, int(rms * 800))
        print(f"  RMS={rms:.5f}  |{bar:<40}|")

    stream.stop(); stream.close()
    print("[1] 마이크 OK\n")

except Exception:
    print("[1] 마이크 오류!")
    traceback.print_exc()
    sys.exit(1)

# ── 2. VAD 테스트 ────────────────────────────────────────────
print("=" * 60)
print("[2] VAD 테스트 — 1초 캘리브레이션 후 2초 음성 감지")
print("=" * 60)
try:
    sys.path.insert(0, ".")
    from vad import AdaptiveVAD

    buf2 = collections.deque(maxlen=44100 * 2)

    def _cb2(indata, frames, ti, status):
        buf2.extend(indata[:, 0].copy())

    stream2 = sd.InputStream(samplerate=44100, channels=1,
                              blocksize=441, dtype="float32", callback=_cb2)
    stream2.start()

    print("  [캘리브레이션] 조용히 계세요...")
    calib_rms = []
    for _ in range(10):          # 1초
        time.sleep(0.1)
        needed = int(44100 * 0.3)
        if len(buf2) < needed: continue
        chunk = np.array(list(buf2)[-needed:], dtype=np.float64)
        chunk -= np.mean(chunk)
        calib_rms.append(float(np.sqrt(np.mean(chunk ** 2))))

    vad = AdaptiveVAD()
    vad.calibrate(calib_rms)
    print(f"  noise_rms={vad.noise_rms:.5f}  threshold={vad.threshold:.5f}")

    print("  [VAD 감지] 이제 말해보세요 (2초)...")
    for _ in range(20):
        time.sleep(0.1)
        needed = int(44100 * 0.3)
        if len(buf2) < needed: continue
        chunk = np.array(list(buf2)[-needed:], dtype=np.float64)
        chunk -= np.mean(chunk)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        is_v, _ = vad.check(chunk, pitch_lo=150, pitch_hi=400)
        status_str = "● 음성" if is_v else "○ 침묵"
        print(f"  RMS={rms:.5f}  {status_str}")

    stream2.stop(); stream2.close()
    print("[2] VAD OK\n")

except Exception:
    print("[2] VAD 오류!")
    traceback.print_exc()

# ── 3. Praat 포먼트 추출 테스트 ────────────────────────────────
print("=" * 60)
print("[3] Praat 포먼트 추출 테스트 (단일 청크, 말해보세요)")
print("=" * 60)
try:
    import parselmouth
    from parselmouth.praat import call
    from formant_engine import preemphasis, FormantEngine

    buf3 = collections.deque(maxlen=44100 * 2)

    def _cb3(indata, frames, ti, status):
        buf3.extend(indata[:, 0].copy())

    stream3 = sd.InputStream(samplerate=44100, channels=1,
                              blocksize=441, dtype="float32", callback=_cb3)
    stream3.start()

    print("  3초간 말해보세요...")
    time.sleep(1.0)   # 버퍼 채우기

    engine = FormantEngine()
    for i in range(10):
        time.sleep(0.3)
        needed = int(44100 * 0.3)
        if len(buf3) < needed:
            print(f"  [{i}] 버퍼 부족")
            continue
        chunk = np.array(list(buf3)[-needed:], dtype=np.float64)
        chunk -= np.mean(chunk)
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        try:
            t0 = time.time()
            res = engine.extract(chunk, "female")
            dt = time.time() - t0
            print(f"  [{i}] RMS={rms:.4f}  "
                  f"F1={res['f1']:.0f if res['f1'] else '---'}  "
                  f"F2={res['f2']:.0f if res['f2'] else '---'}  "
                  f"F3={res['f3']:.0f if res['f3'] else '---'}  "
                  f"F0={res['f0']:.0f if res['f0'] else '---'}  "
                  f"HNR={res['hnr']:.1f if res['hnr'] else '---'}  "
                  f"conf={res['confidence']:.2f}  "
                  f"시간={dt*1000:.0f}ms")
        except Exception:
            print(f"  [{i}] extract() 오류!")
            traceback.print_exc()

    stream3.stop(); stream3.close()
    print("[3] 포먼트 추출 OK\n")

except Exception:
    print("[3] 포먼트 추출 오류!")
    traceback.print_exc()

print("=" * 60)
print("진단 완료. 위 결과를 보고 문제 위치를 확인하세요.")
print("=" * 60)
