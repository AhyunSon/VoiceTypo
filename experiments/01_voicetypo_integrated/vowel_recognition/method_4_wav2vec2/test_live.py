"""wav2vec2 모음 인식 실시간 테스트. 캘리브레이션 없이 바로 시작.

디버그 모드: 모음별 확률 분포를 표시.
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from audio_capture.capture import AudioCapture
from pitch_detection.yin import YinDetector
from pitch_detection.vad import VoiceActivityDetector
from vowel_recognition.method_4_wav2vec2.features import Wav2Vec2KoreanCTC
from vowel_recognition.method_4_wav2vec2.classifier import VOWELS

SR = 44100


def main():
    detector = YinDetector(SR)
    vad = VoiceActivityDetector()
    ctc = Wav2Vec2KoreanCTC()
    capture = AudioCapture(sample_rate=SR)

    # 실시간 버퍼 (0.4초 누적)
    audio_buf = []
    buf_samples = int(SR * 0.4)

    def on_audio(chunk, sr):
        nonlocal audio_buf
        freq, rms = detector.detect(chunk)
        vad.update(rms, freq)

        if vad.is_active:
            audio_buf.extend(chunk)
            if len(audio_buf) >= buf_samples:
                segment = np.array(audio_buf[:buf_samples], dtype=np.float32)
                audio_buf = audio_buf[buf_samples // 2:]

                probs = ctc.get_vowel_probs(segment, sr)
                if probs:
                    # 확률 내림차순 정렬
                    sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    best = sorted_p[0]
                    # 디버그: 전체 확률 분포 출력
                    dist = "  ".join(f"{v}:{p:.3f}" for v, p in sorted_p)
                    print(f'\r  [{best[0]}] {dist}  freq={freq:.0f}Hz   ',
                          end='', flush=True)
        else:
            audio_buf.clear()
            print(f'\r  (silence)                                          ',
                  end='', flush=True)

    capture.add_listener(on_audio)
    print('=== wav2vec2 Vowel Recognition (Debug) ===')
    print('Speak vowels. Ctrl+C to stop.\n')
    capture.start()
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    capture.stop()
    print('\nStopped.')


if __name__ == '__main__':
    main()
