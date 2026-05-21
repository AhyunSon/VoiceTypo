"""모음 인식 실시간 테스트.

1단계: 캘리브레이션 (8개 모음 각 2초씩)
2단계: 학습
3단계: 실시간 인식
"""

import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from audio_capture.capture import AudioCapture
from pitch_detection.yin import YinDetector
from pitch_detection.vad import VoiceActivityDetector
from vowel_recognition.method_3_mfcc_cmvn_svm import VowelClassifier, VOWELS

SR = 44100
BLOCKSIZE = 2048
CAL_DURATION = 2.5  # 모음당 캘리브레이션 시간 (초)

detector = YinDetector(SR)
vad = VoiceActivityDetector()
classifier = VowelClassifier()
capture = AudioCapture(sample_rate=SR, blocksize=BLOCKSIZE)

# 현재 모드
mode = {'state': 'idle'}  # idle, calibrating, recognizing


def on_audio(chunk, sr):
    freq, rms = detector.detect(chunk)
    vad.update(rms, freq)

    if mode['state'] == 'calibrating':
        if vad.is_active:
            classifier.calibrate_feed(chunk, sr)
    elif mode['state'] == 'recognizing':
        if vad.is_active:
            classifier.feed(chunk, sr)
            vowel, conf = classifier.get_result()
            if vowel:
                bar = '#' * int(conf * 20)
                print(f'\r  [{vowel}] {conf:.0%} {bar:<20s}  '
                      f'freq={freq:.0f}Hz', end='', flush=True)
        else:
            print(f'\r  (silence)                              ', end='', flush=True)


def calibrate():
    print('\n=== 캘리브레이션 ===')
    print(f'각 모음을 {CAL_DURATION}초간 발성해주세요.\n')

    for vowel in VOWELS:
        input(f'  [{vowel}] 준비되면 Enter...')
        print(f'  >>> "{vowel}" 발성하세요! ({CAL_DURATION}초)')
        classifier.calibrate_start(vowel)
        time.sleep(CAL_DURATION)
        classifier.calibrate_end()
        counts = classifier.get_calibration_counts()
        print(f'  <<< 완료. {counts[vowel]}개 벡터 수집\n')

    print('캘리브레이션 데이터:')
    for v, c in classifier.get_calibration_counts().items():
        print(f'  {v}: {c}개')

    print('\n학습 중...')
    if classifier.train():
        print('학습 완료!\n')
        return True
    else:
        print('학습 실패.\n')
        return False


def recognize():
    print('=== 실시간 모음 인식 ===')
    print('소리를 내보세요. Ctrl+C로 종료.\n')
    mode['state'] = 'recognizing'
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    mode['state'] = 'idle'
    print('\n\n종료.')


def main():
    capture.add_listener(on_audio)
    capture.start()
    print('마이크 시작됨.\n')

    try:
        # 캘리브레이션
        mode['state'] = 'calibrating'
        if calibrate():
            # 실시간 인식
            recognize()
    except KeyboardInterrupt:
        pass

    capture.stop()


if __name__ == '__main__':
    main()
