"""포먼트 기반 모음 인식 실시간 테스트. 캘리브레이션 없이 바로 시작."""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from audio_capture.capture import AudioCapture
from pitch_detection.yin import YinDetector
from pitch_detection.vad import VoiceActivityDetector
from vowel_recognition.method_2_formant_lpc import FormantVowelClassifier

SR = 44100
detector = YinDetector(SR)
vad = VoiceActivityDetector()
classifier = FormantVowelClassifier()
capture = AudioCapture(sample_rate=SR)


def on_audio(chunk, sr):
    freq, rms = detector.detect(chunk)
    vad.update(rms, freq)

    if vad.is_active:
        classifier.feed(chunk, sr)
        vowel, conf = classifier.get_result()
        f1, f2 = classifier.get_formants()
        if vowel:
            bar = '#' * int(conf * 20)
            print(f'\r  [{vowel}] {conf:.0%} {bar:<20s}  '
                  f'F1={f1:.0f} F2={f2:.0f}  freq={freq:.0f}Hz   ',
                  end='', flush=True)
    else:
        print(f'\r  (silence)                                          ',
              end='', flush=True)


capture.add_listener(on_audio)
print('=== Formant Vowel Recognition (no calibration) ===')
print('Speak vowels. Ctrl+C to stop.\n')
capture.start()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
capture.stop()
print('\nStopped.')
