"""피치 감지 + VAD 실시간 테스트."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio_capture.capture import AudioCapture
from pitch_detection.yin import YinDetector
from pitch_detection.vibrato import VibratoAnalyzer
from pitch_detection.vad import VoiceActivityDetector

detector = YinDetector(44100)
vibrato = VibratoAnalyzer(44100 / 2048)
vad = VoiceActivityDetector()

prev_active = [False]

def on_audio(chunk, sr):
    freq, rms = detector.detect(chunk)
    vad.update(rms, freq)

    if freq > 0:
        vibrato.push(freq)
    rate, extent = vibrato.get()

    # VAD 상태 변화 표시
    if vad.is_active != prev_active[0]:
        tag = '>>> VOICE ON' if vad.is_active else '<<< VOICE OFF'
        print(tag)
        prev_active[0] = vad.is_active

    vib_str = f'  vib={rate:.1f}Hz/{extent:.1f}st' if rate > 0 else ''
    vad_tag = '[V]' if vad.is_active else '[ ]'
    if freq > 0:
        print(f'  {vad_tag} {freq:.1f}Hz  rms={rms:.4f}{vib_str}')
    elif rms > 0.005:
        print(f'  {vad_tag} ---  rms={rms:.4f}')


cap = AudioCapture()
cap.add_listener(on_audio)
print('=== Pitch + VAD Live Test ===')
print('[V]=voice active, [ ]=inactive')
print('Speak or hum. Press Ctrl+C to stop.')
print()
cap.start()
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
cap.stop()
print('Stopped.')
