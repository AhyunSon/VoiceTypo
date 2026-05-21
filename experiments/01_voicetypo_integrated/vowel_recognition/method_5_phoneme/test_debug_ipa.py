"""디버그: 각 모음 발음 시 모델이 실제로 어떤 IPA 음소를 출력하는지 확인.

모든 vocab 토큰 중 상위 10개를 보여줌.
이를 통해 정확한 IPA 매핑을 찾을 수 있음.
"""

import sys, os, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download
from audio_capture.capture import AudioCapture
from pitch_detection.yin import YinDetector
from pitch_detection.vad import VoiceActivityDetector

SR = 44100
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def main():
    print("[debug] Loading model...", flush=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.eval()

    vocab_path = hf_hub_download(MODEL_NAME, "vocab.json")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    # id → token 역매핑
    id2token = {v: k for k, v in vocab.items()}

    detector = YinDetector(SR)
    vad = VoiceActivityDetector()
    capture = AudioCapture(sample_rate=SR)

    audio_buf = []
    buf_samples = int(SR * 0.5)

    def on_audio(chunk, sr):
        nonlocal audio_buf
        freq, rms = detector.detect(chunk)
        vad.update(rms, freq)

        if vad.is_active:
            audio_buf.extend(chunk)
            if len(audio_buf) >= buf_samples:
                segment = np.array(audio_buf[:buf_samples], dtype=np.float32)
                audio_buf = audio_buf[buf_samples // 2:]

                # 리샘플링
                ratio = 16000 / sr
                n_out = int(len(segment) * ratio)
                indices = np.arange(n_out) / ratio
                idx = np.clip(indices.astype(int), 0, len(segment) - 1)
                resampled = segment[idx]

                inputs = fe(resampled, sampling_rate=16000,
                           return_tensors="pt", padding=False)
                with torch.no_grad():
                    logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0)  # (T, V)

                # 프레임 최대값 (각 토큰의 최대 확률 프레임)
                max_probs = probs.max(dim=0).values  # (V,)

                # 상위 10개 토큰
                top10 = torch.topk(max_probs, 10)
                parts = []
                for prob, tid in zip(top10.values, top10.indices):
                    token = id2token.get(tid.item(), '?')
                    parts.append(f"{token}:{prob:.3f}")
                line = "  ".join(parts)
                print(f"\r  TOP: {line}      ", end='', flush=True)
        else:
            audio_buf.clear()
            print(f'\r  (silence)                                                    ',
                  end='', flush=True)

    capture.add_listener(on_audio)
    print('=== IPA Debug: Top 10 phonemes ===')
    print('각 모음을 길게 발음하면서 어떤 IPA가 나오는지 확인.')
    print('Ctrl+C to stop.\n')
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
