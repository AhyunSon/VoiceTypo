"""한국어 fine-tuned wav2vec2 CTC 모델.

Kkonjeong/wav2vec2-base-korean 모델 사용.
프레임별 자모 확률을 출력하므로,
모음 자모(ㅏㅓㅗㅜㅡㅣㅔㅐ)의 확률을 직접 읽어 모음 인식.
캘리브레이션 불필요.
"""

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2KoreanCTC:
    def __init__(self, model_name: str = "Kkonjeong/wav2vec2-base-korean"):
        print(f"[wav2vec2] Loading model: {model_name}...", flush=True)
        self._processor = Wav2Vec2Processor.from_pretrained(model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self._model.eval()
        self._target_sr = 16000

        # vocab에서 모음 자모 토큰 ID 매핑
        vocab = self._processor.tokenizer.get_vocab()
        self._vowel_jamo_ids = {}  # {token_id: 모음 한글}

        # 자모 → 모음 매핑
        jamo_to_vowel = {
            'ㅏ': '아', 'ㅓ': '어', 'ㅗ': '오', 'ㅜ': '우',
            'ㅡ': '으', 'ㅣ': '이', 'ㅔ': '에', 'ㅐ': '애',
        }

        for jamo, vowel in jamo_to_vowel.items():
            if jamo in vocab:
                self._vowel_jamo_ids[vocab[jamo]] = vowel

        print(f"[wav2vec2] Vowel tokens found: {len(self._vowel_jamo_ids)}/{len(jamo_to_vowel)}")
        for tid, v in self._vowel_jamo_ids.items():
            print(f"  token {tid} → {v}")
        print("[wav2vec2] Model loaded.", flush=True)

    def get_vowel_probs(self, audio: np.ndarray, sr: int) -> dict:
        """오디오 → 모음별 확률.

        Args:
            audio: float32 1D 배열
            sr: 샘플레이트

        Returns:
            {모음: 확률} dict. 예: {"아": 0.82, "이": 0.05, ...}
        """
        # 리샘플링
        if sr != self._target_sr:
            ratio = self._target_sr / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            idx = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[idx]

        inputs = self._processor(
            audio, sampling_rate=self._target_sr,
            return_tensors="pt", padding=False
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, T, vocab_size)

        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (T, vocab_size)

        # 모음 토큰 ID 목록
        vowel_ids = list(self._vowel_jamo_ids.keys())

        # 프레임별 모음 확률 합산 → 모음이 활성화된 프레임 찾기
        vowel_frame_sum = probs[:, vowel_ids].sum(dim=1)  # (T,)

        # 모음 확률 상위 30% 프레임만 사용
        n_frames = len(vowel_frame_sum)
        k = max(1, n_frames // 3)
        top_indices = torch.topk(vowel_frame_sum, k).indices

        # 상위 프레임에서 각 모음의 평균 확률
        top_probs = probs[top_indices]  # (k, vocab_size)
        avg_top = top_probs.mean(dim=0)  # (vocab_size,)

        result = {}
        for tid, vowel in self._vowel_jamo_ids.items():
            result[vowel] = float(avg_top[tid])

        return result
