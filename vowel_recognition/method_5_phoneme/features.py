"""다국어 음소 인식 모델 기반 모음 분류.

facebook/wav2vec2-lv-60-espeak-cv-ft 사용.
IPA 음소를 직접 출력하므로, 한국어 모음에 해당하는
IPA 음소 확률을 읽어 모음 판별. 캘리브레이션 불필요.
"""

import json
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download

# 한국어 모음 → IPA 매핑 (analyze_ipa.py 결과 기반 재조정)
# 각 모음에 여러 IPA 후보를 두고 확률 합산하여 정확도 향상
VOWEL_IPA_MULTI = {
    '아': ['a', 'aː'],          # a가 1위, aː가 장모음 변형
    '어': ['ɑ', 'ɑː'],          # ʌ→ɑ 변경 (분석결과 ɑ가 실제 1위)
    '오': ['o', 'oː'],          # o 유지
    '우': ['u', 'uː', 'ʉ'],    # u 유지 + 관련 변형
    '으': ['ɯ', 'ʉ', 'ɨ'],     # ɯ 유지 + 근접 IPA 추가 (으는 약하므로 보강)
    '이': ['i', 'iː'],          # i 압도적
    '에': ['e', 'eː', 'ɛ'],    # e 유지 + 근접 IPA
    '애': ['æ'],                 # 유지
}

# 하위 호환용 단일 매핑
VOWEL_IPA = {
    '아': 'a', '어': 'ɑ', '오': 'o', '우': 'u',
    '으': 'ɯ', '이': 'i', '에': 'e', '애': 'æ',
}


class PhonemeVowelDetector:
    def __init__(self, model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"):
        print(f"[phoneme] Loading model: {model_name}...", flush=True)
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self._model.eval()
        self._target_sr = 16000

        # vocab.json 직접 로드 (espeak 의존성 우회)
        vocab_path = hf_hub_download(model_name, "vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        # 다중 IPA 매핑: {한국어 모음: [token_id, ...]}
        self._vowel_multi_ids = {}
        # 단일 매핑 (하위 호환)
        self._vowel_token_ids = {}

        for kr_vowel, ipa_list in VOWEL_IPA_MULTI.items():
            ids = []
            for ipa in ipa_list:
                if ipa in vocab:
                    ids.append(vocab[ipa])
            self._vowel_multi_ids[kr_vowel] = ids
            if ids:
                self._vowel_token_ids[ids[0]] = kr_vowel
            try:
                ipa_str = ", ".join(ipa_list)
                id_str = ", ".join(str(i) for i in ids)
                print(f"  {kr_vowel} -> [{ipa_str}] (tokens [{id_str}])")
            except UnicodeEncodeError:
                print(f"  {kr_vowel} -> [IPA] (tokens {ids})")

        mapped = sum(1 for ids in self._vowel_multi_ids.values() if ids)
        print(f"[phoneme] {mapped}/{len(VOWEL_IPA_MULTI)} vowels mapped.")
        print("[phoneme] Model loaded.", flush=True)

    def get_vowel_probs(self, audio: np.ndarray, sr: int) -> dict:
        """오디오 → 모음별 확률.

        Returns:
            {모음: 확률} dict
        """
        # 리샘플링
        if sr != self._target_sr:
            ratio = self._target_sr / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            idx = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[idx]

        inputs = self._feature_extractor(
            audio, sampling_rate=self._target_sr,
            return_tensors="pt", padding=False
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, T, vocab_size)

        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (T, vocab_size)

        # 모든 모음 관련 토큰 ID 수집
        all_vowel_ids = []
        for ids in self._vowel_multi_ids.values():
            all_vowel_ids.extend(ids)

        if not all_vowel_ids:
            return {}

        # 모음 토큰들의 프레임별 확률 합 → 모음 활성 프레임 찾기
        vowel_frame_sum = probs[:, all_vowel_ids].sum(dim=1)  # (T,)

        # 모음 활성 상위 30% 프레임만 사용
        n_frames = len(vowel_frame_sum)
        k = max(1, n_frames // 3)
        top_indices = torch.topk(vowel_frame_sum, k).indices

        top_probs = probs[top_indices]  # (k, vocab_size)
        avg_top = top_probs.mean(dim=0)  # (vocab_size,)

        # 각 모음의 다중 IPA 토큰 확률 합산
        result = {}
        for vowel, ids in self._vowel_multi_ids.items():
            if ids:
                result[vowel] = sum(float(avg_top[tid]) for tid in ids)
            else:
                result[vowel] = 0.0

        return result
