"""데이터셋으로 각 한국어 모음별 실제 IPA 출력 분석.

모음별로 모델이 실제로 어떤 IPA 토큰을 출력하는지 확인하여
매핑 재조정에 사용.
"""

import sys, os, json
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]

_MEDIAL_TO_VOWEL = {
    0: '아', 1: '애', 4: '어', 5: '에',
    8: '오', 13: '우', 18: '으', 20: '이',
}


def syllable_to_vowel(ch):
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None
    medial = (code % (28 * 21)) // 28
    return _MEDIAL_TO_VOWEL.get(medial)


def parse_vowel(filename):
    stem = os.path.splitext(filename)[0]
    first = stem.split('_')[0]
    if first in VOWELS:
        return first
    if len(first) == 1:
        return syllable_to_vowel(first)
    return None


def load_audio(path):
    seg = AudioSegment.from_file(path)
    seg = seg.set_channels(1)
    sr = seg.frame_rate
    raw = seg.raw_data
    sw = seg.sample_width
    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    return audio, sr


def main():
    model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    print(f"Loading model: {model_name}...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()

    vocab_path = hf_hub_download(model_name, "vocab.json")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id2token = {v: k for k, v in vocab.items()}
    print(f"Vocab size: {len(vocab)}")

    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.mp3')])

    # 모음별로 파일 그룹핑
    groups = defaultdict(list)
    for f in files:
        v = parse_vowel(f)
        if v:
            groups[v].append(f)

    # 모음별 최대 10개만 샘플링
    MAX_PER_VOWEL = 10
    # 결과 저장: {vowel: {token_id: [probs...]}}
    vowel_token_probs = {v: defaultdict(list) for v in VOWELS}

    for vowel in VOWELS:
        sample_files = groups[vowel][:MAX_PER_VOWEL]
        if not sample_files:
            print(f"\n[{vowel}] 샘플 없음")
            continue

        print(f"\n[{vowel}] {len(sample_files)}개 샘플 분석 중...", flush=True)

        for fname in sample_files:
            fpath = os.path.join(dataset_dir, fname)
            audio, sr = load_audio(fpath)

            # 리샘플링 to 16kHz
            if sr != 16000:
                ratio = 16000 / sr
                n_out = int(len(audio) * ratio)
                indices = np.arange(n_out) / ratio
                idx = np.clip(indices.astype(int), 0, len(audio) - 1)
                audio = audio[idx]

            inputs = fe(audio, sampling_rate=16000, return_tensors="pt", padding=False)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (T, V)

            # 전체 프레임 평균
            avg_probs = probs.mean(dim=0)  # (V,)

            for tid in range(len(avg_probs)):
                vowel_token_probs[vowel][tid].append(float(avg_probs[tid]))

    # 결과 출력
    print("\n" + "=" * 70)
    print("각 한국어 모음별 모델 출력 상위 15개 IPA 토큰")
    print("=" * 70)

    # 현재 매핑도 표시
    current_map = {'아': 'a', '어': 'ʌ', '오': 'o', '우': 'u',
                   '으': 'ɯ', '이': 'i', '에': 'e', '애': 'æ'}

    for vowel in VOWELS:
        if not vowel_token_probs[vowel]:
            continue

        # 토큰별 평균 확률
        avg = {}
        for tid, vals in vowel_token_probs[vowel].items():
            avg[tid] = np.mean(vals)

        # 상위 15개
        top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:15]

        current_ipa = current_map.get(vowel, '?')
        print(f"\n[{vowel}] (현재 매핑: /{current_ipa}/)")
        print(f"  {'순위':>4s}  {'IPA':>6s}  {'token_id':>8s}  {'avg_prob':>8s}  비고")
        print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}")

        for rank, (tid, prob) in enumerate(top, 1):
            token = id2token.get(tid, '?')
            note = ""
            if token == current_ipa:
                note = "<-- 현재 매핑"
            # 특수 토큰 표시
            if token in ('<pad>', '<s>', '</s>', '<unk>'):
                note = "(특수토큰)"
            try:
                print(f"  {rank:4d}  {token:>6s}  {tid:8d}  {prob:8.5f}  {note}")
            except UnicodeEncodeError:
                print(f"  {rank:4d}  {'(ipa)':>6s}  {tid:8d}  {prob:8.5f}  {note}")

    # 추천 매핑 제안
    print("\n" + "=" * 70)
    print("추천 매핑 (각 모음에서 가장 높은 모음성 IPA)")
    print("=" * 70)

    # IPA 모음 목록 (자음/특수토큰 제외)
    ipa_vowels = set('aeiouyɑɒæɐɔəɛɜɞɤɪɨʉʊʌøœɶɘ' + 'ɯ')

    for vowel in VOWELS:
        if not vowel_token_probs[vowel]:
            continue
        avg = {}
        for tid, vals in vowel_token_probs[vowel].items():
            avg[tid] = np.mean(vals)

        # IPA 모음만 필터
        vowel_candidates = []
        for tid, prob in sorted(avg.items(), key=lambda x: x[1], reverse=True):
            token = id2token.get(tid, '')
            if len(token) == 1 and token in ipa_vowels:
                vowel_candidates.append((token, prob))
            if len(vowel_candidates) >= 5:
                break

        current_ipa = current_map.get(vowel, '?')
        top_ipa = vowel_candidates[0][0] if vowel_candidates else '?'
        changed = " *변경필요*" if top_ipa != current_ipa else ""
        cands = ", ".join(f"{t}({p:.4f})" for t, p in vowel_candidates[:5])
        try:
            print(f"  {vowel}: 현재=/{current_ipa}/  추천=/{top_ipa}/{changed}  상위5: {cands}")
        except UnicodeEncodeError:
            print(f"  {vowel}: (encoding error, see token IDs above)")


if __name__ == '__main__':
    main()
