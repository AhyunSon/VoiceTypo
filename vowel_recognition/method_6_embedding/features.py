"""wav2vec2/HuBERT 중간 레이어 임베딩 추출기.

pooling 모드:
  - mean: 평균만 (768차원)
  - mean_std: 평균+표준편차 (1536차원)

formant 옵션:
  - F1/F2를 추가하여 원순/후설 모음 구분 보강
"""

import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


def extract_formants(audio: np.ndarray, sr: int, n_formants: int = 2) -> list:
    """LPC 기반 포먼트(F1/F2) 추출. method_2에서 가져옴."""
    LPC_ORDER = 12

    # 프리엠퍼시스
    emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    windowed = emphasized * np.hamming(len(emphasized))

    # 자기상관
    n = len(windowed)
    r = np.zeros(LPC_ORDER + 1)
    for i in range(LPC_ORDER + 1):
        r[i] = np.sum(windowed[:n - i] * windowed[i:])
    if r[0] < 1e-10:
        return [0.0] * n_formants

    # Levinson-Durbin
    a = np.zeros(LPC_ORDER + 1)
    e = r[0]
    a[0] = 1.0
    for i in range(1, LPC_ORDER + 1):
        acc = sum(a[j] * r[i - j] for j in range(i))
        k = -acc / max(e, 1e-12)
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new
        e *= (1 - k * k)
        if e <= 0:
            break

    # 근 → 주파수
    poly = np.concatenate(([1.0], a[1:LPC_ORDER + 1]))
    roots = np.roots(poly)
    roots = roots[np.imag(roots) > 0.01]
    if len(roots) == 0:
        return [0.0] * n_formants

    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2.0 * np.pi))

    bandwidths = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.abs(roots))
    valid = bandwidths < 400
    freqs = freqs[valid]
    freqs = freqs[(freqs > 50) & (freqs < 5500)]

    if len(freqs) == 0:
        return [0.0] * n_formants

    freqs = np.sort(freqs)
    result = freqs[:n_formants].tolist()
    while len(result) < n_formants:
        result.append(0.0)
    return result


def extract_formants_windowed(audio: np.ndarray, sr: int) -> np.ndarray:
    """오디오를 여러 윈도우로 나눠 F1/F2 중간값 추출. 안정적."""
    frame_size = int(0.03 * sr)  # 30ms
    hop = frame_size // 2
    f1_list, f2_list = [], []

    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start:start + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 0.01:  # 무음 스킵
            continue
        formants = extract_formants(frame, sr, 2)
        if formants[0] > 0 and formants[1] > 0:
            f1_list.append(formants[0])
            f2_list.append(formants[1])

    if f1_list:
        f1 = np.median(f1_list)
        f2 = np.median(f2_list)
    else:
        f1, f2 = 0.0, 0.0

    # Hz를 정규화 (대략 0~1 범위로)
    f1_norm = f1 / 1000.0  # F1: 보통 200~900Hz
    f2_norm = f2 / 3000.0  # F2: 보통 700~2500Hz
    return np.array([f1_norm, f2_norm], dtype=np.float32)


class EmbeddingExtractor:
    def __init__(self, model_name: str = "facebook/wav2vec2-base",
                 layers: tuple = (6, 7, 8, 9),
                 pooling: str = "mean",
                 use_formants: bool = False):
        print(f"[embedding] Loading model: {model_name}...", flush=True)
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self._model = Wav2Vec2Model.from_pretrained(model_name)
        self._model.eval()
        self._target_sr = 16000
        self._layers = layers
        self._pooling = pooling
        self._use_formants = use_formants
        self._hidden_dim = self._model.config.hidden_size  # 768

        # 최종 출력 차원 계산
        if pooling == "mean_std":
            dim = self._hidden_dim * 2  # 1536
        else:
            dim = self._hidden_dim  # 768
        if use_formants:
            dim += 2  # +F1, +F2
        self._embed_dim = dim

        print(f"[embedding] Layers: {layers}, pooling: {pooling}, "
              f"formants: {use_formants}, dim: {self._embed_dim}")
        print("[embedding] Model loaded.", flush=True)

    @property
    def embed_dim(self):
        return self._embed_dim

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """오디오 → 임베딩 벡터.

        Returns:
            np.ndarray shape (embed_dim,)
        """
        # 포먼트는 원본 sr에서 추출 (리샘플링 전)
        if self._use_formants:
            formant_feat = extract_formants_windowed(audio, sr)

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
            outputs = self._model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        # 선택된 레이어 평균
        selected = []
        for layer_idx in self._layers:
            if layer_idx < len(hidden_states):
                selected.append(hidden_states[layer_idx].squeeze(0))

        if not selected:
            stacked = outputs.last_hidden_state.squeeze(0)
        else:
            stacked = torch.stack(selected).mean(dim=0)  # (T, 768)

        # 에너지 기반 프레임 선택 (상위 50%)
        frame_energy = stacked.norm(dim=1)
        n_frames = len(frame_energy)
        k = max(1, n_frames // 2)
        top_indices = torch.topk(frame_energy, k).indices
        selected_frames = stacked[top_indices]  # (k, 768)

        # 풀링
        mean_vec = selected_frames.mean(dim=0).numpy().astype(np.float32)

        if self._pooling == "mean_std":
            std_vec = selected_frames.std(dim=0).numpy().astype(np.float32)
            embedding = np.concatenate([mean_vec, std_vec])
        else:
            embedding = mean_vec

        # 포먼트 결합
        if self._use_formants:
            embedding = np.concatenate([embedding, formant_feat])

        return embedding
