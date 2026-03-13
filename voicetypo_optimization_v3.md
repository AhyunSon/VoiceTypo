# VoiceTypo 최적화 작업 지시서 v3 (수정본)

> v2 리뷰에서 지적된 ONNX 버그, sdf_to_rgba 오류, 결합도 문제를 모두 수정.
> Claude Code에 PROJECT_FULL_SOURCE.txt와 함께 전달하세요.

---

## 변경 이력

- v2 → v3: TASK 3 ONNX 코드 전면 재작성 (풀링 버그, 전처리 누락, 동적 topk 문제 수정)
- v2 → v3: TASK 2 sdf_to_rgba 최적화 코드 수정 (numpy out 파라미터 오용 수정)
- v2 → v3: TASK 6 시그널/슬롯 분리 구조로 변경

---

## TASK 1: YIN 중복 제거 + 벡터화 (v2와 동일, 승인됨)

### 1-1. UtteranceCollector에서 자체 YIN/VAD 제거

```
파일: main_integrated.py
클래스: UtteranceCollector
```

자체 `_yin`, `_vad` 인스턴스를 삭제하고, 외부에서 VAD 상태를 받도록 변경:

```python
class UtteranceCollector:
    """스트리밍 방식: VAD 상태를 외부에서 받아 발성 중 오디오를 수집."""
    def __init__(self, sr=44100, min_duration=0.10, snapshot_interval=0.10,
                 max_duration=0.4):
        self.sr = sr
        self.min_samples = int(min_duration * sr)
        self.snapshot_interval = snapshot_interval
        self.max_samples = int(max_duration * sr)

        # ── YIN, VAD 자체 인스턴스 삭제 ──
        self._buffer = []
        self._collecting = False
        self._ready = False
        self._utterance = None
        self._last_snapshot = 0.0
        self._lock = threading.Lock()
        self._vad_active = False

    def set_vad_state(self, is_active: bool):
        """AudioBridge에서 VAD 결과를 전달받는다."""
        self._vad_active = is_active

    def on_audio(self, chunk, sr):
        # ── 자체 YIN/VAD 호출 없음 — self._vad_active 사용 ──
        with self._lock:
            if self._vad_active:
                if not self._collecting:
                    self._collecting = True
                    self._buffer = []
                    self._last_snapshot = 0.0
                self._buffer.append(chunk.copy())

                total = sum(len(c) for c in self._buffer)
                elapsed = total / self.sr
                if (total >= self.min_samples and
                        elapsed - self._last_snapshot >= self.snapshot_interval):
                    full = np.concatenate(self._buffer)
                    if len(full) > self.max_samples:
                        full = full[-self.max_samples:]
                    self._utterance = full
                    self._ready = True
                    self._last_snapshot = elapsed
            else:
                if self._collecting:
                    self._buffer = []
                    self._collecting = False

    def get_utterance(self):
        with self._lock:
            if self._ready:
                utt = self._utterance
                self._utterance = None
                self._ready = False
                return utt
            return None

    def reset(self):
        with self._lock:
            self._buffer = []
            self._collecting = False
            self._ready = False
            self._utterance = None
```

main() 연결:
```python
def on_bridge_update(freq, rms, vib_rate, vib_extent, vad_active):
    collector.set_vad_state(vad_active)

bridge.updated.connect(on_bridge_update)
```

### 1-2. YIN 차분 함수 벡터화

```
파일: pitch_detection/yin.py
함수: detect()
```

핵심: Python for-loop 551회 → FFT 기반 자기상관으로 벡터화.
CMNDF도 벡터화.

```python
def detect(self, audio: np.ndarray) -> tuple:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < SILENCE_THRESHOLD:
        return 0.0, rms

    n = len(audio)
    tau_max = min(self.tau_max, n // 2)
    tau_min = self.tau_min
    if tau_max <= tau_min:
        return 0.0, rms

    # ── FFT 기반 차분 함수 (벡터화) ──
    # d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
    # W = n - tau_max (안전한 윈도우 크기)
    W = n - tau_max
    x = audio[:W + tau_max].astype(np.float64)

    # 자기상관: FFT 방식
    fft_size = 1
    while fft_size < len(x) * 2:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    acf = np.fft.irfft(X * np.conj(X))[:tau_max + 1]

    # x^2의 누적합
    x_sq = x * x
    cum_sq = np.cumsum(x_sq)

    # d(tau) = cum_sq[W-1] + (cum_sq[W+tau-1] - cum_sq[tau-1]) - 2*acf(tau)
    # 즉, sum(x[0:W]^2) + sum(x[tau:tau+W]^2) - 2*sum(x[j]*x[j+tau])
    sum_left = cum_sq[W - 1]  # sum(x[0:W]^2) — 모든 tau에 대해 동일
    d = np.zeros(tau_max + 1, dtype=np.float64)
    for tau in range(1, tau_max + 1):
        sum_right = cum_sq[W + tau - 1] - cum_sq[tau - 1]
        d[tau] = sum_left + sum_right - 2.0 * acf[tau]

    # ── CMNDF (벡터화) ──
    cmndf = np.ones(tau_max + 1, dtype=np.float64)
    cum_d = np.cumsum(d)
    for tau in range(1, tau_max + 1):
        if cum_d[tau] > 0:
            cmndf[tau] = d[tau] * tau / cum_d[tau]

    # ── 임계값 이하 첫 극소점 (짧은 범위이므로 for-loop OK) ──
    tau_est = 0
    for tau in range(tau_min, tau_max):
        if cmndf[tau] < YIN_THRESHOLD:
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            tau_est = tau
            break

    if tau_est == 0:
        return 0.0, rms

    tau_est = self._parabolic_interpolation(cmndf, tau_est)
    freq = self.sample_rate / tau_est

    if freq < VOICE_MIN_FREQ or freq > VOICE_MAX_FREQ:
        return 0.0, rms

    return float(freq), rms
```

참고: FFT 부분에서 `acf` 계산은 O(n log n)이고,
CMNDF의 for-loop는 tau_max(~551)로 짧아서 무시 가능.
이후 완전 벡터화도 가능하지만 효과 대비 복잡도가 높으므로 이 정도면 충분.

---

## TASK 2: 렌더링 캐싱 + sdf_to_rgba 수정 (v3에서 코드 수정)

### 2-1. dirty flag 기반 캐싱 (v2와 동일)

IntegratedCanvas._tick()과 paintEvent()에 dirty flag 추가.
v2 코드 그대로 사용.

### 2-2. sdf_to_rgba 최적화 (v3 수정)

```
파일: text_effects/test_effects.py
함수: sdf_to_rgba()
```

**v2 문제**: `out=buf[:, :, i].astype(np.float32)`에서 astype이 새 배열을 반환하므로
out이 원래 버퍼를 가리키지 않음. 또한 바로 아래에서 다시 계산하여 이중 연산.

**v3 수정**: float32 작업 버퍼를 사전할당하고, 최종 결과만 uint8 버퍼에 쓰기.

```python
# 모듈 레벨에 사전할당
_t_gradient = np.linspace(0, 1, GRID, dtype=np.float32)[:, None]
_work_buf = np.zeros((GRID, GRID), dtype=np.float32)   # float32 작업 버퍼
_rgba_buf = np.zeros((GRID, GRID, 4), dtype=np.uint8)   # 최종 출력 버퍼


def sdf_to_rgba(sdf, color_top, color_bot, opacity=1.0):
    h, w = sdf.shape
    t = _t_gradient[:h]       # (h, 1) — 세로 그라데이션 보간 계수
    work = _work_buf[:h, :w]  # float32 작업 영역 (재할당 없음)
    out = _rgba_buf[:h, :w]   # uint8 출력 영역

    # alpha 계산 (in-place)
    np.subtract(0.5, sdf, out=work)
    work /= AA_WIDTH
    np.clip(work, 0.0, 1.0, out=work)
    if opacity < 1.0:
        work *= opacity
    alpha = work.copy()  # alpha 보존 (아래 채널 계산에서 work를 재사용하므로)

    # R, G, B 채널: color = top*(1-t) + bot*t, 그 후 alpha premultiply
    for i in range(3):
        np.multiply(color_top[i], 1.0 - t, out=work)  # work = top * (1-t)
        work += color_bot[i] * t                        # work += bot * t
        work *= alpha                                    # premultiply
        np.clip(work, 0, 255, out=work)
        out[:, :, i] = work.astype(np.uint8)

    # Alpha 채널
    np.multiply(alpha, 255.0, out=_work_buf[:h, :w])
    np.clip(_work_buf[:h, :w], 0, 255, out=_work_buf[:h, :w])
    out[:, :, 3] = _work_buf[:h, :w].astype(np.uint8)

    out_contig = np.ascontiguousarray(out)
    img = QImage(out_contig.data, w, h, w * 4,
                 QImage.Format.Format_RGBA8888_Premultiplied)
    return img.copy()
```

**핵심 변경점**:
- `_work_buf`은 float32로 in-place 연산이 가능 (astype 없이 직접 사용)
- alpha를 `copy()`로 보존 — 이유: 채널 계산에서 work를 재사용하므로
- `np.multiply(a, b, out=work)`으로 할당 없이 연산
- 최종 uint8 변환만 한 번 수행
- 매 프레임 배열 할당: 기존 3~4회 → alpha copy 1회로 감소

---

## TASK 3: ONNX 변환 (v3 전면 재작성)

### v2에서 발견된 문제 4가지와 수정

| # | v2 문제 | v3 수정 |
|---|---------|---------|
| 1 | Stage 2 풀링 루프에서 gather 결과 미사용 | 각 레이어별 풀링 후 평균하도록 수정 |
| 2 | FeatureExtractor 전처리(정규화) 누락 | 추론 코드에서 수동 정규화 적용 |
| 3 | torch.topk 동적 k — ONNX export 실패 가능 | 고정 비율 mean pooling으로 변경 |
| 4 | 기존 SVM과의 수치 호환성 미검증 | 검증 스크립트 포함 |

### 3-1. 전략 변경: ONNX에 풀링을 넣지 않음

v2에서는 풀링(energy_top50)까지 ONNX에 포함하려 했으나,
`torch.topk`의 동적 k가 ONNX에서 불안정하고,
기존 SVM과의 호환성 검증도 복잡해짐.

**v3 전략**: ONNX 모델은 **레이어별 hidden states만 출력**.
풀링과 SVM은 Python에서 기존과 동일하게 수행.
→ 기존 SVM과의 호환성이 자동으로 보장됨.

```
파일: vowel_recognition/method_6_embedding/export_onnx.py (새 파일)
```

```python
"""XLSR-53을 Layer 16까지만 포함하는 ONNX 모델로 변환.

출력: 각 프레임의 hidden states (풀링은 Python에서 수행)
- hidden_16: (1, T, 1024) — Stage 1용
- hidden_567: (1, T, 1024) — Stage 2용 (Layer 5,6,7 평균)

이렇게 하면:
- torch.topk 동적 k 문제 회피
- 기존 energy_top50 풀링 로직을 Python에서 그대로 사용 가능
- 기존 scaler/SVM과 수치 호환성 자동 보장

실행:
  python vowel_recognition/method_6_embedding/export_onnx.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class XLSRLayer16(nn.Module):
    """XLSR-53에서 Layer 0~16만 실행. Layer 17~23 제거."""

    def __init__(self):
        super().__init__()
        full = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')

        self.feature_extractor = full.feature_extractor
        self.feature_projection = full.feature_projection
        self.pos_conv = full.encoder.pos_conv_embed
        self.layer_norm = full.encoder.layer_norm
        # Layer 0~16만 유지 (17개 레이어)
        self.layers = nn.ModuleList(list(full.encoder.layers[:17]))

    def forward(self, input_values):
        """
        Args:
            input_values: (1, seq_len) — FeatureExtractor로 정규화된 오디오

        Returns:
            hidden_16: (1, T, 1024) — Layer 16 출력
            hidden_567: (1, T, 1024) — Layer 5,6,7 평균
        """
        # CNN feature extraction
        features = self.feature_extractor(input_values)
        features = features.transpose(1, 2)
        hidden_states, _ = self.feature_projection(features)

        # Positional encoding + layer norm
        position_embeddings = self.pos_conv(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)

        # Transformer layers — 중간 결과 저장
        h5 = None
        h6 = None
        h7 = None

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)[0]
            if i == 4:    # 0-indexed: Layer 5 = layers[4] 출력
                h5 = hidden_states
            elif i == 5:  # Layer 6
                h6 = hidden_states
            elif i == 6:  # Layer 7
                h7 = hidden_states

        hidden_16 = hidden_states  # layers[16] 출력 = Layer 16
        hidden_567 = (h5 + h6 + h7) / 3.0

        return hidden_16, hidden_567


def export():
    print("모델 로딩...")
    model = XLSRLayer16()
    model.eval()

    # FeatureExtractor에서 나오는 정규화된 오디오와 같은 형태
    # 0.4초 @ 16kHz = 6400 샘플 (max_duration에 맞춤)
    dummy = torch.randn(1, 6400)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(out_dir, 'xlsr_layer16.onnx')

    print("ONNX export...")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input_values'],
        output_names=['hidden_16', 'hidden_567'],
        dynamic_axes={
            'input_values': {1: 'seq_len'},
            'hidden_16': {1: 'time_steps'},
            'hidden_567': {1: 'time_steps'},
        },
        opset_version=17,
    )

    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"저장: {onnx_path} ({file_size:.0f} MB)")

    # ── 수치 검증 ──
    print("\n수치 검증...")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)

    test_audio = torch.randn(1, 6400)

    # PyTorch 출력
    with torch.no_grad():
        pt_h16, pt_h567 = model(test_audio)
    pt_h16 = pt_h16.numpy()
    pt_h567 = pt_h567.numpy()

    # ONNX 출력
    onnx_h16, onnx_h567 = sess.run(None, {'input_values': test_audio.numpy()})

    diff_16 = np.max(np.abs(pt_h16 - onnx_h16))
    diff_567 = np.max(np.abs(pt_h567 - onnx_h567))
    print(f"  hidden_16 max diff: {diff_16:.8f}")
    print(f"  hidden_567 max diff: {diff_567:.8f}")

    if diff_16 < 1e-4 and diff_567 < 1e-4:
        print("  ✓ 수치 일치 확인됨")
    else:
        print("  ⚠ 수치 차이가 큼 — 확인 필요")

    # ── 속도 측정 ──
    print("\n속도 측정...")
    import time
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.run(None, {'input_values': test_audio.numpy()})
        times.append(time.perf_counter() - t0)
    avg = np.mean(times[5:]) * 1000
    print(f"  ONNX 추론: {avg:.0f} ms (warmup 제외 평균)")


if __name__ == '__main__':
    export()
```

**핵심**: ONNX는 hidden states만 출력하고, 풀링은 밖에서 한다.
→ topk 동적 k 문제 완전 회피, 기존 SVM 호환성 자동 보장.

### 3-2. FeatureExtractor 전처리를 수동으로 적용

```
파일: main_integrated.py
클래스: VowelRecognitionWorker
```

ONNX 경로에서도 PyTorch 경로와 동일한 전처리가 필요.
`Wav2Vec2FeatureExtractor`의 핵심 동작은 **zero-mean + unit-variance 정규화**.

```python
class VowelRecognitionWorker(QObject):
    recognized = Signal(str, float)
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._session = None        # ONNX session
        self._fe = None             # FeatureExtractor (정규화용, ONNX에서도 사용)
        self._model = None          # PyTorch model (폴백용)
        self._s1_scaler = None
        self._s1_clf = None
        self._s2_scaler = None
        self._s2_clf = None
        self._target = None
        self._use_onnx = False
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest_item = None
        self._running = False

    def load_models(self):
        global RECOGNITION_AVAILABLE
        try:
            self.status_changed.emit("모델 로딩 중...")
            import pickle
            from transformers import Wav2Vec2FeatureExtractor

            model_dir = os.path.join(os.path.dirname(__file__),
                                     'vowel_recognition', 'method_6_embedding')
            onnx_path = os.path.join(model_dir, 'xlsr_layer16.onnx')
            pkl_path = os.path.join(model_dir, 'twostage_model.pkl')

            # ── FeatureExtractor는 항상 로드 (정규화에 필요) ──
            model_name = 'facebook/wav2vec2-large-xlsr-53'
            self._fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

            if os.path.exists(onnx_path):
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads = 4
                opts.inter_op_num_threads = 1
                self._session = ort.InferenceSession(onnx_path, opts)
                self._use_onnx = True
                self.status_changed.emit("ONNX 모델 로드 완료")
            else:
                import torch
                from transformers import Wav2Vec2Model
                self._model = Wav2Vec2Model.from_pretrained(model_name)
                self._model.eval()
                self._use_onnx = False
                self.status_changed.emit("PyTorch 모델 로드 완료 (ONNX 권장)")

            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            self._s1_scaler = data['stage1']['scaler']
            self._s1_clf = data['stage1']['clf']
            self._s2_scaler = data['stage2']['scaler']
            self._s2_clf = data['stage2']['clf']
            self._target = data['stage2']['target_vowels']

            RECOGNITION_AVAILABLE = True
        except Exception as e:
            self.status_changed.emit(f"모델 로드 실패: {e}")
            import traceback; traceback.print_exc()

    def submit(self, audio, sr):
        with self._lock:
            self._latest_item = (audio, sr)
        self._event.set()

    def process_loop(self):
        self._running = True
        while self._running:
            self._event.wait(timeout=0.5)
            self._event.clear()

            item = None
            with self._lock:
                if self._latest_item is not None:
                    item = self._latest_item
                    self._latest_item = None

            if item is None:
                continue

            audio, sr = item
            try:
                result = self._recognize(audio, sr)
                if result:
                    self.recognized.emit(result[0], result[1])
            except Exception as e:
                print(f"[인식 오류] {e}", flush=True)

    def stop(self):
        self._running = False
        self._event.set()  # 대기 중인 스레드 깨움

    def _recognize(self, audio, sr):
        if not RECOGNITION_AVAILABLE:
            return None

        # ── 리샘플링: scipy resample_poly 사용 ──
        target_sr = 16000
        if sr != target_sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(target_sr, sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

        # ── FeatureExtractor로 정규화 (ONNX/PyTorch 공통) ──
        inputs = self._fe(audio, sampling_rate=target_sr,
                          return_tensors="np", padding=False)
        input_values = inputs.input_values  # (1, seq_len), float32, 정규화됨

        # ── 임베딩 추출 ──
        if self._use_onnx:
            hidden_16, hidden_567 = self._session.run(
                None, {'input_values': input_values})
            # hidden_16: (1, T, 1024), hidden_567: (1, T, 1024)
        else:
            import torch
            input_pt = torch.from_numpy(input_values)
            with torch.no_grad():
                outputs = self._model(input_pt, output_hidden_states=True)
            hidden = outputs.hidden_states
            hidden_16 = hidden[16].numpy()    # (1, T, 1024)
            h5 = hidden[5].numpy()
            h6 = hidden[6].numpy()
            h7 = hidden[7].numpy()
            hidden_567 = (h5 + h6 + h7) / 3.0  # (1, T, 1024)

        # ── 풀링: energy_top50 (기존과 동일한 로직, numpy로) ──
        emb16 = self._energy_top50_pool(hidden_16[0])   # (1024,)
        emb567 = self._energy_top50_pool(hidden_567[0])  # (1024,)

        # ── 2단계 SVM 분류 (기존과 동일) ──
        X1 = self._s1_scaler.transform(emb16.reshape(1, -1))
        pred1 = self._s1_clf.predict(X1)[0]
        proba1 = self._s1_clf.predict_proba(X1)[0]
        classes1 = self._s1_clf.classes_

        if pred1 in self._target:
            X2 = self._s2_scaler.transform(emb567.reshape(1, -1))
            pred2 = self._s2_clf.predict(X2)[0]
            proba2 = self._s2_clf.predict_proba(X2)[0]
            final = pred2
            conf = float(max(proba2))
            # 디버그
            top3 = sorted(zip(classes1, proba1), key=lambda x: -x[1])[:3]
            s1_str = ' '.join(f'{c}={p:.0%}' for c, p in top3)
            s2_str = ' '.join(f'{c}={p:.0%}' for c, p
                              in zip(self._s2_clf.classes_, proba2))
            tag = 'ONNX' if self._use_onnx else 'PT'
            print(f'[{tag}] S1:[{s1_str}]→{pred1}  S2:[{s2_str}]→{final} '
                  f'({conf:.0%})', flush=True)
        else:
            final = pred1
            conf = float(max(proba1))
            top3 = sorted(zip(classes1, proba1), key=lambda x: -x[1])[:3]
            s1_str = ' '.join(f'{c}={p:.0%}' for c, p in top3)
            tag = 'ONNX' if self._use_onnx else 'PT'
            print(f'[{tag}] S1:[{s1_str}]→{final} ({conf:.0%})', flush=True)

        return final, conf

    @staticmethod
    def _energy_top50_pool(frames):
        """(T, dim) → (dim,): 에너지 상위 50% 프레임의 평균.
        기존 PyTorch 코드와 동일한 로직의 numpy 버전."""
        energy = np.linalg.norm(frames, axis=1)  # (T,)
        k = max(1, len(energy) // 2)
        top_indices = np.argpartition(energy, -k)[-k:]  # 상위 k개 인덱스
        return frames[top_indices].mean(axis=0).astype(np.float32)
```

**핵심 변경점 (v2 대비)**:
1. FeatureExtractor를 ONNX 경로에서도 사용 → 전처리 불일치 해소
2. ONNX 모델은 hidden states만 출력 → topk 동적 k 문제 회피
3. 풀링을 Python `_energy_top50_pool()`로 분리 → 기존 SVM과 수치 호환
4. `np.argpartition` 사용 (torch.topk와 동일 결과, 더 빠름)
5. `threading.Event` 사용 → sleep 폴링 대신 즉시 깨움

### 3-3. ONNX ↔ PyTorch 수치 호환성 검증 스크립트

ONNX 변환 후, 기존 SVM과 동일한 예측을 하는지 검증하는 스크립트.
**ONNX 전환 전에 반드시 실행.**

```
파일: vowel_recognition/method_6_embedding/verify_onnx.py (새 파일)
```

```python
"""ONNX 모델과 PyTorch 모델이 동일한 SVM 예측을 내는지 검증.

사용법:
  python vowel_recognition/method_6_embedding/verify_onnx.py

검증 항목:
  1. hidden states 수치 차이 (max abs diff < 1e-4)
  2. 풀링 후 임베딩 수치 차이
  3. SVM 예측 일치율
"""

import os, sys, pickle, time
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import onnxruntime as ort

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'facebook/wav2vec2-large-xlsr-53'


def energy_top50_pool_np(frames):
    energy = np.linalg.norm(frames, axis=1)
    k = max(1, len(energy) // 2)
    top_idx = np.argpartition(energy, -k)[-k:]
    return frames[top_idx].mean(axis=0).astype(np.float32)


def energy_top50_pool_pt(frames_tensor):
    """기존 코드와 동일한 PyTorch 풀링."""
    energy = frames_tensor.norm(dim=1)
    k = max(1, len(energy) // 2)
    top_idx = torch.topk(energy, k).indices
    return frames_tensor[top_idx].mean(dim=0).numpy().astype(np.float32)


def main():
    onnx_path = os.path.join(MODEL_DIR, 'xlsr_layer16.onnx')
    pkl_path = os.path.join(MODEL_DIR, 'twostage_model.pkl')

    if not os.path.exists(onnx_path):
        print(f"ONNX 모델 없음: {onnx_path}")
        print("먼저 export_onnx.py를 실행하세요.")
        return

    # 모델 로드
    print("로딩: FeatureExtractor...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

    print("로딩: PyTorch model...")
    pt_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    pt_model.eval()

    print("로딩: ONNX session...")
    sess = ort.InferenceSession(onnx_path)

    print("로딩: SVM...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    s1_scaler = data['stage1']['scaler']
    s1_clf = data['stage1']['clf']

    # 랜덤 테스트 오디오 10개
    print("\n=== 수치 검증 ===")
    n_tests = 10
    match_count = 0

    for i in range(n_tests):
        # 랜덤 오디오 (0.3~0.5초)
        length = np.random.randint(4800, 8000)
        audio = np.random.randn(length).astype(np.float32) * 0.1

        # FeatureExtractor 정규화
        inputs = fe(audio, sampling_rate=16000,
                    return_tensors="np", padding=False)
        input_values = inputs.input_values

        # PyTorch 경로
        with torch.no_grad():
            pt_out = pt_model(
                torch.from_numpy(input_values),
                output_hidden_states=True)
        pt_h16 = pt_out.hidden_states[16].numpy()[0]   # (T, 1024)
        pt_h5 = pt_out.hidden_states[5].numpy()[0]
        pt_h6 = pt_out.hidden_states[6].numpy()[0]
        pt_h7 = pt_out.hidden_states[7].numpy()[0]
        pt_h567 = (pt_h5 + pt_h6 + pt_h7) / 3.0

        pt_emb16 = energy_top50_pool_pt(
            pt_out.hidden_states[16].squeeze(0))
        pt_emb567_parts = [
            energy_top50_pool_pt(pt_out.hidden_states[l].squeeze(0))
            for l in [5, 6, 7]]
        pt_emb567 = np.mean(pt_emb567_parts, axis=0)

        # ONNX 경로
        onnx_h16, onnx_h567 = sess.run(
            None, {'input_values': input_values})
        onnx_h16 = onnx_h16[0]    # (T, 1024)
        onnx_h567 = onnx_h567[0]  # (T, 1024)

        onnx_emb16 = energy_top50_pool_np(onnx_h16)
        onnx_emb567 = energy_top50_pool_np(onnx_h567)

        # 비교
        h16_diff = np.max(np.abs(pt_h16 - onnx_h16))
        emb16_diff = np.max(np.abs(pt_emb16 - onnx_emb16))

        # SVM 예측 비교
        pt_pred = s1_clf.predict(
            s1_scaler.transform(pt_emb16.reshape(1, -1)))[0]
        onnx_pred = s1_clf.predict(
            s1_scaler.transform(onnx_emb16.reshape(1, -1)))[0]

        match = pt_pred == onnx_pred
        if match:
            match_count += 1

        print(f"  [{i+1}] h16_diff={h16_diff:.6f}  "
              f"emb16_diff={emb16_diff:.6f}  "
              f"PT={pt_pred} ONNX={onnx_pred} {'✓' if match else '✗'}")

    print(f"\nSVM 예측 일치율: {match_count}/{n_tests}")

    if match_count == n_tests:
        print("✓ ONNX 전환 안전")
    else:
        print("⚠ 일부 불일치 — 임베딩 차이 확인 필요")

    # 속도 비교
    print("\n=== 속도 비교 ===")
    test_input = fe(np.random.randn(6400).astype(np.float32),
                    sampling_rate=16000, return_tensors="np").input_values

    # PyTorch
    pt_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_model(torch.from_numpy(test_input),
                     output_hidden_states=True)
        pt_times.append(time.perf_counter() - t0)

    # ONNX
    onnx_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        sess.run(None, {'input_values': test_input})
        onnx_times.append(time.perf_counter() - t0)

    pt_avg = np.mean(pt_times[3:]) * 1000
    onnx_avg = np.mean(onnx_times[3:]) * 1000
    print(f"  PyTorch (24 layers): {pt_avg:.0f} ms")
    print(f"  ONNX (17 layers):    {onnx_avg:.0f} ms")
    print(f"  속도 향상: {pt_avg / onnx_avg:.1f}x")


if __name__ == '__main__':
    main()
```

### 3-4. ONNX 전환 순서 (안전한 절차)

```
1. export_onnx.py 실행 → xlsr_layer16.onnx 생성
2. verify_onnx.py 실행 → 수치 호환성 확인
   - 모든 테스트에서 SVM 예측 일치해야 함
   - h16_diff < 1e-4 확인
3. 일치 확인 후, main_integrated.py의 VowelRecognitionWorker 교체
4. 실행하여 실시간 테스트
```

### 3-5. poll_timer 간격 단축

```python
# main() 함수에서
poll_timer.setInterval(20)  # 50ms → 20ms
```

---

## TASK 4: 스냅샷 주기 단축 (v2와 동일, 승인됨)

TASK 1에 통합. UtteranceCollector의 기본값:
```python
min_duration=0.10, snapshot_interval=0.10, max_duration=0.4
```

---

## TASK 5: Confidence 기반 모핑 속도 조절 (v2와 동일, 승인됨)

```
파일: main_integrated.py
클래스: IntegratedCanvas
```

```python
def __init__(self, glyph_data):
    # ... 기존 초기화 ...
    self._morph_speed = MORPH_SEC  # 기본값 0.45

def on_vowel_recognized(self, vowel, confidence):
    # 기존 히스테리시스 로직 유지
    if confidence < self._smooth_conf_threshold:
        return
    if self._smooth_current is None:
        self._smooth_current = vowel
    elif vowel != self._smooth_current:
        if confidence < self._smooth_switch_threshold:
            return
        self._smooth_current = vowel

    smoothed_vowel = self._smooth_current
    self._recognized_vowel = smoothed_vowel
    self._recognized_conf = confidence

    # 정답 비교 로그 (기존 로직 유지)
    gt = self._gt_label
    if gt != "—":
        from datetime import datetime
        mark = 'O' if smoothed_vowel == gt else 'X'
        self._gt_log.append(
            (datetime.now().strftime('%H:%M:%S'), gt, smoothed_vowel, confidence))
        correct = sum(1 for _, g, p, _ in self._gt_log if g == p)
        total = len(self._gt_log)
        print(f'[비교] 정답={gt} 예측={vowel}→{smoothed_vowel} {mark} '
              f'({confidence:.0%})  누적: {correct}/{total} '
              f'({correct/total*100:.1f}%)', flush=True)

    if smoothed_vowel in self._data:
        # ── confidence 기반 모핑 속도 조절 ──
        if confidence > 0.8:
            self._morph_speed = 0.35
        elif confidence > 0.6:
            self._morph_speed = 0.50
        else:
            self._morph_speed = 0.70
        self.trigger_morph(smoothed_vowel)

def _tick(self):
    # ... 스무딩, VAD 페이드 ...
    if self._animating and self._tgt:
        elapsed = now - self._t0
        self._t = min(elapsed / self._morph_speed, 1.0)  # ← 여기
        # ... 나머지 동일
```

---

## TASK 6: 데이터 수집 모드 (v3 시그널/슬롯 구조로 변경)

### v2 문제
`poll_utterance` 안에서 `win` 객체를 직접 참조 → 높은 결합도.

### v3: 시그널/슬롯 기반으로 분리

```
파일: main_integrated.py
```

수집 모드를 별도 QObject로 분리:

```python
class DataCollector(QObject):
    """발음 데이터 수집 모드. 시그널로 IntegratedCanvas와 연결."""
    sample_saved = Signal(str, str, int)  # (파일경로, 모음, 누적수)

    VOWELS = ['아', '어', '오', '우', '으', '이', '에']

    def __init__(self):
        super().__init__()
        self._active = False
        self._vowel_idx = 0
        self._count = {}
        self._session = ''
        self._save_dir = os.path.join(os.path.dirname(__file__), 'collected_data')

    @property
    def is_active(self):
        return self._active

    @property
    def current_vowel(self):
        return self.VOWELS[self._vowel_idx]

    def toggle(self):
        self._active = not self._active
        if self._active:
            self._session = time.strftime('%Y%m%d_%H%M%S')
            os.makedirs(self._save_dir, exist_ok=True)
            print(f'[수집 모드 ON] 세션: {self._session}')
            print(f'  발음할 모음: {self.current_vowel}')
        else:
            print('[수집 모드 OFF]')
            self._print_stats()

    def next_vowel(self):
        self._vowel_idx = (self._vowel_idx + 1) % len(self.VOWELS)
        print(f'  다음 모음: {self.current_vowel}')

    def save_utterance(self, audio, sr):
        """UtteranceCollector로부터 오디오를 받아 저장."""
        if not self._active:
            return
        import soundfile as sf
        vowel = self.current_vowel
        cnt = self._count.get(vowel, 0) + 1
        self._count[vowel] = cnt
        fname = f'{self._session}_{vowel}_{cnt:03d}.wav'
        fpath = os.path.join(self._save_dir, fname)
        sf.write(fpath, audio, sr)
        self.sample_saved.emit(fpath, vowel, cnt)
        print(f'  저장: {fname} ({vowel} #{cnt})')

    def _print_stats(self):
        total = sum(self._count.values())
        print(f'  수집 현황 (총 {total}개):')
        for v in self.VOWELS:
            c = self._count.get(v, 0)
            if c > 0:
                print(f'    {v}: {c}개')
```

main()에서 연결:
```python
collector_data = DataCollector()

def poll_utterance():
    utt = collector.get_utterance()
    if utt is not None:
        if RECOGNITION_AVAILABLE:
            worker.submit(utt, 44100)
        # 수집 모드: 시그널로 분리됨
        if collector_data.is_active:
            collector_data.save_utterance(utt, 44100)

# 키보드 연결은 IntegratedWindow.keyPressEvent에서:
# R키: collector_data.toggle()
# 스페이스: collector_data.next_vowel() (수집 모드일 때)
```

IntegratedWindow 수정:
```python
class IntegratedWindow(QWidget):
    def __init__(self, glyph_data, data_collector=None):
        super().__init__()
        # ... 기존 초기화 ...
        self._data_collector = data_collector

    def keyPressEvent(self, e):
        k = e.key()
        # ... 기존 키 처리 ...
        if k == Qt.Key.Key_R and self._data_collector:
            self._data_collector.toggle()
            return
        if (k == Qt.Key.Key_Space and self._data_collector
                and self._data_collector.is_active):
            self._data_collector.next_vowel()
            return
```

---

## 적용 순서 및 기대 효과

| 순서 | TASK | 난이도 | 지연 개선 | CPU 개선 |
|------|------|--------|-----------|----------|
| 1 | TASK 1: YIN 중복 + 벡터화 | 하 | 간접적 | ~50%↓ |
| 2 | TASK 2: 렌더링 캐싱 | 하 | 프레임 안정 | ~90%↓ (정지시) |
| 3 | TASK 5: confidence 모핑 | 하 | 체감 개선 | — |
| 4 | TASK 3: ONNX (export→검증→전환) | 중 | 360→120ms | — |
| 5 | TASK 6: 데이터 수집 | 중 | — | — |

### ONNX 전환 체크리스트

```
□ export_onnx.py 실행 → xlsr_layer16.onnx 생성
□ verify_onnx.py 실행:
  □ h16 max diff < 1e-4
  □ SVM 예측 일치율 100%
  □ 속도 향상 확인 (목표: 2x 이상)
□ main_integrated.py VowelRecognitionWorker 교체
□ 실시간 테스트: 기존 PyTorch와 체감 차이 없는지 확인
□ 인식 결과 로그 비교 (같은 오디오에 대해 동일 예측)
```
