# VoiceTypo 최적화 작업 지시서 v2

> 실제 소스코드 분석 기반. Claude Code에 PROJECT_FULL_SOURCE.txt와 함께 전달하세요.
> 각 TASK를 순서대로 진행. 하나 완료 후 다음으로.

---

## 코드 분석에서 발견한 문제점

### 🔴 심각한 비효율

1. **YIN 피치 감지가 2번 중복 실행됨**
   - `AudioBridge._detector` (YinDetector) — 매 프레임
   - `UtteranceCollector._yin` (YinDetector) — 매 프레임, 같은 오디오
   - `UtteranceCollector._vad` (VoiceActivityDetector) — AudioBridge._vad와 중복
   - **YIN 차분 함수가 Python for-loop** (yin.py:1161 `for tau in range(1, tau_max)`)
     → 44.1kHz, blocksize=2048일 때 tau_max=551, 매 프레임 Python for 551회 반복

2. **XLSR-53 전체 24레이어 forward pass**
   - `output_hidden_states=True`로 24개 레이어 전부 계산
   - 실제 사용하는 것은 Layer 5, 6, 7, 16뿐
   - Layer 17~23 (약 30%의 연산)이 완전히 낭비

3. **sdf_to_rgba()가 매 프레임 호출됨 (60fps × 512×512)**
   - 매 프레임마다 512×512×4 numpy 배열 생성 + QImage 생성 + .copy()
   - `paintEvent()` → `apply_vibrato_to_sdf()` (sin 4회 × 512×512)
                    → `sdf_to_rgba()` (512×512×4 할당)
                    → `QImage` 생성 + `.copy()`
   - 비브라토 없고 모핑도 없을 때도 매번 재계산

4. **리샘플링이 nearest-neighbor** (main_integrated.py:198-203)
   - 44.1kHz → 16kHz 변환 시 `indices.astype(int)` 단순 인덱싱
   - 에일리어싱 발생 → 모음 인식 정확도에 악영향

5. **worker 폴링이 sleep 기반** (main_integrated.py:168-175)
   - `time.sleep(0.02)` 루프 → 최대 20ms 추가 지연
   - `poll_timer.setInterval(50)` → 최대 50ms 추가 지연
   - 합산: 최대 70ms의 불필요한 폴링 지연

### 🟡 구조적 문제

6. **UtteranceCollector 스냅샷 주기 0.15초**
   - snapshot_interval=0.15, max_duration=0.5
   - 즉, 최초 인식까지 최소 150ms(수집) + 50ms(폴링) + 360ms(추론) = 560ms
   - 발성 시작 → 글리프 반응까지 체감 0.5~0.7초

7. **모핑 중 _blend_glyphs가 매 프레임 _warp_sdf를 여러 번 호출**
   - _warp_sdf: 512×512 좌표 연산 + 인덱싱
   - 매칭된 획 수 × 2(src+dst) + 비매칭 획들
   - 모핑 0.45초 동안 약 27프레임, 각각 무거운 연산

### 🟢 참고: 이미 시도했으나 효과 없었던 것

- **포먼트(F1/F2) 피처 추가**: features.py에 이미 구현되어 있고, "+0.5%"로 효과 없었음
  → 이전 지시서의 TASK 4(포먼트 보조 피처)는 제외
- **중앙 안정구간 사용**: -5.4% 역효과
- **L0 임베딩**: 남성만 개선
- **H1-H2 피처**: 효과 없음

---

## TASK 1: YIN 중복 제거 + 벡터화 (즉시 효과)

### 문제
AudioBridge와 UtteranceCollector가 각각 YinDetector, VoiceActivityDetector를 별도로 생성하여
동일한 오디오에 대해 YIN 차분 함수를 2번 계산한다.
또한 YIN의 핵심 루프가 Python for-loop이라 프레임당 수 밀리초 소요.

### 수정 1-1: UtteranceCollector에서 자체 YIN/VAD 제거

```
파일: main_integrated.py
클래스: UtteranceCollector
```

UtteranceCollector가 자체 YIN/VAD를 쓰지 않고,
AudioBridge의 VAD 결과를 받아 오디오 수집만 담당하도록 변경:

```python
class UtteranceCollector:
    """스트리밍 방식: VAD 상태를 외부에서 받아 발성 중 오디오를 수집."""
    def __init__(self, sr=44100, min_duration=0.15, snapshot_interval=0.15,
                 max_duration=0.5):
        self.sr = sr
        self.min_samples = int(min_duration * sr)
        self.snapshot_interval = snapshot_interval
        self.max_samples = int(max_duration * sr)

        # YIN, VAD 자체 인스턴스 제거 — 외부(AudioBridge)에서 VAD 상태를 받음
        self._buffer = []
        self._collecting = False
        self._ready = False
        self._utterance = None
        self._last_snapshot = 0.0
        self._lock = threading.Lock()
        self._vad_active = False  # 외부에서 설정

    def set_vad_state(self, is_active: bool):
        """AudioBridge에서 VAD 결과를 전달받는다."""
        self._vad_active = is_active

    def on_audio(self, chunk, sr):
        # 자체 YIN/VAD 호출 삭제 — self._vad_active 사용
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
    # poll(), reset()은 그대로
```

main() 함수에서 연결:
```python
# AudioBridge.updated 시그널에서 VAD 상태를 collector에 전달
def on_bridge_update(freq, rms, vib_rate, vib_extent, vad_active):
    collector.set_vad_state(vad_active)

bridge.updated.connect(on_bridge_update)
```

### 수정 1-2: YIN 차분 함수 벡터화

```
파일: pitch_detection/yin.py
함수: detect()
```

현재 Python for-loop:
```python
# 현재 코드 (느림):
d = np.zeros(tau_max, dtype=np.float32)
for tau in range(1, tau_max):
    diff = audio[:n - tau] - audio[tau:n]
    d[tau] = np.sum(diff ** 2)
```

numpy 벡터화된 차분 함수로 교체:
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

    # ── 벡터화된 차분 함수 (FFT 기반 자기상관) ──
    # d(tau) = sum_j (x[j] - x[j+tau])^2
    #        = 2 * (sum x^2 의 windowed sum - autocorrelation)
    x = audio[:tau_max * 2].astype(np.float64)
    # zero-padded FFT로 자기상관 계산
    fft_size = 1
    while fft_size < len(x) * 2:
        fft_size *= 2
    X = np.fft.rfft(x, n=fft_size)
    acf_full = np.fft.irfft(X * np.conj(X))[:tau_max]

    # cumulative sum of x^2
    x_sq = x * x
    cum = np.cumsum(x_sq)

    d = np.zeros(tau_max, dtype=np.float64)
    # d(tau) = cum[n-1] - cum[tau-1] + cum[n-1] - cum[n-tau-1] - 2*acf(tau)
    # 간소화: rolling sum 방식
    m = len(x)
    for tau in range(1, tau_max):
        # 정확한 차분: sum_{j=0}^{m-tau-1} (x[j] - x[j+tau])^2
        d[tau] = cum[m - tau - 1] + (cum[m - 1] - cum[tau - 1]) - 2.0 * acf_full[tau]

    # CMNDF (벡터화)
    cmndf = np.ones(tau_max, dtype=np.float64)
    cum_d = np.cumsum(d)
    taus = np.arange(1, tau_max, dtype=np.float64)
    valid = cum_d[1:tau_max] > 0
    cmndf[1:tau_max] = np.where(valid, d[1:tau_max] * taus / cum_d[1:tau_max], 1.0)

    # 임계값 이하 첫 극소점 (이 부분은 짧아서 for-loop OK)
    tau_est = 0
    for tau in range(tau_min, tau_max - 1):
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

**기대 효과**: YIN 프레임당 처리 시간 ~3ms → ~0.3ms, 중복 제거로 총 CPU 부하 50%↓

---

## TASK 2: 렌더링 파이프라인 최적화 (프레임 드롭 해결)

### 문제
paintEvent()가 매 프레임(60fps) 비브라토가 없어도 512×512 sin 연산 4회 + RGBA 배열 생성 + QImage 복사를 수행.

### 수정 2-1: SDF 변경 시에만 재계산 (더티 플래그)

```
파일: main_integrated.py
클래스: IntegratedCanvas
```

```python
class IntegratedCanvas(QWidget):
    def __init__(self, glyph_data):
        # ... 기존 초기화 ...

        # ── 캐싱 추가 ──
        self._cached_qimg = None       # 캐싱된 QImage
        self._cache_dirty = True       # SDF 변경 시 True
        self._prev_pitch_ratio = None  # 색상 변경 감지용
        self._prev_opacity = None      # 투명도 변경 감지용
        self._prev_vib_amount = 0.0    # 비브라토 변경 감지용
        self._prev_vib_phase = 0.0

    def _tick(self):
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        # 스무딩 (기존과 동일)
        self._pitch_ratio += SCALE_SMOOTH * (self._raw_pitch - self._pitch_ratio)
        self._volume_scale += VOLUME_SMOOTH * (self._raw_volume - self._volume_scale)
        self._vib_amount += VIBRATO_SMOOTH * (self._raw_vib_amount - self._vib_amount)
        self._vib_speed += VIBRATO_SMOOTH * (self._raw_vib_speed - self._vib_speed)

        if self._vad_active:
            self._opacity = min(self._opacity + dt * FADE_IN_SPEED, 1.0)
        else:
            self._opacity = max(self._opacity - dt * FADE_OUT_SPEED, 0.0)

        self._vibrato_phase += dt * self._vib_speed * 3.0

        # 모핑 애니메이션
        if self._animating and self._tgt:
            elapsed = now - self._t0
            self._t = min(elapsed / MORPH_SEC, 1.0)
            self._sdf = _blend_glyphs(self._data[self._cur],
                                       self._data[self._tgt], self._t)
            s = self._t
            self._joint_dist = ((1 - s) * self._joint_dists[self._cur]
                                + s * self._joint_dists[self._tgt])
            if self._t >= 1.0:
                self._animating = False
                self._cur = self._tgt
                self._tgt = None
                self._sdf = _resting_sdf(self._data[self._cur])
                self._joint_dist = self._joint_dists[self._cur]
            self._cache_dirty = True  # 모핑 중 → 항상 재계산

        # ── 캐시 무효화 조건 판단 ──
        pitch_changed = (self._prev_pitch_ratio is None or
                        abs(self._pitch_ratio - self._prev_pitch_ratio) > 0.005)
        opacity_changed = (self._prev_opacity is None or
                          abs(self._opacity - self._prev_opacity) > 0.005)
        vib_active = self._vib_amount > 0.5

        if pitch_changed or opacity_changed or vib_active:
            self._cache_dirty = True

        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(0, 0, 0))

        # ── 캐시된 QImage 사용 ──
        if self._cache_dirty or self._cached_qimg is None:
            if self._vib_amount > 0.5:
                sdf = apply_vibrato_to_sdf(self._sdf, self._vib_amount,
                                            self._vibrato_phase, self._joint_dist)
            else:
                sdf = self._sdf

            c_top, c_bot, glow_color = compute_colors(self._pitch_ratio)
            self._cached_qimg = sdf_to_rgba(sdf, c_top, c_bot, self._opacity)
            self._cached_glow_color = glow_color
            self._prev_pitch_ratio = self._pitch_ratio
            self._prev_opacity = self._opacity
            self._cache_dirty = False

        qimg = self._cached_qimg
        glow = self._cached_glow_color

        # 이후 스케일/글로우/HUD는 기존과 동일 ...
        # (drawImage에 qimg 사용)
```

### 수정 2-2: sdf_to_rgba 자체 최적화

```
파일: text_effects/test_effects.py
함수: sdf_to_rgba()
```

현재: 매번 512×512×4 배열을 새로 할당.
개선: 사전할당된 버퍼 재사용.

```python
# 모듈 레벨에 버퍼 사전 할당
_rgba_buffer = np.zeros((GRID, GRID, 4), dtype=np.uint8)
_alpha_buffer = np.zeros((GRID, GRID), dtype=np.float32)
_t_gradient = np.linspace(0, 1, GRID, dtype=np.float32)[:, None]  # 한 번만 계산

def sdf_to_rgba(sdf, color_top, color_bot, opacity=1.0):
    h, w = sdf.shape
    # 사전할당 버퍼 사용 (배열 생성 비용 제거)
    np.clip(0.5 - sdf / AA_WIDTH, 0.0, 1.0, out=_alpha_buffer[:h, :w])
    alpha = _alpha_buffer[:h, :w]
    if opacity < 1.0:
        alpha = alpha * opacity  # 이 경우만 곱셈

    t = _t_gradient[:h]
    buf = _rgba_buffer[:h, :w]

    # in-place 연산으로 할당 최소화
    for i in range(3):
        np.multiply(color_top[i], 1.0 - t, out=buf[:, :, i].astype(np.float32))
        # 단순화: 직접 계산
        channel = (color_top[i] * (1 - t) + color_bot[i] * t) * alpha
        np.clip(channel, 0, 255, out=channel)
        buf[:, :, i] = channel.astype(np.uint8)

    np.clip(alpha * 255, 0, 255, out=_alpha_buffer[:h, :w])
    buf[:, :, 3] = _alpha_buffer[:h, :w].astype(np.uint8)

    buf_contig = np.ascontiguousarray(buf)
    img = QImage(buf_contig.data, w, h, w * 4,
                 QImage.Format.Format_RGBA8888_Premultiplied)
    return img.copy()
```

**기대 효과**: 비브라토/모핑이 없는 정지 상태에서 paintEvent CPU 부하 ~90% 감소.
모핑 중에도 불필요한 배열 재할당 제거로 30~50% 감소.

---

## TASK 3: XLSR-53 추론 최적화 (지연 핵심 해결)

### 문제
- 전체 24 레이어 forward pass → Layer 16까지만 필요 (Layer 17~23 낭비)
- 리샘플링이 nearest-neighbor → 에일리어싱
- worker가 sleep(0.02) 폴링
- poll_timer가 50ms 간격

### 수정 3-1: ONNX 변환 — Layer 16까지만 추출

```
파일: vowel_recognition/method_6_embedding/export_onnx.py (새 파일)
```

```python
"""XLSR-53을 Layer 16까지만 포함하는 ONNX 모델로 변환.

Stage 1용 (Layer 16): xlsr_s1.onnx
Stage 2용 (Layer 5-7): xlsr_s2.onnx

실행:
  python vowel_recognition/method_6_embedding/export_onnx.py
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Model

class XLSRTruncatedS1(nn.Module):
    """Stage 1용: Layer 16 출력만. Layer 17~23 제거."""
    def __init__(self):
        super().__init__()
        full = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        self.feature_extractor = full.feature_extractor
        self.feature_projection = full.feature_projection
        # Layer 0~16만 유지 (17~23 삭제)
        self.pos_conv = full.encoder.pos_conv_embed
        self.layer_norm = full.encoder.layer_norm
        self.layers = nn.ModuleList(list(full.encoder.layers[:17]))  # 0~16

    def forward(self, input_values):
        # CNN feature extraction
        features = self.feature_extractor(input_values)
        features = features.transpose(1, 2)
        hidden_states, _ = self.feature_projection(features)

        # Positional encoding + layer norm
        position_embeddings = self.pos_conv(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)

        # Transformer layers 0~16
        all_hidden = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
            all_hidden.append(hidden_states)

        # energy_top50 풀링 — Layer 16
        h16 = all_hidden[16]  # (1, T, 1024)
        energy = h16.norm(dim=2)  # (1, T)
        k = max(1, h16.shape[1] // 2)
        top_idx = torch.topk(energy, k, dim=1).indices  # (1, k)
        # gather
        top_idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, h16.shape[2])
        selected = torch.gather(h16, 1, top_idx_exp)  # (1, k, 1024)
        emb16 = selected.mean(dim=1)  # (1, 1024)

        # Stage 2 피처: Layer 5,6,7 평균
        h5 = all_hidden[5]
        h6 = all_hidden[6]
        h7 = all_hidden[7]
        for h in [h5, h6, h7]:
            e = h.norm(dim=2)
            k2 = max(1, h.shape[1] // 2)
            idx = torch.topk(e, k2, dim=1).indices
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, h.shape[2])
            sel = torch.gather(h, 1, idx_exp)
        # 간소화: 3개 레이어 평균 후 energy_top50
        h567 = (h5 + h6 + h7) / 3.0
        energy567 = h567.norm(dim=2)
        k3 = max(1, h567.shape[1] // 2)
        top_idx3 = torch.topk(energy567, k3, dim=1).indices
        top_idx3_exp = top_idx3.unsqueeze(-1).expand(-1, -1, h567.shape[2])
        emb567 = torch.gather(h567, 1, top_idx3_exp).mean(dim=1)

        return emb16, emb567  # (1, 1024), (1, 1024)

def export():
    model = XLSRTruncatedS1()
    model.eval()

    # 0.5초 @ 16kHz = 8000 샘플 (max_duration=0.5초에 맞춤)
    dummy = torch.randn(1, 8000)

    torch.onnx.export(
        model, dummy,
        'vowel_recognition/method_6_embedding/xlsr_twostage.onnx',
        input_names=['audio'],
        output_names=['emb16', 'emb567'],
        dynamic_axes={'audio': {1: 'seq_len'}},
        opset_version=17
    )
    print('Exported xlsr_twostage.onnx')

    # 검증
    import onnxruntime as ort
    sess = ort.InferenceSession(
        'vowel_recognition/method_6_embedding/xlsr_twostage.onnx')
    result = sess.run(None, {'audio': dummy.numpy()})
    print(f'emb16 shape: {result[0].shape}')   # (1, 1024)
    print(f'emb567 shape: {result[1].shape}')   # (1, 1024)

    # 속도 측정
    import time
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.run(None, {'audio': dummy.numpy()})
        times.append(time.perf_counter() - t0)
    avg = np.mean(times[5:]) * 1000
    print(f'ONNX 추론 평균: {avg:.0f}ms (warmup 제외)')

if __name__ == '__main__':
    export()
```

### 수정 3-2: 리샘플링을 scipy.signal.resample_poly로 교체

```
파일: main_integrated.py
함수: VowelRecognitionWorker._recognize()
```

```python
# 파일 상단에 import 추가
from scipy.signal import resample_poly
from math import gcd

# _recognize() 내부 리샘플링 부분 교체:
# 기존:
#   ratio = target_sr / sr
#   n_out = int(len(audio) * ratio)
#   indices = np.arange(n_out) / ratio
#   idx = np.clip(indices.astype(int), 0, len(audio) - 1)
#   audio = audio[idx]
# 새로:
if sr != target_sr:
    g = gcd(target_sr, sr)
    audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
```

### 수정 3-3: ONNX 추론기로 교체

```
파일: main_integrated.py
클래스: VowelRecognitionWorker
```

```python
class VowelRecognitionWorker(QObject):
    recognized = Signal(str, float)
    status_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._session = None
        self._s1_scaler = None
        self._s1_clf = None
        self._s2_scaler = None
        self._s2_clf = None
        self._target = None
        self._lock = threading.Lock()
        self._event = threading.Event()  # sleep 대신 Event 사용
        self._latest_item = None
        self._running = False

    def load_models(self):
        global RECOGNITION_AVAILABLE
        try:
            self.status_changed.emit("모델 로딩 중...")
            import pickle

            model_dir = os.path.join(os.path.dirname(__file__),
                                     'vowel_recognition', 'method_6_embedding')

            onnx_path = os.path.join(model_dir, 'xlsr_twostage.onnx')
            pkl_path = os.path.join(model_dir, 'twostage_model.pkl')

            if os.path.exists(onnx_path):
                # ONNX 모드
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.intra_op_num_threads = 4
                opts.inter_op_num_threads = 1
                self._session = ort.InferenceSession(onnx_path, opts)
                self._use_onnx = True
                self.status_changed.emit("ONNX 모델 로드 완료")
            else:
                # PyTorch 폴백
                import torch
                from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
                model_name = 'facebook/wav2vec2-large-xlsr-53'
                self._fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
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
        """최신 오디오만 유지 + Event로 즉시 깨움."""
        with self._lock:
            self._latest_item = (audio, sr)
        self._event.set()  # 워커 즉시 깨움

    def process_loop(self):
        self._running = True
        while self._running:
            self._event.wait(timeout=0.5)  # sleep 대신 Event 대기
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

    def _recognize(self, audio, sr):
        if not RECOGNITION_AVAILABLE:
            return None

        target_sr = 16000
        if sr != target_sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(target_sr, sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)

        if self._use_onnx:
            return self._recognize_onnx(audio, target_sr)
        else:
            return self._recognize_pytorch(audio, target_sr)

    def _recognize_onnx(self, audio, sr):
        """ONNX 추론: emb16 + emb567이 한 번에 나옴."""
        audio_input = audio.reshape(1, -1).astype(np.float32)
        emb16, emb567 = self._session.run(None, {'audio': audio_input})

        # Stage 1
        X1 = self._s1_scaler.transform(emb16)
        pred1 = self._s1_clf.predict(X1)[0]
        proba1 = self._s1_clf.predict_proba(X1)[0]
        classes1 = self._s1_clf.classes_

        # Stage 2
        if pred1 in self._target:
            X2 = self._s2_scaler.transform(emb567)
            pred2 = self._s2_clf.predict(X2)[0]
            proba2 = self._s2_clf.predict_proba(X2)[0]
            final = pred2
            conf = float(max(proba2))
        else:
            final = pred1
            conf = float(max(proba1))

        # 디버그 출력
        top3 = sorted(zip(classes1, proba1), key=lambda x: -x[1])[:3]
        s1_str = ' '.join(f'{c}={p:.0%}' for c, p in top3)
        print(f'[ONNX] S1:[{s1_str}]→{pred1}  final={final} ({conf:.0%})', flush=True)

        return final, conf

    def _recognize_pytorch(self, audio, sr):
        """기존 PyTorch 추론 (ONNX 없을 때 폴백)."""
        import torch
        inputs = self._fe(audio, sampling_rate=sr,
                          return_tensors="pt", padding=False)
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states

        def pool(frames):
            energy = frames.norm(dim=1)
            k = max(1, len(energy) // 2)
            top_idx = torch.topk(energy, k).indices
            return frames[top_idx].mean(dim=0).numpy().astype(np.float32)

        emb16 = pool(hidden[16].squeeze(0))
        emb5 = pool(hidden[5].squeeze(0))
        emb6 = pool(hidden[6].squeeze(0))
        emb7 = pool(hidden[7].squeeze(0))
        emb567 = (emb5 + emb6 + emb7) / 3.0

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
        else:
            final = pred1
            conf = float(max(proba1))

        return final, conf
```

### 수정 3-4: poll_timer 간격 단축

```
파일: main_integrated.py
함수: main()
```

```python
# 기존:
# poll_timer.setInterval(50)
# 변경:
poll_timer.setInterval(20)  # 50ms → 20ms (응답성 향상)
```

**기대 효과**:
- ONNX + Layer 16 절단: 추론 360ms → ~120-180ms (CPU), ~40-60ms (GPU)
- Event 기반 워커: 폴링 지연 0~20ms → 0~2ms
- poll_timer 20ms: 50ms → 20ms
- 리샘플링 개선: 에일리어싱 제거로 정확도 소폭 향상
- **총 end-to-end**: 560ms → ~300-350ms (ONNX CPU), ~200ms (ONNX GPU)

---

## TASK 4: 스냅샷 주기 단축 + 발성 시작 즉시 추론

### 문제
현재 최초 스냅샷까지 min_duration=0.15초 대기.
발성 시작 후 150ms 동안 아무 인식도 안 됨.

### 수정

```
파일: main_integrated.py
클래스: UtteranceCollector.__init__()
```

```python
# 기존:
# min_duration=0.15, snapshot_interval=0.15, max_duration=0.5
# 변경:
def __init__(self, sr=44100, min_duration=0.10, snapshot_interval=0.10,
             max_duration=0.4):
```

주의: min_duration을 0.10초(100ms) 미만으로 줄이면 너무 짧은 오디오가 넘어가서
인식 품질이 떨어질 수 있음. 0.10초가 안전한 하한선.

**기대 효과**: 첫 인식까지 150ms → 100ms, 갱신 주기 150ms → 100ms

---

## TASK 5: Confidence 기반 모핑 속도 조절

### 문제
현재 모음 인식 결과의 confidence와 관계없이 동일한 MORPH_SEC(0.45초)로 전환.
낮은 confidence일 때 즉각적인 전환은 시각적 오류로 느껴짐.

### 수정

```
파일: main_integrated.py
클래스: IntegratedCanvas
```

on_vowel_recognized 수정:

```python
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

    if smoothed_vowel in self._data:
        # ── confidence 기반 모핑 속도 조절 ──
        if confidence > 0.8:
            self._morph_speed = 0.35   # 빠르게 전환
        elif confidence > 0.6:
            self._morph_speed = 0.50   # 보통
        else:
            self._morph_speed = 0.70   # 느리게 (불확실할 때)
        self.trigger_morph(smoothed_vowel)
```

_tick에서 MORPH_SEC 대신 self._morph_speed 사용:

```python
def __init__(self, glyph_data):
    # ... 기존 초기화 ...
    self._morph_speed = MORPH_SEC  # 기본값

def _tick(self):
    # ...
    if self._animating and self._tgt:
        elapsed = now - self._t0
        self._t = min(elapsed / self._morph_speed, 1.0)  # MORPH_SEC → self._morph_speed
        # ... 나머지 동일
```

---

## TASK 6: 데이터 수집 모드 (장기적 정확도 향상)

이전 지시서와 동일하되, 실제 코드의 구조에 맞게 통합.

### 핵심: IntegratedWindow에 수집 모드 추가

```
파일: main_integrated.py
```

keyPressEvent에 'R' 키로 수집 모드 토글:

```python
def keyPressEvent(self, e):
    k = e.key()
    # ... 기존 키 처리 ...
    if k == Qt.Key.Key_R:
        self._toggle_collect_mode()
        return

def _toggle_collect_mode(self):
    if not hasattr(self, '_collect_mode'):
        self._collect_mode = False
        self._collect_vowel_idx = 0
        self._collect_count = {}
        self._collect_dir = os.path.join(os.path.dirname(__file__), 'collected_data')
        os.makedirs(self._collect_dir, exist_ok=True)

    self._collect_mode = not self._collect_mode
    if self._collect_mode:
        self._collect_session = time.strftime('%Y%m%d_%H%M%S')
        print(f'[수집 모드 ON] 세션: {self._collect_session}')
        print(f'  발음할 모음: {VOWELS[self._collect_vowel_idx]}')
        print(f'  스페이스바: 다음 모음, S: 저장')
    else:
        print('[수집 모드 OFF]')
```

수집 모드에서 UtteranceCollector의 스냅샷을 WAV로 저장:

```python
# poll_utterance를 수정하여 수집 모드일 때 저장도 수행
def poll_utterance():
    utt = collector.get_utterance()
    if utt is not None and RECOGNITION_AVAILABLE:
        worker.submit(utt, 44100)
        # 수집 모드일 때 자동 저장
        if hasattr(win, '_collect_mode') and win._collect_mode:
            import soundfile as sf
            vowel = VOWELS[win._collect_vowel_idx]
            cnt = win._collect_count.get(vowel, 0) + 1
            win._collect_count[vowel] = cnt
            fname = f'{win._collect_session}_{vowel}_{cnt:03d}.wav'
            fpath = os.path.join(win._collect_dir, fname)
            sf.write(fpath, utt, 44100)
            print(f'  저장: {fname} ({vowel} #{cnt})')
```

---

## 적용 순서 및 기대 효과

| 순서 | TASK | 주요 변경 | 지연 개선 | CPU 개선 | 난이도 |
|------|------|-----------|-----------|----------|--------|
| 1 | TASK 1 | YIN 중복제거 + 벡터화 | 간접적 | 50%↓ | 하 |
| 2 | TASK 2 | 렌더링 캐싱 | 프레임 안정화 | 90%↓ (정지시) | 하 |
| 3 | TASK 3 | ONNX + Event + 리샘플링 | 360→120ms | — | 중 |
| 4 | TASK 4 | 스냅샷 주기 단축 | 150→100ms | — | 하 |
| 5 | TASK 5 | confidence 모핑 속도 | 체감 개선 | — | 하 |
| 6 | TASK 6 | 데이터 수집 모드 | — | — | 중 |

### 의존성 설치

```bash
pip install onnxruntime onnx soundfile
# GPU 있다면:
# pip install onnxruntime-gpu
```

### ONNX 변환 (TASK 3 적용 전 필수)

```bash
cd VoiceTypo_integrated
python vowel_recognition/method_6_embedding/export_onnx.py
```

### 검증

```bash
# TASK 1 검증: YIN이 한 번만 실행되는지 확인
# AudioBridge.on_audio에 타이밍 로그 추가하여 측정

# TASK 2 검증: paintEvent 내 sdf_to_rgba 호출 횟수 로깅
# 정지 상태에서 1초간 호출 횟수가 60 → 0~2로 줄어야 함

# TASK 3 검증: ONNX 추론 시간
python -c "
import numpy as np, time, onnxruntime as ort
sess = ort.InferenceSession(
    'vowel_recognition/method_6_embedding/xlsr_twostage.onnx')
audio = np.random.randn(1, 8000).astype(np.float32)
times = []
for i in range(20):
    t0 = time.perf_counter()
    sess.run(None, {'audio': audio})
    times.append(time.perf_counter() - t0)
print(f'ONNX: {np.mean(times[5:])*1000:.0f}ms')
"
```

---

## 이전 지시서 v1에서 제거/수정된 항목

| v1 항목 | 변경 이유 |
|---------|-----------|
| TASK 1 파이프라인 비동기화 | 이미 비동기임 (VowelRecognitionWorker가 별도 스레드). 구조 변경 불필요 |
| TASK 4 포먼트 보조 피처 | 이미 시도하여 "+0.5%"로 효과 없음. 코드에 기록 있음 |
| TASK 5 SDFMorphRenderer 클래스 | 기존 IntegratedCanvas에 통합하는 것이 적절 |
| 200ms 버퍼 + 50ms hop | UtteranceCollector의 VAD 기반 수집이 이미 이 역할 수행 |
