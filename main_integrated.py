"""VoiceTypo 통합 데모 — 음성 → 모음 인식 → 타이포 변형 + 모핑.

발성하면:
  1. 모음 인식 → 해당 글리프로 모핑
  2. 피치 → 색상(빨강/파랑) + 세로/가로 스케일
  3. 볼륨 → 전체 크기
  4. 비브라토 → 떨림
  5. VAD → 페이드 인/아웃

실행:
  python main_integrated.py
"""

import sys, os, math, time, threading
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal, QObject
from PySide6.QtGui import (QPainter, QColor, QImage, QPen, QFont,
                            QRadialGradient, QBrush)
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout

from audio_capture.capture import AudioCapture
from pitch_detection.yin import YinDetector
from pitch_detection.vibrato import VibratoAnalyzer
from pitch_detection.vad import VoiceActivityDetector

from text_morphing.glyph_morph_sdf_final_v2 import (
    VOWELS as MORPH_VOWELS, GRID, MORPH_SEC, AA_WIDTH,
    _resting_sdf, _blend_glyphs, GlyphData, StrokeInfo,
)
from text_effects.test_effects import (
    load_glyph_data, compute_joint_mask, apply_vibrato_to_sdf,
    sdf_to_rgba, compute_colors,
)

# 모음 인식 모듈 (지연 로딩)
RECOGNITION_AVAILABLE = False


# ═══════════════════════════════════════════════════
#  모음 인식 워커 (백그라운드 스레드)
# ═══════════════════════════════════════════════════
class VowelRecognitionWorker(QObject):
    """백그라운드에서 XLSR-53 추론 → 모음 예측 결과를 시그널로 전달."""
    recognized = Signal(str, float)  # (vowel, confidence)
    status_changed = Signal(str)     # 상태 메시지

    def __init__(self):
        super().__init__()
        self._extractor = None
        self._classifier = None
        self._lock = threading.Lock()
        self._queue = []  # (audio, sr) 큐
        self._running = False

    def load_models(self):
        """모델 로드 (무거우므로 백그라운드에서 호출)."""
        global RECOGNITION_AVAILABLE
        try:
            self.status_changed.emit("모델 로딩 중...")

            import torch
            from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
            import pickle

            model_dir = os.path.join(os.path.dirname(__file__),
                                     'vowel_recognition', 'method_6_embedding')
            model_path = os.path.join(model_dir, 'twostage_model.pkl')

            if not os.path.exists(model_path):
                self.status_changed.emit("모델 파일 없음: twostage_model.pkl")
                return

            # Feature extractor + model
            model_name = 'facebook/wav2vec2-large-xlsr-53'
            self._fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self._model = Wav2Vec2Model.from_pretrained(model_name)
            self._model.eval()

            # Classifier
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self._s1_scaler = data['stage1']['scaler']
            self._s1_clf = data['stage1']['clf']
            self._s2_scaler = data['stage2']['scaler']
            self._s2_clf = data['stage2']['clf']
            self._target = data['stage2']['target_vowels']

            RECOGNITION_AVAILABLE = True
            self.status_changed.emit("모델 로드 완료")
        except Exception as e:
            self.status_changed.emit(f"모델 로드 실패: {e}")

    def submit(self, audio, sr):
        """오디오를 인식 큐에 추가."""
        with self._lock:
            self._queue = [(audio, sr)]  # 최신 것만 유지

    def process_loop(self):
        """백그라운드 스레드에서 실행되는 메인 루프."""
        import torch
        self._running = True
        while self._running:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)

            if item is None:
                time.sleep(0.02)
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

    def _recognize(self, audio, sr):
        """XLSR-53 추론 + 2단계 분류."""
        import torch

        if not RECOGNITION_AVAILABLE:
            return None

        # 리샘플링 to 16kHz
        target_sr = 16000
        if sr != target_sr:
            ratio = target_sr / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            idx = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[idx]

        inputs = self._fe(audio, sampling_rate=target_sr,
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

        # Stage 1
        X1 = self._s1_scaler.transform(emb16.reshape(1, -1))
        pred1 = self._s1_clf.predict(X1)[0]
        proba1 = self._s1_clf.predict_proba(X1)[0]
        classes1 = self._s1_clf.classes_

        # Stage 2
        if pred1 in self._target:
            X2 = self._s2_scaler.transform(emb567.reshape(1, -1))
            pred2 = self._s2_clf.predict(X2)[0]
            proba2 = self._s2_clf.predict_proba(X2)[0]
            final = pred2
            conf = float(max(proba2))
            # 디버그
            top3 = sorted(zip(classes1, proba1), key=lambda x: -x[1])[:3]
            s1_str = ' '.join(f'{c}={p:.0%}' for c, p in top3)
            s2_str = ' '.join(f'{c}={p:.0%}' for c, p in zip(self._s2_clf.classes_, proba2))
            print(f'[모음] S1:[{s1_str}]→{pred1}  S2:[{s2_str}]→{final} ({conf:.0%})', flush=True)
        else:
            final = pred1
            conf = float(max(proba1))
            # 디버그
            top3 = sorted(zip(classes1, proba1), key=lambda x: -x[1])[:3]
            s1_str = ' '.join(f'{c}={p:.0%}' for c, p in top3)
            print(f'[모음] S1:[{s1_str}]→{final} ({conf:.0%})', flush=True)

        return final, conf


# ═══════════════════════════════════════════════════
#  오디오 → Qt 브릿지 (피치/VAD/비브라토)
# ═══════════════════════════════════════════════════
class AudioBridge(QObject):
    updated = Signal(float, float, float, float, bool)  # freq, rms, vib_rate, vib_extent, vad
    spectrum_updated = Signal(object)  # FFT magnitude array (numpy)

    def __init__(self, sample_rate=44100, blocksize=2048):
        super().__init__()
        self._sr = sample_rate
        self._detector = YinDetector(sample_rate)
        self._vibrato = VibratoAnalyzer(sample_rate / blocksize)
        self._vad = VoiceActivityDetector()
        self._window = np.hanning(blocksize).astype(np.float32)

    def on_audio(self, chunk, sr):
        freq, rms = self._detector.detect(chunk)
        self._vad.update(rms, freq)
        self._vibrato.push(freq, rms)
        rate, extent = self._vibrato.get()
        self.updated.emit(freq, rms, rate, extent, self._vad.is_active)

        # FFT spectrum (0 ~ 4000 Hz)
        n = len(chunk)
        if n == len(self._window):
            fft_mag = np.abs(np.fft.rfft(chunk * self._window))
        else:
            fft_mag = np.abs(np.fft.rfft(chunk * np.hanning(n)))
        max_bin = int(4000 * n / sr)
        self.spectrum_updated.emit(fft_mag[:max_bin])


# ═══════════════════════════════════════════════════
#  발화 수집기 (VAD 기반)
# ═══════════════════════════════════════════════════
class UtteranceCollector:
    """스트리밍 방식: 발성 중 주기적으로 현재 버퍼 스냅샷을 제공."""
    def __init__(self, sr=44100, min_duration=0.15, snapshot_interval=0.15,
                 max_duration=0.5):
        self.sr = sr
        self.min_samples = int(min_duration * sr)
        self.snapshot_interval = snapshot_interval  # 초 단위
        self.max_samples = int(max_duration * sr)   # 스냅샷 최대 길이

        self._yin = YinDetector(sr)
        self._vad = VoiceActivityDetector()

        self._buffer = []
        self._collecting = False
        self._ready = False
        self._utterance = None
        self._last_snapshot = 0.0
        self._lock = threading.Lock()

    def on_audio(self, chunk, sr):
        freq, rms = self._yin.detect(chunk)
        self._vad.update(rms, freq)

        with self._lock:
            if self._vad.is_active:
                if not self._collecting:
                    self._collecting = True
                    self._buffer = []
                    self._last_snapshot = 0.0
                self._buffer.append(chunk.copy())

                # 최소 길이 이상이면 주기적으로 스냅샷 제공
                total = sum(len(c) for c in self._buffer)
                elapsed = total / self.sr
                if (total >= self.min_samples and
                        elapsed - self._last_snapshot >= self.snapshot_interval):
                    full = np.concatenate(self._buffer)
                    # 최근 max_duration만 잘라서 제공
                    if len(full) > self.max_samples:
                        full = full[-self.max_samples:]
                    self._utterance = full
                    self._ready = True
                    self._last_snapshot = elapsed
            else:
                if self._collecting:
                    # 발성 종료 — 버퍼 리셋 (끝부분 잡음 인식 방지)
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


# ═══════════════════════════════════════════════════
#  효과 파라미터
# ═══════════════════════════════════════════════════
SCALE_SMOOTH = 0.08
VOLUME_SMOOTH = 0.18
VIBRATO_SMOOTH = 0.15
FADE_IN_SPEED = 8.0
FADE_OUT_SPEED = 2.5
BASELINE_PITCH = 220.0
PITCH_UP_RANGE = 18
PITCH_DOWN_RANGE = 6


def pitch_to_ratio(freq, baseline=BASELINE_PITCH):
    if freq <= 0 or baseline <= 0:
        return 0.0
    semitones = 12.0 * math.log2(freq / baseline)
    if semitones >= 0:
        return min(semitones / PITCH_UP_RANGE, 1.0)
    else:
        return max(semitones / PITCH_DOWN_RANGE, -1.0)


def rms_to_volume(rms):
    return min(0.2 + (rms ** 0.6) * 8.0, 3.5)


# ═══════════════════════════════════════════════════
#  모음 이름 → 모핑 시스템 모음 이름 매핑
# ═══════════════════════════════════════════════════
# 인식 시스템: ["아","어","오","우","으","이","에"]
# 모핑 시스템 MORPH_VOWELS: ["아","이","우","에","오","으","어"]
# 동일한 글자이므로 직접 사용 가능


# ═══════════════════════════════════════════════════
#  메인 캔버스
# ═══════════════════════════════════════════════════
class IntegratedCanvas(QWidget):
    def __init__(self, glyph_data):
        super().__init__()
        self._data = glyph_data
        self._cur = "아"
        self._tgt = None
        self._t = 0.0
        self._t0 = 0.0
        self._animating = False
        self._sdf = _resting_sdf(self._data["아"])
        self._joint_dists = {ch: compute_joint_mask(d) for ch, d in glyph_data.items()}
        self._joint_dist = self._joint_dists["아"]
        self._vibrato_phase = 0.0
        self._last_tick = time.monotonic()

        # 스무딩된 효과 파라미터
        self._pitch_ratio = 0.0
        self._volume_scale = 0.5
        self._vib_amount = 0.0
        self._vib_speed = 0.0
        self._opacity = 0.0
        self._glow = 0.0

        # 원본 입력값
        self._raw_pitch = 0.0
        self._raw_volume = 0.5
        self._raw_vib_amount = 0.0
        self._raw_vib_speed = 0.0
        self._vad_active = False

        # HUD 표시용
        self._freq = 0.0
        self._rms = 0.0
        self._vib_rate = 0.0
        self._vib_extent = 0.0
        self._recognized_vowel = "—"
        self._recognized_conf = 0.0
        self._model_status = "모델 로딩 대기..."

        # 스무딩: 히스테리시스
        self._smooth_conf_threshold = 0.35   # 이 미만이면 무시
        self._smooth_switch_threshold = 0.6  # 다른 모음으로 전환하려면 이 이상
        self._smooth_current = None          # 현재 확정된 모음

        # 디버그: 정답 라벨 입력
        self._gt_label = "—"  # 현재 선택된 정답
        self._gt_log = []     # (시각, 정답, 예측, 확신도) 기록

        # 60 FPS 타이머
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(1000 // 60)
        self._timer.start()

    def on_voice_data(self, freq, rms, vib_rate, vib_extent, vad_active):
        self._freq = freq
        self._rms = rms
        self._vib_rate = vib_rate
        self._vib_extent = vib_extent
        self._vad_active = vad_active

        if vad_active and freq > 0:
            self._raw_pitch = pitch_to_ratio(freq)
            self._raw_volume = rms_to_volume(rms)
            self._glow = min(rms * 5, 1.0)
        else:
            self._raw_pitch = 0.0
            self._raw_volume = 0.5
            self._glow = 0.0
            # 발성 종료 시 히스테리시스 상태 리셋
            self._smooth_current = None

        if vib_rate > 0 and vib_extent > 0:
            self._raw_vib_amount = min(vib_extent * 8.0, 30.0)
            self._raw_vib_speed = min(vib_rate / 10.0, 1.0)
        else:
            self._raw_vib_amount = 0.0
            self._raw_vib_speed = 0.0

    def set_gt_label(self, vowel):
        """키보드로 정답 라벨 설정."""
        self._gt_label = vowel
        print(f'[정답 설정] {vowel}', flush=True)

    def on_vowel_recognized(self, vowel, confidence):
        # ── 스무딩: 히스테리시스 ──
        # 1) 신뢰도 임계값 미만이면 무시
        if confidence < self._smooth_conf_threshold:
            return

        # 2) 히스테리시스: 같은 모음이면 유지, 다른 모음은 높은 신뢰도 필요
        if self._smooth_current is None:
            self._smooth_current = vowel
        elif vowel != self._smooth_current:
            if confidence < self._smooth_switch_threshold:
                return  # 전환 문턱 미달 → 현재 모음 유지
            self._smooth_current = vowel

        smoothed_vowel = self._smooth_current

        # HUD도 스무딩된 결과만 표시
        self._recognized_vowel = smoothed_vowel
        self._recognized_conf = confidence

        # 정답 비교 로그 (스무딩 전 raw 결과 기록)
        gt = self._gt_label
        if gt != "—":
            from datetime import datetime
            mark = 'O' if smoothed_vowel == gt else 'X'
            self._gt_log.append((datetime.now().strftime('%H:%M:%S'), gt, smoothed_vowel, confidence))
            correct = sum(1 for _, g, p, _ in self._gt_log if g == p)
            total = len(self._gt_log)
            print(f'[비교] 정답={gt} 예측={vowel}→{smoothed_vowel} {mark} ({confidence:.0%})  '
                  f'누적: {correct}/{total} ({correct/total*100:.1f}%)', flush=True)
        if smoothed_vowel in self._data:
            self.trigger_morph(smoothed_vowel)

    def on_model_status(self, status):
        self._model_status = status

    def trigger_morph(self, vowel):
        if vowel == self._cur and not self._animating:
            return
        if self._animating and self._tgt:
            self._cur = self._tgt
        self._tgt = vowel
        self._t0 = time.monotonic()
        self._animating = True

    def _tick(self):
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        self._pitch_ratio += SCALE_SMOOTH * (self._raw_pitch - self._pitch_ratio)
        self._volume_scale += VOLUME_SMOOTH * (self._raw_volume - self._volume_scale)
        self._vib_amount += VIBRATO_SMOOTH * (self._raw_vib_amount - self._vib_amount)
        self._vib_speed += VIBRATO_SMOOTH * (self._raw_vib_speed - self._vib_speed)

        if self._vad_active:
            self._opacity = min(self._opacity + dt * FADE_IN_SPEED, 1.0)
        else:
            self._opacity = max(self._opacity - dt * FADE_OUT_SPEED, 0.0)

        self._vibrato_phase += dt * self._vib_speed * 3.0

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

        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(0, 0, 0))

        sdf = apply_vibrato_to_sdf(self._sdf, self._vib_amount,
                                    self._vibrato_phase, self._joint_dist)
        c_top, c_bot, glow = compute_colors(self._pitch_ratio)
        qimg = sdf_to_rgba(sdf, c_top, c_bot, self._opacity)

        sx = 1 - self._pitch_ratio * 0.65
        sy = 1 + self._pitch_ratio * 0.9
        area = sx * sy
        norm = 1.0 / math.sqrt(max(area, 0.01))
        sx *= norm
        sy *= norm

        # 글로우
        if self._glow > 0.01:
            cx, cy = w / 2, h / 2
            radius = min(w, h) * 0.3 * (self._volume_scale / 3.5)
            grad = QRadialGradient(QPointF(cx, cy), radius)
            gc = QColor(glow[0], glow[1], glow[2], int(80 * self._glow))
            grad.setColorAt(0, gc)
            grad.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), radius, radius)

        # 이미지
        margin = 40
        max_w, max_h = w - margin * 2, h - margin * 2
        fit = min(max_w / sx, max_h / sy)
        scale = fit * (self._volume_scale / 3.5)
        img_w = sx * scale
        img_h = sy * scale
        x0 = (w - img_w) / 2
        y0 = (h - img_h) / 2
        p.drawImage(QRectF(x0, y0, img_w, img_h),
                     qimg, QRectF(0, 0, GRID, GRID))

        # HUD 상단: 피치/VAD
        p.setPen(QPen(QColor(100, 100, 100), 1))
        p.setFont(QFont("Consolas", 10))
        vad_str = "VOICE" if self._vad_active else "---"
        freq_str = f"{self._freq:.0f}Hz" if self._freq > 0 else "---"
        vib_str = (f"vib: {self._vib_rate:.1f}Hz/{self._vib_extent:.1f}st"
                   if self._vib_rate > 0 else "vib: ---")
        hud_top = f"[{vad_str}] {freq_str}  rms={self._rms:.3f}  {vib_str}"
        p.drawText(QRectF(12, 8, w - 24, 20), Qt.AlignmentFlag.AlignLeft, hud_top)

        # HUD 하단: 인식 결과
        p.setFont(QFont("Consolas", 11))
        recog_str = f"recognized: {self._recognized_vowel} ({self._recognized_conf:.0%})"
        gt_str = f"GT: {self._gt_label}"
        if self._gt_log:
            correct = sum(1 for _, g, p, _ in self._gt_log if g == p)
            total = len(self._gt_log)
            gt_str += f"  [{correct}/{total}={correct/total*100:.0f}%]"
        cur_str = f"current: {self._cur}"
        model_str = self._model_status
        hud_bot = f"{recog_str}  |  {gt_str}  |  {cur_str}  |  {model_str}"
        p.drawText(QRectF(12, h - 28, w - 24, 20), Qt.AlignmentFlag.AlignLeft, hud_bot)

        # 키 가이드
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.setFont(QFont("Consolas", 9))
        guide = "1:아 2:어 3:오 4:우 5:으 6:이 7:에 0:해제  |  D: debug  |  ESC: quit"
        p.drawText(QRectF(12, h - 48, w - 24, 20), Qt.AlignmentFlag.AlignLeft, guide)

        p.end()


# ═══════════════════════════════════════════════════
#  디버그 패널
# ═══════════════════════════════════════════════════
class DebugPanel(QWidget):
    """스펙트럼, 피치/볼륨 히스토리, 실시간 수치 표시."""

    PANEL_WIDTH = 300
    SPECTRUM_BINS = 80
    HISTORY_LEN = 120

    def __init__(self):
        super().__init__()
        self.setFixedWidth(self.PANEL_WIDTH)

        # spectrum
        self._spectrum = np.zeros(self.SPECTRUM_BINS, dtype=np.float32)

        # histories
        self._pitch_history = deque([0.0] * self.HISTORY_LEN,
                                     maxlen=self.HISTORY_LEN)
        self._rms_history = deque([0.0] * self.HISTORY_LEN,
                                   maxlen=self.HISTORY_LEN)

        # raw stats
        self._freq = 0.0
        self._rms = 0.0
        self._vib_rate = 0.0
        self._vib_extent = 0.0
        self._vad_active = False
        self._recognized_vowel = "—"
        self._recognized_conf = 0.0
        self._model_status = "..."

        # effect params (synced from canvas)
        self._pitch_ratio = 0.0
        self._volume_scale = 0.5
        self._vib_amount = 0.0
        self._vib_speed = 0.0
        self._opacity = 0.0
        self._morph_state = ""

    # ---------- slots ----------

    def on_spectrum(self, fft_mag):
        n = len(fft_mag)
        if n == 0:
            return
        bins = self.SPECTRUM_BINS
        if n >= bins:
            idx = np.linspace(0, n - 1, bins).astype(int)
            self._spectrum = fft_mag[idx]
        else:
            self._spectrum[:n] = fft_mag
            self._spectrum[n:] = 0

    def on_voice_data(self, freq, rms, vib_rate, vib_extent, vad_active):
        self._freq = freq
        self._rms = rms
        self._vib_rate = vib_rate
        self._vib_extent = vib_extent
        self._vad_active = vad_active
        self._pitch_history.append(freq if freq > 0 else 0.0)
        self._rms_history.append(rms)
        self.update()

    def on_vowel_recognized(self, vowel, confidence):
        self._recognized_vowel = vowel
        self._recognized_conf = confidence

    def on_model_status(self, status):
        self._model_status = status

    def sync_from_canvas(self, canvas):
        self._pitch_ratio = canvas._pitch_ratio
        self._volume_scale = canvas._volume_scale
        self._vib_amount = canvas._vib_amount
        self._vib_speed = canvas._vib_speed
        self._opacity = canvas._opacity
        if canvas._animating and canvas._tgt:
            self._morph_state = f"{canvas._cur} → {canvas._tgt} ({canvas._t:.0%})"
        else:
            self._morph_state = canvas._cur

    # ---------- paint ----------

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(17, 17, 17))

        pad = 10
        dw = w - pad * 2
        y = pad

        # ── SPECTRUM ──
        p.setPen(QPen(QColor(90, 90, 90)))
        p.setFont(QFont("Consolas", 9))
        p.drawText(pad, y + 10, "SPECTRUM")
        y += 16
        spec_h = 80
        self._draw_spectrum(p, pad, y, dw, spec_h)
        y += spec_h + 2

        # freq labels
        p.setPen(QPen(QColor(50, 50, 50)))
        p.setFont(QFont("Consolas", 7))
        for fl in [500, 1000, 2000, 3000]:
            xp = pad + int(fl / 4000 * dw)
            p.drawText(xp - 8, y + 9, str(fl))
        y += 14

        # ── PITCH HISTORY ──
        p.setPen(QPen(QColor(90, 90, 90)))
        p.setFont(QFont("Consolas", 9))
        p.drawText(pad, y + 10, "PITCH")
        y += 16
        pitch_h = 60
        self._draw_pitch_history(p, pad, y, dw, pitch_h)
        y += pitch_h + 6

        # ── VOLUME HISTORY ──
        p.setPen(QPen(QColor(90, 90, 90)))
        p.setFont(QFont("Consolas", 9))
        p.drawText(pad, y + 10, "VOLUME")
        y += 16
        rms_h = 40
        self._draw_rms_history(p, pad, y, dw, rms_h)
        y += rms_h + 10

        # ── SEPARATOR ──
        p.setPen(QPen(QColor(40, 40, 40)))
        p.drawLine(pad, y, w - pad, y)
        y += 8

        # ── STATS ──
        p.setFont(QFont("Consolas", 10))
        lh = 18

        def stat(label, value, color=QColor(140, 140, 140)):
            nonlocal y
            p.setPen(QPen(QColor(75, 75, 75)))
            p.drawText(pad, y + 12, label)
            p.setPen(QPen(color))
            p.drawText(pad + 120, y + 12, str(value))
            y += lh

        vad_c = QColor(74, 222, 128) if self._vad_active else QColor(100, 100, 100)
        stat("VAD", "ON" if self._vad_active else "OFF", vad_c)

        freq_s = f"{self._freq:.1f} Hz" if self._freq > 0 else "---"
        stat("Frequency", freq_s, QColor(100, 180, 255))

        semitones = ""
        if self._freq > 0:
            st = 12.0 * math.log2(self._freq / BASELINE_PITCH)
            semitones = f"  ({st:+.1f} st)"
        stat("Pitch", f"{self._pitch_ratio:+.3f}{semitones}", QColor(180, 140, 255))

        db = 20 * math.log10(max(self._rms, 1e-6))
        stat("Volume", f"{db:.1f} dB  (rms {self._rms:.4f})")

        vowel_c = QColor(255, 200, 80) if self._recognized_conf > 0.5 else QColor(100, 100, 100)
        stat("Vowel", f"{self._recognized_vowel}  ({self._recognized_conf:.0%})", vowel_c)

        if self._vib_rate > 0:
            vib_s = f"{self._vib_rate:.1f} Hz / {self._vib_extent:.2f} st"
        else:
            vib_s = "---"
        stat("Vibrato", vib_s)

        y += 4
        p.setPen(QPen(QColor(40, 40, 40)))
        p.drawLine(pad, y, w - pad, y)
        y += 8

        # ── EFFECT PARAMS ──
        p.setPen(QPen(QColor(90, 90, 90)))
        p.setFont(QFont("Consolas", 9))
        p.drawText(pad, y + 10, "EFFECTS")
        y += 16
        p.setFont(QFont("Consolas", 10))

        stat("Vol scale", f"{self._volume_scale:.2f}")
        stat("Vib amount", f"{self._vib_amount:.1f}")
        stat("Vib speed", f"{self._vib_speed:.2f}")
        stat("Opacity", f"{self._opacity:.2f}")

        sx = 1 - self._pitch_ratio * 0.65
        sy = 1 + self._pitch_ratio * 0.9
        stat("Scale X / Y", f"{sx:.2f} / {sy:.2f}")

        stat("Morph", self._morph_state, QColor(255, 160, 100))

        y += 6
        p.setPen(QPen(QColor(40, 40, 40)))
        p.drawLine(pad, y, w - pad, y)
        y += 8

        # Model status
        p.setPen(QPen(QColor(55, 55, 55)))
        p.setFont(QFont("Consolas", 8))
        p.drawText(pad, y + 10, self._model_status)

        p.end()

    # ---------- draw helpers ----------

    def _draw_spectrum(self, p, x, y, w, h):
        spec = self._spectrum
        n = len(spec)
        if n < 2:
            return
        max_val = max(float(np.max(spec)), 1e-6)
        bar_w = w / n
        for i in range(n):
            val = min(float(spec[i]) / max_val, 1.0)
            bar_h = val * h
            hue = max(0, 240 - int(300 * i / n)) % 360
            lum = int(50 + 90 * val)
            color = QColor.fromHsl(hue, 200, lum)
            p.fillRect(QRectF(x + i * bar_w, y + h - bar_h,
                              max(bar_w - 0.5, 1), bar_h), color)

    def _draw_pitch_history(self, p, x, y, w, h):
        pts = list(self._pitch_history)
        n = len(pts)
        if n < 2:
            return
        # grid lines (log scale 80-1100 Hz)
        log_lo, log_hi = math.log2(80), math.log2(1100)
        p.setPen(QPen(QColor(30, 30, 30)))
        p.setFont(QFont("Consolas", 7))
        for fl in [100, 200, 400, 800]:
            ratio = (math.log2(fl) - log_lo) / (log_hi - log_lo)
            fy = y + h - ratio * h
            if y <= fy <= y + h:
                p.drawLine(int(x), int(fy), int(x + w), int(fy))
                p.setPen(QPen(QColor(45, 45, 45)))
                p.drawText(int(x + w - 26), int(fy - 2), str(fl))
                p.setPen(QPen(QColor(30, 30, 30)))

        # pitch line
        p.setPen(QPen(QColor(100, 200, 255), 1.5))
        step = w / (n - 1)
        prev = None
        for i, freq in enumerate(pts):
            if freq <= 0:
                prev = None
                continue
            ratio = (math.log2(max(freq, 80)) - log_lo) / (log_hi - log_lo)
            ratio = max(0.0, min(1.0, ratio))
            px = x + i * step
            py = y + h - ratio * h
            if prev is not None:
                p.drawLine(QPointF(*prev), QPointF(px, py))
            prev = (px, py)

    def _draw_rms_history(self, p, x, y, w, h):
        pts = list(self._rms_history)
        n = len(pts)
        if n < 2:
            return
        max_rms = max(max(pts), 0.01)
        step = w / (n - 1)

        # filled area
        fill_color = QColor(80, 200, 120, 50)
        for i in range(n - 1):
            v = min(pts[i] / max_rms, 1.0)
            bh = v * h
            p.fillRect(QRectF(x + i * step, y + h - bh, step + 0.5, bh), fill_color)

        # line
        p.setPen(QPen(QColor(80, 200, 120), 1))
        prev = None
        for i, rms in enumerate(pts):
            val = min(rms / max_rms, 1.0)
            px = x + i * step
            py = y + h - val * h
            if prev is not None:
                p.drawLine(QPointF(*prev), QPointF(px, py))
            prev = (px, py)


# ═══════════════════════════════════════════════════
#  메인 윈도우
# ═══════════════════════════════════════════════════
class IntegratedWindow(QWidget):
    def __init__(self, glyph_data):
        super().__init__()
        self.setWindowTitle("VoiceTypo Integrated Demo")
        self.resize(1100, 700)
        self.setStyleSheet("background: #000;")

        self.canvas = IntegratedCanvas(glyph_data)
        self.debug_panel = DebugPanel()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.debug_panel, 0)

        self._debug_visible = True
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # sync effect params from canvas → debug panel (every frame)
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(
            lambda: self.debug_panel.sync_from_canvas(self.canvas))
        self._sync_timer.setInterval(1000 // 30)
        self._sync_timer.start()

    def _toggle_debug(self):
        self._debug_visible = not self._debug_visible
        self.debug_panel.setVisible(self._debug_visible)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.close()
            return
        if k == Qt.Key.Key_D:
            self._toggle_debug()
            return
        # 1-7: 정답 라벨 설정, 0: 해제
        GT_VOWELS = ['아', '어', '오', '우', '으', '이', '에']
        idx = k - Qt.Key.Key_1
        if 0 <= idx < len(GT_VOWELS):
            self.canvas.set_gt_label(GT_VOWELS[idx])
            return
        if k == Qt.Key.Key_0:
            self.canvas.set_gt_label("—")
            print('[정답 해제]', flush=True)
            return
        super().keyPressEvent(e)


# ═══════════════════════════════════════════════════
#  메인
# ═══════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)

    print("Loading glyphs...", flush=True)
    glyph_data = load_glyph_data()
    print("Done.", flush=True)

    # 오디오 브릿지 (피치/VAD/비브라토)
    bridge = AudioBridge()
    capture = AudioCapture()
    capture.add_listener(bridge.on_audio)

    # 발화 수집기
    collector = UtteranceCollector(sr=44100)
    capture.add_listener(collector.on_audio)

    # 모음 인식 워커
    worker = VowelRecognitionWorker()

    # 윈도우
    win = IntegratedWindow(glyph_data)
    bridge.updated.connect(win.canvas.on_voice_data)
    worker.recognized.connect(win.canvas.on_vowel_recognized)
    worker.status_changed.connect(win.canvas.on_model_status)

    # 디버그 패널 연결
    bridge.updated.connect(win.debug_panel.on_voice_data)
    bridge.spectrum_updated.connect(win.debug_panel.on_spectrum)
    worker.recognized.connect(win.debug_panel.on_vowel_recognized)
    worker.status_changed.connect(win.debug_panel.on_model_status)

    # 모델 로딩 (백그라운드)
    def load_and_run():
        worker.load_models()
        worker.process_loop()

    model_thread = threading.Thread(target=load_and_run, daemon=True)
    model_thread.start()

    # 발화 → 인식 폴링 (100ms 주기)
    poll_timer = QTimer()

    def poll_utterance():
        utt = collector.get_utterance()
        if utt is not None and RECOGNITION_AVAILABLE:
            worker.submit(utt, 44100)

    poll_timer.timeout.connect(poll_utterance)
    poll_timer.setInterval(50)
    poll_timer.start()

    # 마이크 시작
    capture.start()
    print("Mic started. Speak!", flush=True)

    win.show()
    ret = app.exec()

    worker.stop()
    capture.stop()
    poll_timer.stop()

    # 테스트 로그 저장
    log = win.canvas._gt_log
    if log:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(os.path.dirname(__file__), f'live_test_{ts}.csv')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('time,gt,pred,conf\n')
            for t, g, p, c in log:
                f.write(f'{t},{g},{p},{c:.4f}\n')

        correct = sum(1 for _, g, p, _ in log if g == p)
        total = len(log)
        print(f'\n{"="*50}')
        print(f'  테스트 결과: {correct}/{total} ({correct/total*100:.1f}%)')
        vowels_tested = sorted(set(g for _, g, _, _ in log))
        for v in vowels_tested:
            vr = [(g, p) for _, g, p, _ in log if g == v]
            vc = sum(1 for g, p in vr if g == p)
            print(f'    {v}: {vc}/{len(vr)}')
        print(f'  저장: {log_path}')
        print(f'{"="*50}')

    sys.exit(ret)


if __name__ == "__main__":
    main()
