"""마이크 실시간 입력 → 텍스트 효과 시각화 테스트.

audio_capture + pitch_detection → text_effects 연결.
피치→색상/스케일, 볼륨→크기, 비브라토→떨림을 실시간으로 확인.
키 1-7: 모음 전환 (모핑), ESC: 종료
"""

import sys, os, math, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal, QObject
from PySide6.QtGui import (QPainter, QColor, QImage, QPen, QFont,
                            QRadialGradient, QBrush)
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame

from audio_capture.capture import AudioCapture
from pitch_detection.yin import YinDetector
from pitch_detection.vibrato import VibratoAnalyzer
from pitch_detection.vad import VoiceActivityDetector

from text_morphing.glyph_morph_sdf_final_v2 import (
    VOWELS, GRID, MORPH_SEC, AA_WIDTH,
    _resting_sdf, _blend_glyphs, GlyphData, StrokeInfo,
)
from text_effects.test_effects import (
    load_glyph_data, compute_joint_mask, apply_vibrato_to_sdf,
    sdf_to_rgba, compute_colors,
)


# ═══════════════════════════════════════════════════
#  오디오 → Qt 브릿지 (스레드 안전)
# ═══════════════════════════════════════════════════
class AudioBridge(QObject):
    """오디오 콜백(별도 스레드)에서 Qt 메인 스레드로 데이터 전달."""
    updated = Signal(float, float, float, float, bool)  # freq, rms, vib_rate, vib_extent, vad

    def __init__(self, sample_rate=44100, blocksize=2048):
        super().__init__()
        self._detector = YinDetector(sample_rate)
        self._vibrato = VibratoAnalyzer(sample_rate / blocksize)
        self._vad = VoiceActivityDetector()

    def on_audio(self, chunk, sr):
        freq, rms = self._detector.detect(chunk)
        self._vad.update(rms, freq)
        if freq > 0:
            self._vibrato.push(freq)
        rate, extent = self._vibrato.get()
        self.updated.emit(freq, rms, rate, extent, self._vad.is_active)


# ═══════════════════════════════════════════════════
#  효과 파라미터 계산
# ═══════════════════════════════════════════════════
SCALE_SMOOTH = 0.08
VOLUME_SMOOTH = 0.18
COLOR_SMOOTH = 0.08
VIBRATO_SMOOTH = 0.15

FADE_IN_SPEED = 8.0     # 초당 opacity 증가량
FADE_OUT_SPEED = 2.5    # 초당 opacity 감소량 (1/0.4 = 0.4초에 완전 사라짐)

BASELINE_PITCH = 220.0  # 기준음 (Hz)
PITCH_UP_RANGE = 18     # 반음
PITCH_DOWN_RANGE = 6    # 반음


def pitch_to_ratio(freq, baseline=BASELINE_PITCH):
    """주파수 → -1~+1 피치 비율."""
    if freq <= 0 or baseline <= 0:
        return 0.0
    semitones = 12.0 * math.log2(freq / baseline)
    if semitones >= 0:
        return min(semitones / PITCH_UP_RANGE, 1.0)
    else:
        return max(semitones / PITCH_DOWN_RANGE, -1.0)


def rms_to_volume(rms):
    """RMS → 볼륨 스케일 (0.2 ~ 3.5)."""
    return min(0.2 + (rms ** 0.6) * 8.0, 3.5)


# ═══════════════════════════════════════════════════
#  캔버스
# ═══════════════════════════════════════════════════
class LiveCanvas(QWidget):
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
        self._opacity = 1.0
        self._glow = 0.0

        # 원본 입력값 (스무딩 전)
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

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(1000 // 60)
        self._timer.start()

    def on_voice_data(self, freq, rms, vib_rate, vib_extent, vad_active):
        """AudioBridge에서 호출 (메인 스레드)."""
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

        # 비브라토: rate→speed, extent→amount
        if vib_rate > 0 and vib_extent > 0:
            self._raw_vib_amount = min(vib_extent * 8.0, 30.0)
            self._raw_vib_speed = min(vib_rate / 10.0, 1.0)
        else:
            self._raw_vib_amount = 0.0
            self._raw_vib_speed = 0.0

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

        # 스무딩
        self._pitch_ratio += SCALE_SMOOTH * (self._raw_pitch - self._pitch_ratio)
        self._volume_scale += VOLUME_SMOOTH * (self._raw_volume - self._volume_scale)
        self._vib_amount += VIBRATO_SMOOTH * (self._raw_vib_amount - self._vib_amount)
        self._vib_speed += VIBRATO_SMOOTH * (self._raw_vib_speed - self._vib_speed)

        # VAD → opacity 페이드
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

        # HUD
        p.setPen(QPen(QColor(100, 100, 100), 1))
        p.setFont(QFont("Consolas", 10))
        vad_str = "VOICE" if self._vad_active else "---"
        freq_str = f"{self._freq:.0f}Hz" if self._freq > 0 else "---"
        vib_str = (f"vib: {self._vib_rate:.1f}Hz / {self._vib_extent:.1f}st"
                   if self._vib_rate > 0 else "vib: ---")
        hud = f"[{vad_str}] {freq_str}  rms={self._rms:.3f}  {vib_str}"
        p.drawText(QRectF(12, 8, w - 24, 20), Qt.AlignmentFlag.AlignLeft, hud)

        vowel_hud = f"vowel: {self._cur}  (1-7: morph, ESC: quit)"
        p.drawText(QRectF(12, h - 28, w - 24, 20), Qt.AlignmentFlag.AlignLeft, vowel_hud)

        p.end()


# ═══════════════════════════════════════════════════
#  메인 윈도우
# ═══════════════════════════════════════════════════
class LiveEffectWindow(QWidget):
    def __init__(self, glyph_data):
        super().__init__()
        self.setWindowTitle("Live Voice → Text Effects")
        self.resize(800, 700)
        self.setStyleSheet("background: #000;")

        self.canvas = LiveCanvas(glyph_data)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.close()
            return
        idx = k - Qt.Key.Key_1
        if 0 <= idx < len(VOWELS):
            self.canvas.trigger_morph(VOWELS[idx])
        super().keyPressEvent(e)


def main():
    app = QApplication(sys.argv)
    print("Loading glyphs...", flush=True)
    glyph_data = load_glyph_data()
    print("Done.", flush=True)

    # 오디오 브릿지
    bridge = AudioBridge()
    capture = AudioCapture()
    capture.add_listener(bridge.on_audio)

    win = LiveEffectWindow(glyph_data)
    bridge.updated.connect(win.canvas.on_voice_data)

    capture.start()
    print("Mic started. Speak or hum!", flush=True)

    win.show()
    ret = app.exec()

    capture.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
