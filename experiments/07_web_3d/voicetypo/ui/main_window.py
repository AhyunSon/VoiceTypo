"""메인 윈도우 — 실시간 음성 → 타이포 변형.

VowelTracker → SDFRenderer → 캔버스 렌더링.
피치(YIN) → 색·스케일, 볼륨(RMS) → 크기, 비브라토 → 떨림.
"""

import math
import time
from collections import deque

import numpy as np

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, Signal, QObject
from PySide6.QtGui import (QPainter, QColor, QImage, QPen, QFont,
                            QRadialGradient, QBrush)
from PySide6.QtWidgets import QWidget, QHBoxLayout

from ..audio.capture import AudioCapture
from ..audio.formant import FormantTracker
from ..vowel.space import formant_to_xyz
from ..vowel.tracker import VowelTracker
from ..sdf.renderer import SDFRenderer, GRID

import sys, os
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pitch_detection.yin import YinDetector
from pitch_detection.vibrato import VibratoAnalyzer
from pitch_detection.vad import VoiceActivityDetector

# ─── 효과 파라미터 ───
SCALE_SMOOTH = 0.08
VOLUME_SMOOTH = 0.18
VIBRATO_SMOOTH = 0.15
FADE_IN_SPEED = 8.0
FADE_OUT_SPEED = 2.5
BASELINE_PITCH = 220.0
PITCH_UP_RANGE = 18
PITCH_DOWN_RANGE = 6


def _pitch_to_ratio(freq, baseline=BASELINE_PITCH):
    if freq <= 0 or baseline <= 0:
        return 0.0
    semitones = 12.0 * math.log2(freq / baseline)
    if semitones >= 0:
        return min(semitones / PITCH_UP_RANGE, 1.0)
    else:
        return max(semitones / PITCH_DOWN_RANGE, -1.0)


def _rms_to_volume(rms):
    return min(0.2 + (rms ** 0.6) * 8.0, 3.5)


class AudioBridge(QObject):
    """오디오 → 포먼트 + 피치 + VAD 통합 브릿지."""

    updated = Signal(float, float, float, float, bool)  # freq, rms, vib_rate, vib_extent, vad
    formant_updated = Signal(float, float, float)        # f1, f2, f3

    def __init__(self, sample_rate=44100, blocksize=882):
        super().__init__()
        self._sr = sample_rate
        self._yin = YinDetector(sample_rate)
        self._vibrato = VibratoAnalyzer(sample_rate / blocksize)
        self._vad = VoiceActivityDetector()
        self._formant = FormantTracker(sample_rate)

    def on_audio(self, chunk, sr):
        # 피치 + RMS
        freq, rms = self._yin.detect(chunk)
        self._vad.update(rms, freq)
        self._vibrato.push(freq, rms)
        rate, extent = self._vibrato.get()
        self.updated.emit(freq, rms, rate, extent, self._vad.is_active)

        # 포먼트
        f1, f2, f3 = self._formant.update(chunk)
        self.formant_updated.emit(f1, f2, f3)


class VoiceTypoCanvas(QWidget):
    """메인 렌더링 캔버스."""

    def __init__(self, renderer):
        super().__init__()
        self._renderer = renderer
        self._tracker = VowelTracker()

        # 스무딩된 효과 값
        self._pitch_ratio = 0.0
        self._volume_scale = 0.5
        self._vib_amount = 0.0
        self._vib_speed = 0.0
        self._opacity = 0.0
        self._glow = 0.0

        # 원본 입력
        self._raw_pitch = 0.0
        self._raw_volume = 0.5
        self._raw_vib_amount = 0.0
        self._raw_vib_speed = 0.0
        self._vad_active = False

        # HUD
        self._freq = 0.0
        self._rms = 0.0
        self._label = ''
        self._weights = {}
        self._last_tick = time.monotonic()

        # 캐시
        self._cached_qimg = None
        self._cache_dirty = True

        # 60 FPS
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(1000 // 60)
        self._timer.start()

    def set_centroids(self, centroids):
        """캘리브레이션 결과 적용."""
        self._tracker = VowelTracker(centroids=centroids)

    def on_voice_data(self, freq, rms, vib_rate, vib_extent, vad_active):
        self._freq = freq
        self._rms = rms
        self._vad_active = vad_active

        if vad_active and freq > 0:
            self._raw_pitch = _pitch_to_ratio(freq)
            self._raw_volume = _rms_to_volume(rms)
            self._glow = min(rms * 5, 1.0)
        else:
            self._raw_pitch = 0.0
            self._raw_volume = 0.5
            self._glow = 0.0

        if vib_rate > 0 and vib_extent > 0:
            self._raw_vib_amount = min(vib_extent * 8.0, 30.0)
            self._raw_vib_speed = min(vib_rate / 10.0, 1.0)
        else:
            self._raw_vib_amount = 0.0
            self._raw_vib_speed = 0.0

    def on_formant(self, f1, f2, f3):
        """포먼트 데이터로 모음 공간 업데이트 (마이크 모드)."""
        if np.isnan(f1) or np.isnan(f2) or np.isnan(f3):
            return
        if not self._vad_active:
            return

        xyz = formant_to_xyz(f1, f2, f3)
        pos = self._tracker.update(xyz)
        self._label = pos.label
        self._weights = pos.weights

        # SDF 렌더러에 가중치 전달
        if pos.weights:
            self._renderer.update_from_weights(pos.weights)
            self._cache_dirty = True

    def on_formant_direct(self, f1, f2, f3, pos):
        """시뮬레이션 모드: 이미 계산된 VowelPosition을 직접 전달."""
        self._label = pos.label
        self._weights = pos.weights
        if pos.weights:
            self._renderer.update_from_weights(pos.weights)
            self._cache_dirty = True

    def _tick(self):
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        # 스무딩
        self._pitch_ratio += SCALE_SMOOTH * (self._raw_pitch - self._pitch_ratio)
        self._volume_scale += VOLUME_SMOOTH * (self._raw_volume - self._volume_scale)
        self._vib_amount += VIBRATO_SMOOTH * (self._raw_vib_amount - self._vib_amount)
        self._vib_speed += VIBRATO_SMOOTH * (self._raw_vib_speed - self._vib_speed)

        if self._vad_active:
            self._opacity = min(self._opacity + dt * FADE_IN_SPEED, 1.0)
        else:
            self._opacity = max(self._opacity - dt * FADE_OUT_SPEED, 0.0)

        self._cache_dirty = True
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(0, 0, 0))

        # SDF 렌더
        qimg, glow_color, sx, sy = self._renderer.render(
            pitch_ratio=self._pitch_ratio,
            opacity=self._opacity,
            vib_amount=self._vib_amount,
            vib_speed=self._vib_speed,
        )

        # 글로우
        if self._glow > 0.01:
            cx, cy = w / 2, h / 2
            radius = min(w, h) * 0.3 * (self._volume_scale / 3.5)
            grad = QRadialGradient(QPointF(cx, cy), radius)
            gc = QColor(glow_color[0], glow_color[1], glow_color[2],
                        int(80 * self._glow))
            grad.setColorAt(0, gc)
            grad.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy), radius, radius)

        # 이미지
        margin = 40
        max_w, max_h = w - margin * 2, h - margin * 2
        fit = min(max_w / max(sx, 0.01), max_h / max(sy, 0.01))
        scale = fit * (self._volume_scale / 3.5)
        img_w = sx * scale
        img_h = sy * scale
        x0 = (w - img_w) / 2
        y0 = (h - img_h) / 2
        p.drawImage(QRectF(x0, y0, img_w, img_h),
                     qimg, QRectF(0, 0, GRID, GRID))

        # HUD 상단
        p.setPen(QPen(QColor(100, 100, 100), 1))
        p.setFont(QFont("Consolas", 10))
        vad_str = "VOICE" if self._vad_active else "---"
        freq_str = f"{self._freq:.0f}Hz" if self._freq > 0 else "---"
        hud_top = (f"[{vad_str}] {freq_str}  rms={self._rms:.3f}  "
                   f"vowel={self._label or '?'}  "
                   f"glyph={self._renderer.current_vowel}")
        p.drawText(QRectF(12, 8, w - 24, 20),
                    Qt.AlignmentFlag.AlignLeft, hud_top)

        # HUD 중단: 가중치
        if self._weights:
            top3 = sorted(self._weights.items(), key=lambda x: -x[1])[:3]
            w_str = '  '.join(f'{n}={v:.0%}' for n, v in top3)
            p.setPen(QPen(QColor(80, 80, 80), 1))
            p.setFont(QFont("Consolas", 9))
            p.drawText(QRectF(12, 28, w - 24, 20),
                        Qt.AlignmentFlag.AlignLeft, w_str)

        # HUD 하단
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.setFont(QFont("Consolas", 9))
        p.drawText(QRectF(12, h - 28, w - 24, 20),
                    Qt.AlignmentFlag.AlignLeft,
                    "ESC: quit  |  C: calibrate")
        p.end()


class MainWindow(QWidget):
    """메인 윈도우."""

    def __init__(self, renderer=None, capture=None, centroids=None):
        super().__init__()
        self.setWindowTitle("VoiceTypo")
        self.resize(900, 700)
        self.setStyleSheet("background: #000;")

        # 컴포넌트
        if renderer is None:
            renderer = SDFRenderer()
        self._renderer = renderer

        self._capture = capture
        self._owns_capture = capture is None
        if self._capture is None:
            self._capture = AudioCapture()

        self._bridge = AudioBridge()
        self._capture.add_listener(self._bridge.on_audio)

        # 캔버스
        self.canvas = VoiceTypoCanvas(self._renderer)
        if centroids:
            self.canvas.set_centroids(centroids)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)

        # 시그널 연결
        self._bridge.updated.connect(self.canvas.on_voice_data)
        self._bridge.formant_updated.connect(self.canvas.on_formant)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def start(self):
        if not self._capture.is_running:
            self._capture.start()

    def stop(self):
        if self._owns_capture and self._capture.is_running:
            self._capture.stop()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
        elif e.key() == Qt.Key.Key_C:
            self._open_calibration()
        else:
            super().keyPressEvent(e)

    def _open_calibration(self):
        """캘리브레이션 창 열기."""
        from .calibration_window import CalibrationWindow
        self._cal_win = CalibrationWindow(self._capture)
        self._cal_win.calibration_complete.connect(self._on_calibration_done)
        self._cal_win.show()
        self._cal_win.start()

    def _on_calibration_done(self, centroids):
        self.canvas.set_centroids(centroids)
        self._cal_win.close()

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)
