"""캘리브레이션 UI — pyqtgraph 3D 뷰.

사용자 주도 캘리브레이션. 상태별로 UI 전체가 확 바뀜:
  READY     → 어두운 화면, 큰 글씨 "SPACE를 누르세요"
  LISTENING → 노란 테두리 깜빡임, "발성을 기다리는 중..."
  RECORDING → 밝은 테두리, 궤적 + 프로그레스 바 채워짐
  CONFIRMED → 초록 배경, "확정!" 체크 표시
  COMPLETE  → 전체 초록
"""

import numpy as np
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QLabel, QProgressBar, QFrame)
from PySide6.QtGui import QFont

import pyqtgraph.opengl as gl
import pyqtgraph as pg

from ..audio.formant import FormantTracker, FormantSimulator, VOWEL_FORMANTS
from ..vowel.space import formant_to_xyz, DEFAULT_CENTROIDS
from ..vowel.calibrator import Calibrator, VOWELS

VOWEL_COLORS = {
    '아': (1.0, 0.2, 0.2, 1.0),
    '이': (0.2, 1.0, 0.2, 1.0),
    '우': (0.2, 0.2, 1.0, 1.0),
    '에': (1.0, 1.0, 0.2, 1.0),
    '오': (0.2, 1.0, 1.0, 1.0),
    '으': (1.0, 0.2, 1.0, 1.0),
    '어': (1.0, 0.6, 0.2, 1.0),
}

RMS_THRESHOLD = 0.008
SILENCE_FRAMES = 15

READY = 'ready'
LISTENING = 'listening'
RECORDING = 'recording'
CONFIRMED = 'confirmed'
COMPLETE = 'complete'

# 상태별 배경·테두리 스타일
STYLE_READY = "background: #111; border: 3px solid #333;"
STYLE_LISTENING = "background: #1a1a00; border: 3px solid #f0c040;"
STYLE_LISTENING_BLINK = "background: #1a1a00; border: 3px solid #332800;"
STYLE_RECORDING = "background: #001a0a; border: 3px solid #4ade80;"
STYLE_CONFIRMED = "background: #0a2010; border: 3px solid #22c55e;"
STYLE_COMPLETE = "background: #0a2010; border: 3px solid #22c55e;"


class CalibrationWindow(QWidget):
    calibration_complete = Signal(dict)

    def __init__(self, capture=None, simulate=False):
        super().__init__()
        self.setWindowTitle("VoiceTypo — 화자 캘리브레이션")
        self.resize(900, 700)

        self._simulate = simulate or (capture is None)
        self._capture = capture
        self._owns_capture = False

        if not self._simulate:
            if self._capture is None:
                from ..audio.capture import AudioCapture
                self._capture = AudioCapture()
                self._owns_capture = True
            self._formant_tracker = FormantTracker()
        else:
            self._sim = None
            self._sim_frame = 0

        self._calibrator = Calibrator()
        self._state = READY
        self._silence_count = 0
        self._blink_tick = 0

        # ── 전체 프레임 (테두리 색으로 상태 표시) ──
        self._frame = QFrame()
        self._frame.setStyleSheet(STYLE_READY)
        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setSpacing(4)
        frame_layout.setContentsMargins(16, 12, 16, 12)

        # 상단 상태 배너
        self._banner = QLabel()
        self._banner.setFont(QFont("맑은 고딕", 13, QFont.Weight.Bold))
        self._banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._banner.setFixedHeight(36)
        frame_layout.addWidget(self._banner)

        # 모음
        self._label_vowel = QLabel()
        self._label_vowel.setFont(QFont("맑은 고딕", 64, QFont.Weight.Bold))
        self._label_vowel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_vowel.setFixedHeight(100)
        frame_layout.addWidget(self._label_vowel)

        # 안내 텍스트
        self._label_status = QLabel()
        self._label_status.setFont(QFont("맑은 고딕", 16))
        self._label_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_status.setFixedHeight(36)
        frame_layout.addWidget(self._label_status)

        # 프로그레스 바
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(14)
        self._progress.setStyleSheet("""
            QProgressBar { background: #222; border: none; border-radius: 7px; }
            QProgressBar::chunk { background: #4ade80; border-radius: 7px; }
        """)
        frame_layout.addWidget(self._progress)

        # 포먼트 수치
        self._label_formants = QLabel()
        self._label_formants.setFont(QFont("Consolas", 10))
        self._label_formants.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_formants.setStyleSheet("color: #555;")
        frame_layout.addWidget(self._label_formants)

        # 3D 뷰
        self._view = gl.GLViewWidget()
        self._view.setCameraPosition(distance=4.0, elevation=25, azimuth=45)
        self._view.setBackgroundColor(0, 0, 0)
        frame_layout.addWidget(self._view, 1)

        # 3D 요소
        grid = gl.GLGridItem()
        grid.setSize(4, 4, 1)
        grid.setSpacing(0.5, 0.5, 0.5)
        grid.setColor((40, 40, 40, 80))
        self._view.addItem(grid)
        axis = gl.GLAxisItem()
        axis.setSize(1.5, 1.5, 1.5)
        self._view.addItem(axis)

        ref_pos, ref_colors = [], []
        for name, xyz in DEFAULT_CENTROIDS.items():
            ref_pos.append(xyz)
            c = VOWEL_COLORS.get(name, (0.5, 0.5, 0.5, 0.3))
            ref_colors.append((c[0], c[1], c[2], 0.25))
        self._view.addItem(gl.GLScatterPlotItem(
            pos=np.array(ref_pos), color=np.array(ref_colors),
            size=8, pxMode=True))

        self._trail_item = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)), color=(1, 1, 1, 0.4), width=1.5)
        self._view.addItem(self._trail_item)

        self._current_dot = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)), color=np.array([(1, 1, 1, 1)]),
            size=14, pxMode=True)
        self._view.addItem(self._current_dot)

        self._confirmed_gl_items = []

        # 하단 키 안내
        self._label_keys = QLabel()
        self._label_keys.setFont(QFont("Consolas", 10))
        self._label_keys.setStyleSheet("color: #666;")
        self._label_keys.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addWidget(self._label_keys)

        # 전체 레이아웃
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._frame)

        # 오디오 콜백
        self._latest_formants = (float('nan'),) * 3
        self._latest_rms = 0.0
        if not self._simulate and self._capture:
            self._capture.add_listener(self._on_audio)

        # 3D 뷰가 포커스 잡아먹는 것 방지
        self._view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # 타이머
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(20)

        self._update_ui()

    def start(self):
        if not self._simulate and self._capture and not self._capture.is_running:
            self._capture.start()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        if self._owns_capture and self._capture and self._capture.is_running:
            self._capture.stop()

    def _on_audio(self, chunk, sr):
        f1, f2, f3 = self._formant_tracker.update(chunk)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        self._latest_formants = (f1, f2, f3)
        self._latest_rms = rms

    def _tick(self):
        if self._state == COMPLETE:
            return

        self._blink_tick += 1

        if self._simulate:
            f1, f2, f3, rms = self._sim_get_formants()
        else:
            f1, f2, f3 = self._latest_formants
            rms = self._latest_rms

        voice_active = rms > RMS_THRESHOLD

        # 포먼트 수치
        if not np.isnan(f1):
            self._label_formants.setText(
                f"F1: {f1:.0f}   F2: {f2:.0f}   F3: {f3:.0f}   RMS: {rms:.4f}")
            self._label_formants.setStyleSheet("color: #888;")
        elif self._state in (LISTENING, RECORDING):
            self._label_formants.setText(f"RMS: {rms:.4f}")
            self._label_formants.setStyleSheet("color: #555;")

        # ── 상태 머신 ──
        if self._state == READY:
            pass
        elif self._state == LISTENING:
            if voice_active and not np.isnan(f1):
                self._state = RECORDING
                self._silence_count = 0
        elif self._state == RECORDING:
            if voice_active and not np.isnan(f1):
                self._silence_count = 0
                result = self._calibrator.feed(f1, f2, f3)
                if result is not None:
                    self._on_vowel_confirmed(result)
                    self._update_ui()
                    return
            else:
                self._silence_count += 1
                if self._silence_count >= SILENCE_FRAMES:
                    self._calibrator.retry()
                    self._state = READY
                    self._trail_item.setData(pos=np.zeros((1, 3)))
        elif self._state == CONFIRMED:
            pass

        # 3D
        if self._state == RECORDING:
            session = self._calibrator.current_session
            if session and len(session.trajectory) > 1:
                vowel = self._calibrator.current_vowel
                c = VOWEL_COLORS.get(vowel, (1, 1, 1, 0.6))
                self._trail_item.setData(
                    pos=session.trajectory,
                    color=pg.mkColor(int(c[0]*255), int(c[1]*255),
                                     int(c[2]*255), 150))
            if not np.isnan(f1):
                xyz = formant_to_xyz(f1, f2, f3)
                vowel = self._calibrator.current_vowel
                c = VOWEL_COLORS.get(vowel, (1, 1, 1, 1))
                self._current_dot.setData(
                    pos=xyz.reshape(1, 3), color=np.array([c]))

        self._update_ui()

    def _on_vowel_confirmed(self, centroid):
        vowel = self._calibrator.current_vowel
        color = VOWEL_COLORS.get(vowel, (0.3, 1.0, 0.5, 1.0))
        dot = gl.GLScatterPlotItem(
            pos=centroid.reshape(1, 3),
            color=np.array([color]),
            size=20, pxMode=True)
        self._view.addItem(dot)
        self._confirmed_gl_items.append(dot)
        print(f"  [{vowel}] 확정: {centroid.round(3)}", flush=True)
        self._state = CONFIRMED

    def _on_space(self):
        if self._state == READY:
            self._state = LISTENING
            self._silence_count = 0
            self._calibrator.retry()
            self._trail_item.setData(pos=np.zeros((1, 3)))
            if self._simulate:
                self._start_sim_for_vowel()

        elif self._state == CONFIRMED:
            if self._calibrator.is_complete:
                self._state = COMPLETE
                self._on_complete()
            else:
                self._calibrator.advance()
                self._state = READY
                self._trail_item.setData(pos=np.zeros((1, 3)))

    def _update_ui(self):
        vowel = self._calibrator.current_vowel
        done, total = self._calibrator.progress

        # ── 상태별 전체 스타일 변경 ──
        if self._state == READY:
            self._frame.setStyleSheet(STYLE_READY)
            if vowel:
                c = VOWEL_COLORS.get(vowel, (1, 1, 1, 1))
                r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                self._label_vowel.setText(f"「{vowel}」")
                self._label_vowel.setStyleSheet(f"color: rgb({r},{g},{b});")
            self._banner.setText(f"  {done}/{total}  ")
            self._banner.setStyleSheet(
                "background: #222; color: #888; border-radius: 4px;")
            self._label_status.setText("SPACE를 누르면 녹음을 시작합니다")
            self._label_status.setStyleSheet("color: #888;")
            self._progress.setValue(0)
            self._progress.setStyleSheet("""
                QProgressBar { background: #222; border: none; border-radius: 7px; }
                QProgressBar::chunk { background: #555; border-radius: 7px; }
            """)
            self._label_keys.setText(
                "SPACE: 녹음 시작      S: 건너뛰기      ESC: 종료")

        elif self._state == LISTENING:
            # 깜빡이는 노란 테두리
            if (self._blink_tick // 12) % 2 == 0:
                self._frame.setStyleSheet(STYLE_LISTENING)
            else:
                self._frame.setStyleSheet(STYLE_LISTENING_BLINK)
            self._banner.setText(f"  {done}/{total}  ●  대기 중  ")
            self._banner.setStyleSheet(
                "background: #332800; color: #f0c040; border-radius: 4px;")
            self._label_status.setText("마이크에 대고 발성하세요...")
            self._label_status.setStyleSheet("color: #f0c040; font-size: 16px;")
            self._progress.setValue(0)
            self._progress.setStyleSheet("""
                QProgressBar { background: #332800; border: none; border-radius: 7px; }
                QProgressBar::chunk { background: #f0c040; border-radius: 7px; }
            """)
            self._label_keys.setText("ESC: 종료")

        elif self._state == RECORDING:
            self._frame.setStyleSheet(STYLE_RECORDING)
            session = self._calibrator.current_session
            prog = int(session.stability_progress * 100) if session else 0
            self._progress.setValue(prog)

            self._banner.setText(f"  {done}/{total}  ●  녹음 중  ")
            self._banner.setStyleSheet(
                "background: #0a3018; color: #4ade80; border-radius: 4px;")
            if prog > 0:
                self._label_status.setText(
                    f"안정 감지 중... {prog}%  —  멈추지 마세요!")
                self._label_status.setStyleSheet(
                    "color: #4ade80; font-size: 16px;")
            else:
                self._label_status.setText("녹음 중 — 길~~게 발성하세요")
                self._label_status.setStyleSheet(
                    "color: #60b0ff; font-size: 16px;")
            self._progress.setStyleSheet("""
                QProgressBar { background: #0a2010; border: none; border-radius: 7px; }
                QProgressBar::chunk { background: #4ade80; border-radius: 7px; }
            """)
            self._label_keys.setText("")

        elif self._state == CONFIRMED:
            self._frame.setStyleSheet(STYLE_CONFIRMED)
            self._banner.setText(f"  {done}/{total}  ✓  확정  ")
            self._banner.setStyleSheet(
                "background: #0a3018; color: #22c55e; border-radius: 4px;")
            if vowel:
                c = VOWEL_COLORS.get(vowel, (1, 1, 1, 1))
                r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                self._label_vowel.setText(f"「{vowel}」 ✓")
                self._label_vowel.setStyleSheet(f"color: rgb({r},{g},{b});")
            self._label_status.setText("SPACE를 눌러 다음 모음으로")
            self._label_status.setStyleSheet("color: #22c55e; font-size: 16px;")
            self._progress.setValue(100)
            self._progress.setStyleSheet("""
                QProgressBar { background: #0a2010; border: none; border-radius: 7px; }
                QProgressBar::chunk { background: #22c55e; border-radius: 7px; }
            """)
            self._label_keys.setText(
                "SPACE: 다음 모음      R: 재시도      ESC: 종료")

        elif self._state == COMPLETE:
            self._frame.setStyleSheet(STYLE_COMPLETE)
            self._label_vowel.setText("완료!")
            self._label_vowel.setStyleSheet("color: #22c55e;")
            self._banner.setText(f"  {done}/{total}  ✓  ")
            self._banner.setStyleSheet(
                "background: #0a3018; color: #22c55e; border-radius: 4px;")
            self._label_status.setText("캘리브레이션이 완료되었습니다")
            self._label_status.setStyleSheet("color: #22c55e; font-size: 16px;")
            self._progress.setValue(100)
            self._label_keys.setText("")

    # ── 시뮬레이션 ──

    def _start_sim_for_vowel(self):
        vowel = self._calibrator.current_vowel
        if vowel:
            self._sim = FormantSimulator(
                vowel_sequence=[vowel], hold_frames=999,
                transition_frames=1, noise_hz=8.0)
            self._sim_frame = 0

    def _sim_get_formants(self):
        if self._state in (LISTENING, RECORDING) and self._sim:
            self._sim_frame += 1
            f1, f2, f3 = self._sim.next_frame()
            if self._state == LISTENING and self._sim_frame > 5:
                rms = 0.05
            elif self._state == RECORDING:
                rms = 0.05
            else:
                rms = 0.001
            return f1, f2, f3, rms
        return float('nan'), float('nan'), float('nan'), 0.001

    # ── 이벤트 ──

    def _on_complete(self):
        self._update_ui()
        centroids = self._calibrator.centroids
        print(f"\n=== 캘리브레이션 완료 ===", flush=True)
        for v, c in centroids.items():
            print(f"  {v}: {c.round(3)}", flush=True)
        QTimer.singleShot(1500, lambda: self.calibration_complete.emit(centroids))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._on_space()
        elif e.key() == Qt.Key.Key_R:
            if self._state in (CONFIRMED, READY):
                self._calibrator.retry()
                self._state = READY
                self._trail_item.setData(pos=np.zeros((1, 3)))
        elif e.key() == Qt.Key.Key_S:
            if self._state in (READY, LISTENING, CONFIRMED):
                if not self._calibrator.is_complete:
                    vowel = self._calibrator.current_vowel
                    session = self._calibrator.current_session
                    if session and not session.is_confirmed:
                        default_xyz = DEFAULT_CENTROIDS[vowel]
                        session.force_confirm(default_xyz)
                        self._on_vowel_confirmed(default_xyz)
                        print(f"  [{vowel}] 건너뛰기 (기본값)", flush=True)
        elif e.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)
