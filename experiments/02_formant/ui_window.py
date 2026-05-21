"""
ui_window.py — 메인 GUI 창

레이아웃:
  상단: 상태 바 (캘리브레이션, 성별, F1/F2/F3, F0, RMS/VAD, 모음)
  좌측 70%: 시계열 그래프 (포먼트 scatter + F0 + RMS)
  우측 30%: F1/F2 모음 공간 (한국어 단모음 기준 좌표계)

StepByStep 대비 고급 기능:
  - 백그라운드 분석 스레드 (UI 끊김 방지)
  - F1/F2 모음 공간 실시간 시각화 (궤적 포함)
  - 신뢰도(confidence) 기반 점 크기/투명도
  - 적응형 노이즈 바닥 표시
"""

import os
import sys
import time
import queue
import threading
import collections
import numpy as np

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QSplitter, QComboBox,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont

from config import (
    SAMPLE_RATE, ANALYSIS_WIN_SEC, UPDATE_MS,
    DISPLAY_SECS, GENDER_THRESH_HZ, CALIB_SECS,
    PARAMS, VOWEL_REFS, VOWEL_REFS_MALE,
    HNR_VOICE_MIN,
)
from audio_stream     import AudioStream
from vad              import AdaptiveVAD
from formant_engine   import FormantEngine
from vowel_classifier import classify_vowel, set_user_refs
from speaker_tracker import SpeakerF0Tracker


MAX_PTS      = int(DISPLAY_SECS / (UPDATE_MS / 1000))
TRAIL_LEN    = 40   # 모음 공간 궤적 점 개수
AXIS_FONT    = pg.QtGui.QFont("Segoe UI", 9)


class RealtimePraatWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("실시간 포먼트 분석기 v2 — 한국어 단모음")
        self.setMinimumSize(1400, 860)

        # ── 상태 ──
        self.gender          = "female"
        self._vs_gender      = None      # 현재 모음공간에 표시 중인 성별
        self.start_time      = time.time()

        # ── 모음 EMA 확률 누적 (히스테리시스 + 빠른 전환) ──
        _VL = ["아", "에", "이", "오", "우", "으", "어"]
        self._VOWEL_LIST       = _VL
        self._VOWEL_IDX        = {v: i for i, v in enumerate(_VL)}
        self._vowel_ema        = np.zeros(len(_VL))
        self._vowel_display    = "?"
        self._EMA_ALPHA        = 0.40   # 새 프레임 반영 비율
        self._EMA_COMMIT       = 0.18   # 모음 표시 진입 임계값
        self._EMA_RELEASE      = 0.10   # '?' 전환 임계값 (히스테리시스)
        self._EMA_SWITCH       = 0.35   # 다른 모음으로 전환 임계값

        # ── 라이브 F1/F2 smoothing (Praat 청크간 노이즈 흡수) ──
        # 분류에 들어가는 F1/F2/F3 은 최근 N 프레임의 median.
        # 청크별 spurious peak (F0 harmonic 등) 자동 제거.
        import collections
        self._F1_BUF = collections.deque(maxlen=3)
        self._F2_BUF = collections.deque(maxlen=3)
        self._F3_BUF = collections.deque(maxlen=3)

        # ── 캘리브레이션 ──
        self.calib_done = False
        self.calib_rms  = []
        self._norm_tick = 0      # 자동 정규화 상태 표시용 카운터

        # ── VAD / 엔진 / 화자 트래커 ──
        self.vad     = AdaptiveVAD()
        self.engine  = FormantEngine()
        self.tracker = SpeakerF0Tracker()

        # ── 사용자 cal 자동 로드 (UI 시작 시점) ──
        # cal 파일 없으면 cal 다이얼로그를 표시 (after show event)
        self._load_user_refs()

        # ── 오디오 스트림 ──
        self.audio = AudioStream()

        # ── 분석 결과 큐 (백그라운드 → UI) ──
        self.result_q = queue.Queue(maxsize=10)
        self.running  = True

        # ── 시계열 데이터 큐 ──
        def dq(): return collections.deque(maxlen=MAX_PTS)
        self.q_t      = dq()
        self.q_f0     = dq()
        self.q_f1     = dq()
        self.q_f2     = dq()
        self.q_f3     = dq()
        self.q_rms    = dq()
        self.q_jitter = dq()

        # ── jitter EMA (프레임 간 안정화) ──
        self._jitter_ema       = 0.0
        self._JITTER_EMA_ALPHA = 0.25   # 느린 평활 — 순간 스파이크 억제

        # ── 목소리 특성 외부 접근용 (글자 변형 모듈 연동) ──
        # 항상 최신 voiced 프레임 값을 유지; 침묵 중에도 마지막 값 보존
        self.latest_voice_features: dict = {
            "f0":     None,   # Hz  — 현재 F0 (pyworld mean)
            "rms":    0.0,    # 0–1 — 현재 RMS
            "jitter": 0.0,    # %   — F0 local jitter
        }

        # ── 모음 공간 궤적 ──
        self.trail_f1 = collections.deque(maxlen=TRAIL_LEN)
        self.trail_f2 = collections.deque(maxlen=TRAIL_LEN)
        self.trail_conf = collections.deque(maxlen=TRAIL_LEN)

        self._build_ui()
        self.audio.start()

        # 초기 상태 — 캘리브레이션 대기
        self.lbl_calib.setText("노이즈 측정 중...")
        self.lbl_calib.setStyleSheet("color:#FFCC44;")

        # 백그라운드 분석 스레드 시작
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, daemon=True
        )
        self._analysis_thread.start()

        # UI 갱신 타이머
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(UPDATE_MS)

        # 레벨 미터 타이머 (캘리브레이션 중에도 표시)
        self._level_timer = QTimer(self)
        self._level_timer.timeout.connect(self._update_level)
        self._level_timer.start(80)

        # UI 표시 직후 cal 안 받은 상태면 다이얼로그 자동 표시
        QTimer.singleShot(600, self._maybe_show_cal_dialog)

    # ── cal 자동 로드 / 다이얼로그 ──────────────────────────
    def _load_user_refs(self) -> bool:
        from pathlib import Path
        cal_path = Path(__file__).resolve().parent / "user_refs.pkl"
        if not cal_path.exists():
            print("  [cal] 학계 _REFS 사용 (user_refs.pkl 없음)")
            return False
        try:
            import joblib
            user_refs = joblib.load(str(cal_path))
            set_user_refs(user_refs)
            print(f"  [cal] 사용자 refs 로드 ({len(user_refs)}/7 모음)")
            return True
        except Exception as e:
            print(f"  [cal] 로드 실패: {e}")
            return False

    def _maybe_show_cal_dialog(self) -> None:
        from pathlib import Path
        if (Path(__file__).resolve().parent / "user_refs.pkl").exists():
            return
        from cal_dialog import CalibrationDialog
        dlg = CalibrationDialog(self)
        if dlg.exec() and dlg.user_refs is not None:
            set_user_refs(dlg.user_refs)
            print(f"  [cal] 새 cal 완료 — {len(dlg.user_refs)}/7 모음")

    # ══════════════════════════════════════════
    # UI 구성
    # ══════════════════════════════════════════

    def _build_ui(self):
        pg.setConfigOptions(antialias=True, background="#0d0d1a")

        root = QWidget()
        self.setCentralWidget(root)
        main_vbox = QVBoxLayout(root)
        main_vbox.setSpacing(4)
        main_vbox.setContentsMargins(6, 6, 6, 6)

        # ── 마이크 선택 바 ──
        main_vbox.addLayout(self._build_device_bar())

        # ── 상단 상태 바 ──
        main_vbox.addLayout(self._build_status_bar())

        # ── 가운데: 시계열(좌) + 모음공간(우) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        self.glw_ts = pg.GraphicsLayoutWidget()
        self.glw_ts.setBackground("#0d0d1a")
        self._build_timeseries(self.glw_ts)

        self.glw_vs = pg.GraphicsLayoutWidget()
        self.glw_vs.setBackground("#0d0d1a")
        self._build_vowel_space(self.glw_vs)

        splitter.addWidget(self.glw_ts)
        splitter.addWidget(self.glw_vs)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        main_vbox.addWidget(splitter, stretch=1)

        self.setStyleSheet("""
            QMainWindow, QWidget { background:#0d0d1a; color:#ddd; }
            QGroupBox {
                border:1px solid #333355; border-radius:5px;
                margin-top:6px; padding-top:4px;
                color:#aac; font-size:9pt;
            }
            QGroupBox::title { subcontrol-origin:margin; left:8px; }
            QSplitter::handle { background:#222244; }
        """)

    def _build_device_bar(self):
        """마이크 장치 선택 + 실시간 레벨 미터"""
        bar = QHBoxLayout()
        bar.setSpacing(8)

        lbl = QLabel("마이크:")
        lbl.setStyleSheet("color:#aaa; font-size:9pt;")
        bar.addWidget(lbl)

        self.combo_device = QComboBox()
        self.combo_device.setStyleSheet(
            "QComboBox { background:#1a1a2e; color:#ddd; "
            "border:1px solid #333355; border-radius:3px; padding:2px 6px; "
            "font-size:9pt; min-width:320px; }"
            "QComboBox QAbstractItemView { background:#1a1a2e; color:#ddd; "
            "selection-background-color:#333366; }"
        )
        devices = AudioStream.get_input_devices()
        self._device_ids = []
        default_idx = 0
        import sounddevice as _sd
        default_device = _sd.default.device[0]
        for i, (dev_id, name) in enumerate(devices):
            self.combo_device.addItem(f"[{dev_id}] {name}")
            self._device_ids.append(dev_id)
            if dev_id == default_device:
                default_idx = i
        self.combo_device.setCurrentIndex(default_idx)
        self.combo_device.currentIndexChanged.connect(self._on_device_changed)
        bar.addWidget(self.combo_device)

        # 실시간 레벨 미터 레이블
        self.lbl_level = QLabel("레벨: --------")
        self.lbl_level.setStyleSheet(
            "color:#FFCC44; font-family:'Courier New'; font-size:9pt;"
        )
        bar.addWidget(self.lbl_level)

        bar.addStretch()
        return bar

    def _on_device_changed(self, idx):
        """장치 변경 시 오디오 스트림 재시작 + 상태 초기화"""
        dev_id = self._device_ids[idx]
        self._timer.stop()
        self._level_timer.stop()
        self.running = False

        # 분석 스레드가 종료될 때까지 잠깐 대기
        self._analysis_thread.join(timeout=0.5)

        # 상태 초기화
        self.calib_done = False
        self.calib_rms  = []
        self.start_time = time.time()
        self.vad    = AdaptiveVAD()
        self.engine = FormantEngine()
        self.result_q = queue.Queue(maxsize=10)
        for q in (self.q_t, self.q_f0, self.q_f1,
                  self.q_f2, self.q_f3, self.q_rms, self.q_jitter):
            q.clear()
        self.trail_f1.clear()
        self.trail_f2.clear()
        self.trail_conf.clear()

        self._vowel_ema        = np.zeros(len(self._VOWEL_LIST))
        self._vowel_display    = "?"
        self._vs_gender        = None
        self._jitter_ema       = 0.0
        self.latest_voice_features = {"f0": None, "rms": 0.0, "jitter": 0.0}
        self.tracker.reset()

        self.lbl_calib.setText("노이즈 측정 중...")
        self.lbl_calib.setStyleSheet("color:#FFCC44;")

        # 오디오 장치 교체
        self.audio.change_device(dev_id)

        # 분석 스레드 재시작
        self.running = True
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, daemon=True
        )
        self._analysis_thread.start()

        self._timer.start(UPDATE_MS)
        self._level_timer.start(80)

    def _update_level(self):
        """캘리브레이션 포함 항상 실시간 오디오 레벨 표시"""
        needed = int(SAMPLE_RATE * 0.05)  # 50ms 샘플
        chunk = self.audio.get_chunk(needed)
        if chunk is None:
            self.lbl_level.setText("레벨: [버퍼 없음]")
            return
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        bars = int(min(rms * 500, 20))
        thr  = self.vad.threshold
        bar_str = "#" * bars + "-" * (20 - bars)
        self.lbl_level.setText(
            f"레벨: |{bar_str}|  RMS={rms:.4f}  임계={thr:.4f}"
        )
        # 레벨이 임계값보다 높으면 초록, 아니면 노랑
        color = "#44FF88" if rms > thr else "#FFCC44"
        self.lbl_level.setStyleSheet(
            f"color:{color}; font-family:'Courier New'; font-size:9pt;"
        )

    def _build_status_bar(self):
        top = QHBoxLayout()
        top.setSpacing(6)

        def grp(title):
            g = QGroupBox(title)
            v = QVBoxLayout(g)
            v.setSpacing(2)
            return g, v

        def lbl(text, size=13, bold=True, color="#ddd"):
            l = QLabel(text)
            l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.setFont(QFont("Courier New", size,
                            QFont.Weight.Bold if bold else QFont.Weight.Normal))
            l.setStyleSheet(f"color:{color};")
            return l

        # 상태
        g, v = grp("상태")
        self.lbl_calib  = lbl("노이즈 측정 중...", 9, False, "#FFCC44")
        self.lbl_gender = lbl("여성", 16, True, "#FF77BB")
        self.lbl_clf    = lbl("분류기: 포먼트", 8, False, "#888888")
        v.addWidget(self.lbl_calib)
        v.addWidget(self.lbl_gender)
        v.addWidget(self.lbl_clf)
        top.addWidget(g, 1)

        # F1 / F2 / F3
        g2, v2 = grp("포먼트 (Hz)")
        hf = QHBoxLayout()
        self.lbl_f1 = lbl("---", 15, True, "#FF4444")
        self.lbl_f2 = lbl("---", 15, True, "#4488FF")
        self.lbl_f3 = lbl("---", 15, True, "#44CC55")
        for lx in [self.lbl_f1, self.lbl_f2, self.lbl_f3]:
            hf.addWidget(lx)
        v2.addLayout(hf)
        self.lbl_conf = lbl("신뢰도 ---", 8, False, "#888888")
        v2.addWidget(self.lbl_conf)
        top.addWidget(g2, 3)

        # 목소리 특성: 높낮이 / 크기 / 떨림
        g, v = grp("목소리 특성")
        # 높낮이 (F0)
        h_f0 = QHBoxLayout()
        h_f0.setSpacing(4)
        lbl_f0_tag = lbl("높낮이", 7, False, "#888888")
        self.lbl_f0 = lbl("---", 14, True, "#FFBB00")
        lbl_f0_hz   = lbl("Hz", 7, False, "#666666")
        h_f0.addWidget(lbl_f0_tag)
        h_f0.addWidget(self.lbl_f0)
        h_f0.addWidget(lbl_f0_hz)
        v.addLayout(h_f0)
        # 크기 (RMS) + VAD
        h_rms = QHBoxLayout()
        h_rms.setSpacing(4)
        lbl_rms_tag = lbl("크기", 7, False, "#888888")
        self.lbl_rms = lbl("---", 10, True, "#AAAAAA")
        self.lbl_vad = lbl("○", 10, True, "#666666")
        h_rms.addWidget(lbl_rms_tag)
        h_rms.addWidget(self.lbl_rms)
        h_rms.addWidget(self.lbl_vad)
        v.addLayout(h_rms)
        # 떨림 (jitter)
        h_jit = QHBoxLayout()
        h_jit.setSpacing(4)
        lbl_jit_tag = lbl("떨림", 7, False, "#888888")
        self.lbl_jitter = lbl("---", 10, True, "#AA88FF")
        lbl_jit_pct     = lbl("%", 7, False, "#666666")
        h_jit.addWidget(lbl_jit_tag)
        h_jit.addWidget(self.lbl_jitter)
        h_jit.addWidget(lbl_jit_pct)
        v.addLayout(h_jit)
        top.addWidget(g, 2)

        # 모음
        g, v = grp("추정 모음")
        self.lbl_vowel = lbl("?", 28, True, "#FFFF55")
        v.addWidget(self.lbl_vowel)
        top.addWidget(g, 1)

        return top

    def _plt(self, glw, row, title, ylabel, yrange):
        p = glw.addPlot(row=row, col=0)
        p.setTitle(f"<span style='color:#ccc;font-size:10pt'>{title}</span>")
        p.setLabel("left",   ylabel,    **{"color": "#999", "font-size": "9pt"})
        p.setLabel("bottom", "Time (s)", **{"color": "#999", "font-size": "9pt"})
        p.setYRange(*yrange, padding=0.05)
        p.showGrid(x=True, y=True, alpha=0.2)
        p.getAxis("bottom").setTickFont(AXIS_FONT)
        p.getAxis("left").setTickFont(AXIS_FONT)
        return p

    def _build_timeseries(self, glw):
        pen0 = pg.mkPen(None)

        # ── 포먼트 scatter ──
        self.plt_fmt = self._plt(glw, 0,
            "F1 / F2 / F3  [칼만 필터 + 멀티-세일링 + HNR 게이팅]",
            "Frequency (Hz)", (0, 4800))
        legend = self.plt_fmt.addLegend(offset=(10, 10))
        legend.setLabelTextColor("#ccc")

        self.sc_f1 = pg.ScatterPlotItem(
            size=9, pen=pen0, brush=pg.mkBrush(255, 60, 60, 220), name="F1")
        self.sc_f2 = pg.ScatterPlotItem(
            size=9, pen=pen0, brush=pg.mkBrush(60, 110, 255, 220), name="F2")
        self.sc_f3 = pg.ScatterPlotItem(
            size=9, pen=pen0, brush=pg.mkBrush(50, 210, 80, 220), name="F3")
        for sc in [self.sc_f1, self.sc_f2, self.sc_f3]:
            self.plt_fmt.addItem(sc)

        # 모음 F1 참조선 (수평 점선)
        for ref in VOWEL_REFS.values():
            mid = (ref["F1"][0] + ref["F1"][1]) / 2
            self.plt_fmt.addItem(pg.InfiniteLine(
                pos=mid, angle=0,
                pen=pg.mkPen(color=(60, 60, 80),
                             style=Qt.PenStyle.DotLine, width=1),
            ))

        # ── F0 ──
        self.plt_f0 = self._plt(glw, 1, "Pitch  F0", "F0 (Hz)", (50, 520))
        self.line_f0 = self.plt_f0.plot(
            pen=pg.mkPen(color=(255, 185, 0), width=2))
        self.plt_f0.addItem(pg.InfiniteLine(
            pos=GENDER_THRESH_HZ, angle=0,
            pen=pg.mkPen(color=(130, 130, 130),
                         style=Qt.PenStyle.DashLine, width=1),
            label=f"  {GENDER_THRESH_HZ} Hz (성별)",
            labelOpts=dict(color="#aaa", position=0.92,
                           fill=(20, 20, 20, 100)),
        ))

        # ── RMS ──
        self.plt_rms = self._plt(glw, 2,
            "Sound Level  RMS  [적응형 VAD 임계값]",
            "RMS Amplitude", (0, 0.6))
        self.line_rms = self.plt_rms.plot(
            pen=pg.mkPen(color=(150, 150, 150), width=1.5))
        self.vad_line = pg.InfiniteLine(
            pos=self.vad.threshold, angle=0,
            pen=pg.mkPen(color=(255, 90, 90),
                         style=Qt.PenStyle.DashLine, width=1),
            label="  VAD 임계값",
            labelOpts=dict(color="#FF8888", position=0.85,
                           fill=(20, 20, 20, 100)),
        )
        self.plt_rms.addItem(self.vad_line)

        # ── Jitter ──
        self.plt_jitter = self._plt(glw, 3,
            "F0 Jitter  떨림 [local absolute jitter %]",
            "Jitter (%)", (0, 8))
        self.line_jitter = self.plt_jitter.plot(
            pen=pg.mkPen(color=(170, 136, 255), width=1.5))
        # 색 구간 참조선
        for thresh, clr in [(1.0, "#44FF88"), (3.0, "#FFCC44"), (6.0, "#FF8844")]:
            self.plt_jitter.addItem(pg.InfiniteLine(
                pos=thresh, angle=0,
                pen=pg.mkPen(color=clr, style=Qt.PenStyle.DotLine, width=1),
            ))

        # 세로 비율
        ci = glw.ci
        ci.layout.setRowStretchFactor(0, 3)
        ci.layout.setRowStretchFactor(1, 1)
        ci.layout.setRowStretchFactor(2, 1)
        ci.layout.setRowStretchFactor(3, 1)

    def _build_vowel_space(self, glw):
        """F1/F2 모음 공간 플롯 (전통 음성학 좌표계)"""
        p = glw.addPlot(row=0, col=0)
        p.setTitle("<span style='color:#ccc;font-size:10pt'>"
                   "F1/F2 모음 공간 (한국어 단모음)</span>")
        p.setLabel("left",   "F1 (Hz)", **{"color": "#999", "font-size": "9pt"})
        p.setLabel("bottom", "F2 (Hz)", **{"color": "#999", "font-size": "9pt"})
        p.showGrid(x=True, y=True, alpha=0.2)
        p.getAxis("bottom").setTickFont(AXIS_FONT)
        p.getAxis("left").setTickFont(AXIS_FONT)

        # 전통 모음 차트: F2 역전 (높은 F2 = 전설모음 = 왼쪽)
        p.invertX(True)
        # F1 역전 (높은 F1 = 개모음 = 아래쪽)
        p.invertY(True)
        p.setXRange(400, 3300, padding=0.05)
        p.setYRange(150, 1100, padding=0.05)
        self.plt_vs = p

        # ── 성별별 모음 타원: {gender: [(curve, txt), ...]} ──
        self._vs_items = {"female": [], "male": []}
        theta = np.linspace(0, 2 * np.pi, 60)

        for gender_key, refs in [("female", VOWEL_REFS), ("male", VOWEL_REFS_MALE)]:
            for name, ref in refs.items():
                color = ref["color"]
                cx_f2 = (ref["F2"][0] + ref["F2"][1]) / 2
                cy_f1 = (ref["F1"][0] + ref["F1"][1]) / 2
                rx_f2 = (ref["F2"][1] - ref["F2"][0]) / 2
                ry_f1 = (ref["F1"][1] - ref["F1"][0]) / 2

                ex = cx_f2 + rx_f2 * np.cos(theta)
                ey = cy_f1 + ry_f1 * np.sin(theta)
                curve = p.plot(ex, ey, pen=pg.mkPen(color, width=1.2,
                                                     style=Qt.PenStyle.DashLine))

                txt = pg.TextItem(
                    name, color=color, anchor=(0.5, 0.5),
                    fill=pg.mkBrush(13, 13, 26, 210),
                )
                txt.setFont(QFont("Malgun Gothic", 16, QFont.Weight.Bold))
                txt.setPos(cx_f2, cy_f1)
                p.addItem(txt)

                self._vs_items[gender_key].append((curve, txt))

        # 초기 표시: female만 보이게
        self._set_vs_gender("female")

        # ── 현재 위치 궤적 (ScatterPlotItem으로 fade 처리) ──
        self.sc_trail = pg.ScatterPlotItem()
        p.addItem(self.sc_trail)

        # ── 현재 위치 (크고 밝은 점) ──
        self.sc_now = pg.ScatterPlotItem(
            size=22, symbol='o',
            pen=pg.mkPen('white', width=2),
            brush=pg.mkBrush(255, 255, 255, 200),
        )
        p.addItem(self.sc_now)

    # ══════════════════════════════════════════
    # 백그라운드 분석 루프
    # ══════════════════════════════════════════

    def _analysis_loop(self):
        needed = int(SAMPLE_RATE * ANALYSIS_WIN_SEC)

        while self.running:
            chunk = self.audio.get_chunk(needed)
            if chunk is None:
                time.sleep(0.01)
                continue

            # DC 오프셋 제거 (step04)
            chunk = chunk - np.mean(chunk)

            # 캘리브레이션 단계
            if not self.calib_done:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                self.calib_rms.append(rms)
                try:
                    self.result_q.put_nowait(
                        dict(calib=True, rms=rms)
                    )
                except queue.Full:
                    pass
                time.sleep(0.02)
                continue

            # VAD 판단 — pitch 범위는 항상 넓게 (50-500Hz)
            # 이유: gender="female" 기본값이면 pitch_floor=150이 되어
            #       초저음 남성(65-100Hz) 자기상관 탐색 범위를 벗어나 항상 침묵 판정.
            is_voice, rms = self.vad.check(
                chunk,
                pitch_lo=50.0,
                pitch_hi=500.0,
            )

            if not is_voice:
                try:
                    self.result_q.put_nowait(
                        dict(calib=False, is_voice=False,
                             rms=rms, f0=None,
                             f1=None, f2=None, f3=None,
                             hnr=None, confidence=0.0)
                    )
                except queue.Full:
                    pass
                time.sleep(0.01)
                continue

            # 포먼트 추출 — force_extract=True: 희미한/breathy 발음도 추출.
            # Praat 의 BW 필터가 quality 보장. confidence 가 시각화 alpha 로 활용.
            try:
                res = self.engine.extract(
                    chunk, self.gender,
                    ceilings=self.tracker.formant_ceilings,
                    force_extract=True,
                )
            except Exception:
                time.sleep(0.01)
                continue

            # 화자 트래커 갱신 + 성별 자동 전환
            if res["f0"] is not None:
                self.tracker.update(res["f0"])
                self.gender = self.tracker.praat_gender

            # pyworld가 무성음으로 판단한 경우 → 침묵으로 처리
            if not res.get("is_voiced", True):
                try:
                    self.result_q.put_nowait(
                        dict(calib=False, is_voice=False,
                             rms=rms, f0=res.get("f0"),
                             f1=None, f2=None, f3=None,
                             hnr=res.get("hnr"), confidence=0.0,
                             raw_f1=None, raw_f2=None, raw_f3=None)
                    )
                except queue.Full:
                    pass
                time.sleep(0.01)
                continue

            # pyworld 유성음 확인 후 HNR 이중 게이트
            hnr = res.get("hnr")
            is_voice_final = (hnr is None) or np.isnan(hnr) or (hnr >= HNR_VOICE_MIN)

            try:
                self.result_q.put_nowait(
                    dict(calib=False, is_voice=is_voice_final,
                         rms=rms,
                         trk_vtype=self.tracker.voice_type,
                         **res)
                )
            except queue.Full:
                pass

            time.sleep(0.01)

    # ══════════════════════════════════════════
    # UI 갱신 타이머
    # ══════════════════════════════════════════

    def _tick(self):
        # 큐에서 최신 결과만 사용 (쌓인 결과는 버림)
        result = None
        while True:
            try:
                result = self.result_q.get_nowait()
            except queue.Empty:
                break

        # 새 결과가 없어도 그래프는 계속 스크롤
        if result is None:
            self._redraw_graphs()
            return

        # ── 캘리브레이션 진행 ──
        if result.get("calib"):
            elapsed = time.time() - self.start_time
            remain  = max(0.0, CALIB_SECS - elapsed)
            self.lbl_calib.setText(f"노이즈 측정 중... {remain:.1f}s")
            if elapsed >= CALIB_SECS and self.calib_rms:
                self.vad.calibrate(self.calib_rms)
                self.calib_done = True
                self.vad_line.setValue(self.vad.threshold)
                self.lbl_calib.setText(
                    f"보정 완료  noise={self.vad.noise_rms:.4f}"
                )
                self.lbl_calib.setStyleSheet("color:#88FF88;")
                self.engine.reset_kalman()
            return

        # ── 일반 결과 ──
        t        = time.time() - self.start_time
        iv       = result["is_voice"]
        f0       = result["f0"]
        f1       = result["f1"]
        f2       = result["f2"]
        f3       = result["f3"]
        rms      = result["rms"]
        conf     = result.get("confidence", 0.0)
        jitter   = result.get("jitter", 0.0)

        # ── jitter EMA 스무딩 ──
        if iv and jitter > 0.0:
            self._jitter_ema = ((1 - self._JITTER_EMA_ALPHA) * self._jitter_ema
                                + self._JITTER_EMA_ALPHA * jitter)
        else:
            self._jitter_ema *= 0.85   # 침묵 중 서서히 감쇠

        # ── 목소리 특성 외부 접근 갱신 ──
        if iv:
            self.latest_voice_features["f0"]     = f0
            self.latest_voice_features["rms"]    = rms
            self.latest_voice_features["jitter"] = self._jitter_ema

        # 큐에 삽입
        self.q_t.append(t)
        self.q_f0.append(f0  if f0  is not None else np.nan)
        self.q_f1.append(f1  if (f1 is not None and iv) else np.nan)
        self.q_f2.append(f2  if (f2 is not None and iv) else np.nan)
        self.q_f3.append(f3  if (f3 is not None and iv) else np.nan)
        self.q_rms.append(rms)
        self.q_jitter.append(self._jitter_ema if iv else np.nan)

        # ── 레이블 갱신 ──
        self.lbl_f0.setText(f"{f0:.1f}" if f0 else "---")
        self.lbl_f1.setText(f"{f1:.0f}" if (f1 and iv) else "---")
        self.lbl_f2.setText(f"{f2:.0f}" if (f2 and iv) else "---")
        self.lbl_f3.setText(f"{f3:.0f}" if (f3 and iv) else "---")
        self.lbl_rms.setText(f"{rms:.4f}")
        self.lbl_conf.setText(
            f"신뢰도 {conf*100:.0f}%" if iv else "신뢰도 ---"
        )

        # ── jitter 표시 (색상: 안정/보통/떨림) ──
        j = self._jitter_ema
        if not iv or j < 0.01:
            self.lbl_jitter.setText("---")
            self.lbl_jitter.setStyleSheet("color:#AA88FF;")
        else:
            self.lbl_jitter.setText(f"{j:.2f}")
            if j < 1.0:
                jcolor = "#44FF88"    # 안정
            elif j < 3.0:
                jcolor = "#FFCC44"    # 보통/가벼운 떨림
            elif j < 6.0:
                jcolor = "#FF8844"    # 강한 떨림/vibrato
            else:
                jcolor = "#FF4444"    # 매우 강한 tremolo
            self.lbl_jitter.setStyleSheet(f"color:{jcolor}; font-weight:bold;")

        if iv:
            self.lbl_vad.setText("●")
            self.lbl_vad.setStyleSheet("color:#88FF88;")
        else:
            self.lbl_vad.setText("○")
            self.lbl_vad.setStyleSheet("color:#666666;")

        # ── 목소리 유형 표시 (남성/여성/아동) ──
        trk_vtype = result.get("trk_vtype", "female")
        _VTYPE_LABEL = {"female": "여성", "male": "남성", "child": "아동"}
        _VTYPE_COLOR = {"female": "#FF77BB", "male": "#5599FF", "child": "#FF9944"}
        self.lbl_gender.setText(_VTYPE_LABEL.get(trk_vtype, "여성"))
        self.lbl_gender.setStyleSheet(
            f"color:{_VTYPE_COLOR.get(trk_vtype, '#FF77BB')};"
        )
        self._set_vs_gender(self.gender)   # 모음 공간 타원: male/female 전환

        # ── 모음 분류: 포먼트 Mahalanobis ──
        # 남성: male 참조값. 여성/아동: female 참조값.
        clf_gender = "male" if trk_vtype == "male" else "female"

        if iv and f1 is not None and f2 is not None:
            # 최근 3 프레임 median → spurious peak (Praat noise) 흡수
            raw_f3 = result.get("raw_f3")
            self._F1_BUF.append(f1)
            self._F2_BUF.append(f2)
            if raw_f3 is not None:
                self._F3_BUF.append(raw_f3)

            f1_clf = float(np.median(self._F1_BUF))
            f2_clf = float(np.median(self._F2_BUF))
            f3_clf = (float(np.median(self._F3_BUF))
                      if len(self._F3_BUF) > 0 else None)

            vowel_raw, v_conf = classify_vowel(
                f1_clf, f2_clf, clf_gender, f3=f3_clf, scale=1.0,
            )
        else:
            vowel_raw, v_conf = "?", 0.0

        # ── EMA + 히스테리시스로 안정적인 모음 표시 ──────────────────
        n_v = len(self._VOWEL_LIST)
        if iv and vowel_raw != "?" and v_conf > 0.15:
            prob = np.full(n_v, (1.0 - v_conf) / max(n_v - 1, 1))
            prob[self._VOWEL_IDX[vowel_raw]] = v_conf
            self._vowel_ema = ((1 - self._EMA_ALPHA) * self._vowel_ema
                               + self._EMA_ALPHA * prob)
        else:
            decay = 0.55 if not iv else 0.78
            self._vowel_ema *= decay

        best_idx    = int(np.argmax(self._vowel_ema))
        best_ema    = float(self._vowel_ema[best_idx])
        best_vowel  = self._VOWEL_LIST[best_idx]
        cur_vowel   = self._vowel_display

        if cur_vowel == "?":
            # 아직 표시 없음 → COMMIT 이상이면 표시
            if best_ema >= self._EMA_COMMIT:
                self._vowel_display = best_vowel
        elif best_vowel == cur_vowel:
            # 같은 모음 유지 → RELEASE 아래로 내려가야 '?'
            if best_ema < self._EMA_RELEASE:
                self._vowel_display = "?"
        else:
            # 다른 모음으로 전환 → SWITCH 이상 AND 현재보다 충분히 높아야
            cur_idx = self._VOWEL_IDX.get(cur_vowel, 0)
            cur_ema = float(self._vowel_ema[cur_idx])
            if best_ema >= self._EMA_SWITCH and best_ema > cur_ema * 1.25:
                self._vowel_display = best_vowel
            elif best_ema < self._EMA_RELEASE:
                self._vowel_display = "?"

        vowel = self._vowel_display
        self.lbl_vowel.setText(vowel)

        # 모음 컬러 적용
        refs_for_color = VOWEL_REFS if clf_gender == "female" else VOWEL_REFS_MALE
        v_color = refs_for_color.get(vowel, {}).get("color", "#FFFF55")
        self.lbl_vowel.setStyleSheet(f"color:{v_color};")

        # ── VAD 임계값 선 갱신 ──
        self.vad_line.setValue(self.vad.threshold)

        # ── 모음 공간: 음성 프레임만 궤적 추가 ──
        if iv and f1 is not None and f2 is not None:
            self.trail_f1.append(f1)
            self.trail_f2.append(f2)
            self.trail_conf.append(conf)

        self._redraw_graphs()

        # ── 분류기 상태 표시 (매 틱) ──
        self._update_clf_label()

        # ── 화자 분석 상태 표시 (~1초 간격) ──
        if self.calib_done:
            self._norm_tick += 1
            if self._norm_tick % 33 == 0:
                if not self.tracker.ready:
                    self.lbl_calib.setText(self.tracker.status())
                    self.lbl_calib.setStyleSheet("color:#FFCC44;")
                else:
                    self.lbl_calib.setText(
                        f"분석완료 {self.tracker.status()}"
                    )
                    self.lbl_calib.setStyleSheet("color:#44FFCC;")

    def _update_clf_label(self):
        """분류기 표시 (단순 baseline)"""
        self.lbl_clf.setText("분류기: 포먼트 Mahalanobis")
        self.lbl_clf.setStyleSheet("color:#88BBFF;")

    def _set_vs_gender(self, gender: str):
        """모음 공간 타원을 성별에 맞게 전환"""
        if gender == self._vs_gender:
            return
        for g, items in self._vs_items.items():
            visible = (g == gender)
            for curve, txt in items:
                curve.setVisible(visible)
                txt.setVisible(visible)
        self._vs_gender = gender

    def _redraw_graphs(self):
        """그래프 갱신 — 새 결과 없을 때도 매 tick 호출해 부드러운 스크롤 유지"""
        if not self.q_t:
            return

        # ── 배열 변환 ──
        t_arr      = np.array(self.q_t,      dtype=float)
        f0_arr     = np.array(self.q_f0,     dtype=float)
        f1_arr     = np.array(self.q_f1,     dtype=float)
        f2_arr     = np.array(self.q_f2,     dtype=float)
        f3_arr     = np.array(self.q_f3,     dtype=float)
        rms_arr    = np.array(self.q_rms,    dtype=float)
        jitter_arr = np.array(self.q_jitter, dtype=float)

        # X축 스크롤
        if len(t_arr) > 1:
            xmax = t_arr[-1]
            xmin = max(0.0, xmax - DISPLAY_SECS)
            for pl in (self.plt_fmt, self.plt_f0, self.plt_rms, self.plt_jitter):
                pl.setXRange(xmin, xmax, padding=0.01)

        # ── 시계열 scatter ──
        def _pts(arr, lo, hi):
            mask = np.isfinite(arr) & (arr > lo) & (arr < hi)
            return t_arr[mask], arr[mask]

        self.sc_f1.setData(*_pts(f1_arr, 100, 1300))
        self.sc_f2.setData(*_pts(f2_arr, 400, 3600))
        self.sc_f3.setData(*_pts(f3_arr, 1400, 5500))

        f0p = f0_arr.copy()
        f0p[f0p < 50] = np.nan
        self.line_f0.setData(t_arr, f0p)
        self.line_rms.setData(t_arr, rms_arr)
        self.line_jitter.setData(t_arr, jitter_arr)

        # ── 모음 공간 궤적 (confidence 기반 alpha) ──
        # 희미한 발음 = 낮은 confidence → 투명 점. 또렷한 발음 = 진한 점.
        if self.trail_f1:
            n   = len(self.trail_f1)
            tf1 = np.array(self.trail_f1)
            tf2 = np.array(self.trail_f2)
            tcf = np.array(self.trail_conf)

            age_alpha = np.linspace(30, 180, n)
            conf_mul  = np.clip(tcf * 1.5 + 0.3, 0.2, 1.0)
            alphas    = (age_alpha * conf_mul).astype(int)
            sizes     = np.linspace(3, 14, n)
            spots  = [
                {"pos":   (tf2[i], tf1[i]),
                 "size":  float(sizes[i]),
                 "brush": pg.mkBrush(255, 200, 100, int(alphas[i])),
                 "pen":   pg.mkPen(None)}
                for i in range(n)
            ]
            self.sc_trail.setData(spots=spots)
            self.sc_now.setData([tf2[-1]], [tf1[-1]])
        else:
            self.sc_trail.setData(spots=[])
            self.sc_now.setData([], [])

    # ══════════════════════════════════════════
    # 종료
    # ══════════════════════════════════════════

    def closeEvent(self, event):
        self.running = False
        self._timer.stop()
        self._level_timer.stop()
        self.audio.stop()
        event.accept()
