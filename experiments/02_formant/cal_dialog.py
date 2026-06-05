"""cal_dialog.py — UI 내 캘리브레이션 다이얼로그

ui_window 가 띄우는 모달 dialog. 7 모음 × 2 takes 자동 진행.
- 큰 글자로 모음 표시
- 카운트다운 / 녹음 / 결과 시각 표시
- vowel-aware sanity check (Yoon ref ± 3σ 밖이면 자동 재녹음)
"""

import sys
import time
from pathlib import Path

import numpy as np
import joblib
import sounddevice as sd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QComboBox,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SAMPLE_RATE, FORMANT_CEILINGS
from formant_engine import FormantEngine


VOWELS          = ["아", "에", "이", "오", "우", "으", "어"]
RECORD_SEC      = 2.0
TAKES_PER_VOWEL = 2
SANITY_RETRY    = 3
RMS_MIN         = 0.005

STD_FLOOR = {"F1": 50.0, "F2": 100.0, "F3": 150.0}

CAL_PATH = Path(__file__).resolve().parent / "user_refs.pkl"

# Yoon 2015 reference centers (vowel-aware sanity check 용)
SANITY_REFS = {
    "female": {
        "아": (978, 100, 1397, 175),
        "에": (548, 100, 2125, 185),
        "이": (352,  78, 2787, 250),
        "오": (487,  88,  840, 148),
        "우": (367,  78,  660, 121),
        "으": (435,  90, 1404, 217),
        "어": (671, 109, 1212, 178),
    },
    "male": {
        "아": (831,  88, 1145, 143),
        "에": (466,  88, 1743, 152),
        "이": (299,  68, 2285, 205),
        "오": (414,  78,  689, 121),
        "우": (312,  68,  541, 100),
        "으": (370,  79, 1151, 178),
        "어": (570,  95,  994, 146),
    },
}
SANITY_TOL_SIGMA = 5.0   # 3.0→5.0: 마이크/방 색채로 포먼트가 살짝 틀어져도 통과


def _is_sane(vowel: str, gender: str, f1: float, f2: float) -> bool:
    f1_mu, f1_sd, f2_mu, f2_sd = SANITY_REFS[gender][vowel]
    return (abs(f1 - f1_mu) <= SANITY_TOL_SIGMA * f1_sd and
            abs(f2 - f2_mu) <= SANITY_TOL_SIGMA * f2_sd)


class CalibrationDialog(QDialog):
    """7 모음 × 2 takes 자동 캘리브레이션 다이얼로그."""

    def __init__(self, parent=None, device=None):
        super().__init__(parent)
        self.setWindowTitle("캘리브레이션")
        self.setModal(True)
        self.setMinimumSize(520, 460)
        self.setStyleSheet("background:#0d0d1a; color:#FFFFFF;")

        # 녹음에 쓸 입력 장치 (UI에서 고른 마이크; None이면 시스템 기본)
        self.device = device if device is not None else sd.default.device[0]
        self._win = parent          # 메인 윈도우 (레벨 미터는 메인 오디오 버퍼를 읽음)
        self._busy = False

        self.gender = "female"
        self.engine = FormantEngine()
        self.takes = {v: [] for v in VOWELS}    # 모음별 (F1,F2,F3) 리스트
        self.v_idx = 0
        self.t_idx = 0
        self.retry_left = SANITY_RETRY
        self.started = False
        self.user_refs = None

        # ── UI ──
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.lbl_title = QLabel("캘리브레이션")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setFont(QFont("Malgun Gothic", 16, QFont.Weight.Bold))
        layout.addWidget(self.lbl_title)

        # ── 마이크 선택 + 실시간 레벨 미터 ──
        from audio_stream import AudioStream
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("마이크:"))
        self.cmb_mic = QComboBox()
        self.cmb_mic.setFixedHeight(32)
        self._mic_ids = []
        sel = 0
        for i, (dev_id, name) in enumerate(AudioStream.get_input_devices()):
            self.cmb_mic.addItem(f"[{dev_id}] {name}")
            self._mic_ids.append(dev_id)
            if dev_id == self.device:
                sel = i
        if self._mic_ids:
            self.cmb_mic.setCurrentIndex(sel)
            self.device = self._mic_ids[sel]
        self.cmb_mic.currentIndexChanged.connect(self._on_mic_changed)
        mrow.addWidget(self.cmb_mic)
        layout.addLayout(mrow)

        self.lbl_level = QLabel("레벨: ────────────────")
        self.lbl_level.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_level.setFont(QFont("Courier New", 12, QFont.Weight.Bold))
        self.lbl_level.setStyleSheet("color:#FFCC44;")
        layout.addWidget(self.lbl_level)
        self.lbl_level_hint = QLabel("말했을 때 막대가 차오르는 마이크를 고르세요")
        self.lbl_level_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_level_hint.setStyleSheet("color:#888; font-size:9pt;")
        layout.addWidget(self.lbl_level_hint)

        # 성별 선택 + 시작
        row = QHBoxLayout()
        self.cmb_gender = QComboBox()
        self.cmb_gender.addItems(["여성", "남성"])
        self.cmb_gender.setFixedHeight(36)
        row.addWidget(QLabel("성별:"))
        row.addWidget(self.cmb_gender)
        self.btn_start = QPushButton("시작")
        self.btn_start.setFixedHeight(36)
        self.btn_start.setStyleSheet(
            "background:#88FF88; color:#000; font-weight:bold;"
        )
        self.btn_start.clicked.connect(self._start)
        row.addWidget(self.btn_start)
        layout.addLayout(row)

        # 큰 모음 표시
        self.lbl_vowel = QLabel("준비")
        self.lbl_vowel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_vowel.setFont(QFont("Malgun Gothic", 80, QFont.Weight.Bold))
        self.lbl_vowel.setStyleSheet("color:#FFFF55;")
        self.lbl_vowel.setFixedHeight(140)
        layout.addWidget(self.lbl_vowel)

        # 상태 / 카운트다운
        self.lbl_status = QLabel("성별 선택 후 [시작]")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setFont(QFont("Malgun Gothic", 14))
        layout.addWidget(self.lbl_status)

        # 결과 (마지막 take)
        self.lbl_result = QLabel("")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setFont(QFont("Consolas", 11))
        self.lbl_result.setStyleSheet("color:#88BBFF;")
        self.lbl_result.setFixedHeight(24)
        layout.addWidget(self.lbl_result)

        # 진행 막대
        self.bar = QProgressBar()
        self.bar.setRange(0, len(VOWELS) * TAKES_PER_VOWEL)
        self.bar.setValue(0)
        self.bar.setFormat("%v / %m takes")
        layout.addWidget(self.bar)

        # 취소
        self.btn_cancel = QPushButton("취소 (학계 평균값 사용)")
        self.btn_cancel.setFixedHeight(32)
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(self.btn_cancel)

        # 레벨 미터 갱신 타이머 (메인 윈도우의 기존 오디오 버퍼를 읽음 → 추가 스트림 없음)
        self._level_timer = QTimer(self)
        self._level_timer.timeout.connect(self._update_level)
        self._level_timer.start(80)

    # ── 마이크 선택 / 레벨 미터 ──────────────────────────────
    def _on_mic_changed(self, idx: int) -> None:
        if not (0 <= idx < len(self._mic_ids)):
            return
        dev_id = self._mic_ids[idx]
        self.device = dev_id
        # 메인 윈도우 스트림도 같은 장치로 전환 (레벨 미터가 그 버퍼를 읽으므로)
        win = self._win
        if (win is not None and hasattr(win, "combo_device")
                and dev_id in getattr(win, "_device_ids", [])):
            win.combo_device.setCurrentIndex(win._device_ids.index(dev_id))

    def _update_level(self) -> None:
        if self._busy:
            return
        rms = 0.0
        win = self._win
        try:
            if win is not None and hasattr(win, "audio"):
                c = win.audio.get_chunk(int(0.05 * SAMPLE_RATE))
                if c is not None:
                    rms = float(np.sqrt(np.mean(c ** 2)))
        except Exception:
            pass
        bars = int(min(rms * 600, 18))
        self.lbl_level.setText("레벨: " + "█" * bars + "░" * (18 - bars) + f"  {rms:.4f}")
        self.lbl_level.setStyleSheet(
            "color:%s;" % ("#44FF88" if rms > 0.012 else "#FFCC44"))

    def done(self, r: int) -> None:
        try:
            self._level_timer.stop()
        except Exception:
            pass
        super().done(r)

    # ── flow ──────────────────────────────────────────────

    def _start(self) -> None:
        self.gender = "female" if self.cmb_gender.currentText() == "여성" else "male"
        self.cmb_gender.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.started = True
        QTimer.singleShot(400, self._next_vowel)

    def _next_vowel(self) -> None:
        if self.v_idx >= len(VOWELS):
            self._finish()
            return
        if self.t_idx >= TAKES_PER_VOWEL:
            self.v_idx += 1
            self.t_idx = 0
            self.retry_left = SANITY_RETRY
            self._next_vowel()
            return

        v = VOWELS[self.v_idx]
        self.lbl_vowel.setText(v)
        self.lbl_status.setText(
            f"take {self.t_idx + 1}/{TAKES_PER_VOWEL} · 곧 시작"
        )
        self.lbl_result.setText("")
        self._countdown_n = 3
        QTimer.singleShot(700, self._countdown)

    def _countdown(self) -> None:
        if self._countdown_n > 0:
            self.lbl_status.setText(f"{self._countdown_n} ...")
            self._countdown_n -= 1
            QTimer.singleShot(700, self._countdown)
        else:
            self.lbl_status.setText("● 녹음 중 (2초)")
            QTimer.singleShot(50, self._record)

    def _record(self) -> None:
        self._busy = True
        audio = sd.rec(int(RECORD_SEC * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", device=self.device)
        sd.wait()
        audio = audio[:, 0]
        self._busy = False
        self._process(audio)

    def _process(self, audio: np.ndarray) -> None:
        v = VOWELS[self.v_idx]
        rms = float(np.sqrt(np.mean(audio**2)))

        if rms < RMS_MIN:
            self._fail(f"음량 낮음 (RMS={rms:.4f})")
            return

        # force_extract=True: breathy 음성도 추출 (사용자가 분명 발음 중)
        res = self.engine.extract(audio, gender=self.gender,
                                  ceilings=FORMANT_CEILINGS,
                                  force_extract=True)
        f1, f2, f3 = res.get("f1"), res.get("f2"), res.get("f3")

        if f1 is None or f2 is None or f3 is None:
            self._fail("포먼트 실패")
            return

        if not _is_sane(v, self.gender, f1, f2):
            # 추출은 됐고 음량도 충분하면, 재시도 소진 시 그래도 채택 (캘리브레이션 완수)
            ok = (f1, f2, f3) if rms > 0.015 else None
            self._fail(f"비정상 F1={f1:.0f} F2={f2:.0f}", extract=ok)
            return

        # 성공
        self.takes[v].append((f1, f2, f3))
        self.lbl_result.setText(
            f"✓ F1={f1:.0f}  F2={f2:.0f}  F3={f3:.0f}"
        )
        self.lbl_status.setText("좋아요")
        self.t_idx += 1
        self.bar.setValue(self.v_idx * TAKES_PER_VOWEL + self.t_idx)
        self.retry_left = SANITY_RETRY
        QTimer.singleShot(800, self._next_vowel)

    def _fail(self, reason: str, extract=None) -> None:
        v = VOWELS[self.v_idx]
        print(f"[cal] '{v}' take 실패: {reason}", flush=True)   # 터미널에 사유 노출
        self.retry_left -= 1
        if self.retry_left <= 0:
            # 재시도 소진 — 포먼트가 추출됐다면(extract) 그 값이라도 채택해 캘리브레이션 완수
            if extract is not None:
                self.takes[v].append(extract)
                self.lbl_result.setText(f"△ {reason} — 그래도 사용")
            else:
                self.lbl_result.setText(f"✗ {reason} — 이 모음 건너뜀")
            self.t_idx = TAKES_PER_VOWEL  # 다음 모음으로
            self.retry_left = SANITY_RETRY
        else:
            self.lbl_result.setText(f"✗ {reason} — 다시")
        QTimer.singleShot(900, self._next_vowel)

    def _finish(self) -> None:
        refs = {}
        for v, takes in self.takes.items():
            if len(takes) < 1:          # 1 take라도 있으면 사용 (2→1)
                continue
            arr = np.array(takes)
            f1, f2, f3 = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
            sd1 = max(arr[:, 0].std(), STD_FLOOR["F1"])
            sd2 = max(arr[:, 1].std(), STD_FLOOR["F2"])
            sd3 = max(arr[:, 2].std(), STD_FLOOR["F3"])
            refs[v] = (float(f1), float(sd1),
                       float(f2), float(sd2),
                       float(f3), float(sd3))

        if len(refs) < 4:
            self.lbl_status.setText(
                f"⚠ {len(refs)}/7 만 cal — 학계 _REFS 사용"
            )
            self.user_refs = None
            QTimer.singleShot(1200, self.reject)
            return

        joblib.dump(refs, str(CAL_PATH))
        self.user_refs = refs
        self.lbl_vowel.setText("✓")
        self.lbl_status.setText(f"완료 — {len(refs)}/7 모음 저장")
        QTimer.singleShot(900, self.accept)
