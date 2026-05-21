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
SANITY_TOL_SIGMA = 3.0


def _is_sane(vowel: str, gender: str, f1: float, f2: float) -> bool:
    f1_mu, f1_sd, f2_mu, f2_sd = SANITY_REFS[gender][vowel]
    return (abs(f1 - f1_mu) <= SANITY_TOL_SIGMA * f1_sd and
            abs(f2 - f2_mu) <= SANITY_TOL_SIGMA * f2_sd)


class CalibrationDialog(QDialog):
    """7 모음 × 2 takes 자동 캘리브레이션 다이얼로그."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("캘리브레이션")
        self.setModal(True)
        self.setMinimumSize(520, 460)
        self.setStyleSheet("background:#0d0d1a; color:#FFFFFF;")

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
        audio = sd.rec(int(RECORD_SEC * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio[:, 0]
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
            self._fail(f"비정상 F1={f1:.0f} F2={f2:.0f}")
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

    def _fail(self, reason: str) -> None:
        self.retry_left -= 1
        if self.retry_left <= 0:
            self.lbl_result.setText(f"✗ {reason} — 이 모음 포기")
            self.t_idx = TAKES_PER_VOWEL  # 다음 모음으로
            self.retry_left = SANITY_RETRY
        else:
            self.lbl_result.setText(f"✗ {reason} — 다시")
        QTimer.singleShot(900, self._next_vowel)

    def _finish(self) -> None:
        refs = {}
        for v, takes in self.takes.items():
            if len(takes) < 2:
                continue
            arr = np.array(takes)
            f1, f2, f3 = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
            sd1 = max(arr[:, 0].std(), STD_FLOOR["F1"])
            sd2 = max(arr[:, 1].std(), STD_FLOOR["F2"])
            sd3 = max(arr[:, 2].std(), STD_FLOOR["F3"])
            refs[v] = (float(f1), float(sd1),
                       float(f2), float(sd2),
                       float(f3), float(sd3))

        if len(refs) < 5:
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
