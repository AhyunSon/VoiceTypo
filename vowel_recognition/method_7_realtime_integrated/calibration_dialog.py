"""
calibration_dialog.py — 개인 모음 보정 다이얼로그

사용자가 각 모음을 직접 발음하면 F1/F2를 측정해
개인 참조값으로 저장합니다.
"""

import json
import time
import numpy as np
from pathlib import Path

from mfcc_svm import MfccSvmClassifier

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QFont

CALIB_FILE   = Path(__file__).parent / "user_calibration.json"
VOWEL_ORDER  = ["아", "에", "이", "오", "우", "으", "어"]
VOWEL_COLORS = {
    "아": "#FF4444", "에": "#FFAA22", "이": "#FFFF44",
    "오": "#44FF88", "우": "#44DDFF", "으": "#4488FF", "어": "#CC55FF",
}
RECORD_SEC   = 2.0   # 각 모음 녹음 시간


# ── 녹음 · 포먼트 추출 스레드 ──────────────────────────────────
class _RecordThread(QThread):
    done = Signal(float, float, float, float, list, list)  # f1, f2, f1_sd, f2_sd, mfcc_chunks, proto_chunks

    def __init__(self, audio, engine, gender):
        super().__init__()
        self.audio  = audio
        self.engine = engine
        self.gender = gender

    def run(self):
        needed  = int(44100 * 0.30)   # 300ms 청크 (wav2vec2 최적 길이)
        f1_list, f2_list = [], []
        mfcc_chunks  = []   # SVM 학습용
        proto_chunks = []   # wav2vec2 프로토타입용
        t_end   = time.time() + RECORD_SEC

        while time.time() < t_end:
            chunk = self.audio.get_chunk(needed)
            if chunk is None:
                time.sleep(0.02)
                continue
            chunk = chunk - np.mean(chunk)
            rms   = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > 0.008:
                try:
                    res = self.engine.extract(chunk, self.gender)
                    if res.get("is_voiced", False):
                        rf1, rf2 = res.get("raw_f1"), res.get("raw_f2")
                        if (rf1 and rf2
                                and 80 < rf1 < 1500
                                and 300 < rf2 < 4000):
                            f1_list.append(float(rf1))
                            f2_list.append(float(rf2))
                        mfcc_chunks.append(chunk.copy())
                        proto_chunks.append(chunk.copy())  # wav2vec2 프로토타입용
                except Exception:
                    pass
            time.sleep(0.04)

        if len(f1_list) >= 4:
            f1s = sorted(f1_list)
            f2s = sorted(f2_list)
            lo  = len(f1s) // 5
            hi  = 4 * len(f1s) // 5 + 1
            f1_med = float(np.median(f1s[lo:hi]))
            f2_med = float(np.median(f2s[lo:hi]))
            f1_sd  = max(float(np.std(f1_list)), 50.0)
            f2_sd  = max(float(np.std(f2_list)), 100.0)
            self.done.emit(f1_med, f2_med, f1_sd, f2_sd, mfcc_chunks, proto_chunks)
        else:
            self.done.emit(0.0, 0.0, 0.0, 0.0, [], [])


# ── 다이얼로그 ─────────────────────────────────────────────────
class CalibrationDialog(QDialog):
    calibration_done = Signal(dict)   # 완료 시 emit (포먼트 데이터)

    def __init__(self, audio, engine, gender, svm_clf=None, wav2vec_clf=None, parent=None):
        super().__init__(parent)
        self.audio        = audio
        self.engine       = engine
        self.gender       = gender
        self.svm_clf      = svm_clf       # MfccSvmClassifier (선택)
        self.wav2vec_clf  = wav2vec_clf   # Wav2VecVowelClassifier (선택)
        self._data   = {}
        self._idx    = 0
        self._thread = None
        self._cd_val = 0

        self.setWindowTitle("개인 모음 보정")
        self.setMinimumSize(440, 380)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog   { background:#0d0d1a; color:#ddd; }
            QLabel    { color:#ddd; }
            QPushButton {
                background:#1a1a3a; color:#ddd; border:1px solid #333366;
                border-radius:5px; padding:8px 20px; font-size:11pt;
            }
            QPushButton:hover   { background:#222255; }
            QPushButton:disabled{ background:#111; color:#555; }
            QProgressBar {
                border:1px solid #333355; border-radius:4px;
                background:#111; height:12px;
            }
            QProgressBar::chunk { background:#4488FF; border-radius:4px; }
        """)
        self._build()
        self._show_step()

    def _build(self):
        v = QVBoxLayout(self)
        v.setSpacing(10)
        v.setContentsMargins(20, 16, 20, 16)

        # 진행 바
        self.bar = QProgressBar()
        self.bar.setMaximum(len(VOWEL_ORDER))
        v.addWidget(self.bar)

        self.lbl_prog = QLabel()
        self.lbl_prog.setAlignment(Qt.AlignCenter)
        self.lbl_prog.setStyleSheet("color:#888; font-size:9pt;")
        v.addWidget(self.lbl_prog)

        # 안내
        self.lbl_guide = QLabel()
        self.lbl_guide.setAlignment(Qt.AlignCenter)
        self.lbl_guide.setFont(QFont("Malgun Gothic", 12))
        v.addWidget(self.lbl_guide)

        # 모음 (크게)
        self.lbl_vowel = QLabel()
        self.lbl_vowel.setAlignment(Qt.AlignCenter)
        self.lbl_vowel.setFont(QFont("Malgun Gothic", 72, QFont.Weight.Bold))
        v.addWidget(self.lbl_vowel)

        # 카운트다운 / 결과
        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setFont(QFont("Courier New", 11))
        v.addWidget(self.lbl_status)

        # 버튼
        h = QHBoxLayout()
        self.btn_rec  = QPushButton("▶ 녹음 시작")
        self.btn_next = QPushButton("다음 →")
        self.btn_next.setEnabled(False)
        self.btn_rec.clicked.connect(self._start_countdown)
        self.btn_next.clicked.connect(self._advance)
        h.addWidget(self.btn_rec)
        h.addWidget(self.btn_next)
        v.addLayout(h)

        self.btn_finish = QPushButton("보정 완료 · 저장")
        self.btn_finish.setStyleSheet(
            "background:#224422; color:#88FF88; border:1px solid #44AA44;"
            "border-radius:5px; padding:8px; font-size:11pt;"
        )
        self.btn_finish.setEnabled(False)
        self.btn_finish.clicked.connect(self._save_and_close)
        v.addWidget(self.btn_finish)

        self._cd_timer = QTimer(self)
        self._cd_timer.timeout.connect(self._cd_tick)

    def _show_step(self):
        if self._idx >= len(VOWEL_ORDER):
            self._all_done()
            return
        vowel = VOWEL_ORDER[self._idx]
        color = VOWEL_COLORS.get(vowel, "#FFFF44")
        self.lbl_vowel.setText(vowel)
        self.lbl_vowel.setStyleSheet(f"color:{color};")
        self.lbl_guide.setText(f"「{vowel}」를 약 {RECORD_SEC:.0f}초 동안 길게 말하세요")
        self.lbl_prog.setText(f"  {self._idx + 1} / {len(VOWEL_ORDER)}  "
                              f"완료: {', '.join(self._data.keys()) or '없음'}")
        self.bar.setValue(self._idx)
        self.lbl_status.setText("")
        self.btn_rec.setText("▶ 녹음 시작")
        self.btn_rec.setEnabled(True)
        self.btn_next.setEnabled(self._idx in range(len(VOWEL_ORDER)))

    def _start_countdown(self):
        self.btn_rec.setEnabled(False)
        self.btn_next.setEnabled(False)
        self._cd_val = 3
        self.lbl_status.setText(f"준비 {self._cd_val}...")
        self.lbl_status.setStyleSheet("color:#FFCC44;")
        self._cd_timer.start(1000)

    def _cd_tick(self):
        self._cd_val -= 1
        if self._cd_val > 0:
            self.lbl_status.setText(f"준비 {self._cd_val}...")
        else:
            self._cd_timer.stop()
            self.lbl_status.setText(f"말하세요! ({RECORD_SEC:.0f}초)")
            self.lbl_status.setStyleSheet("color:#88FF88;")
            self._do_record()

    def _do_record(self):
        self._thread = _RecordThread(self.audio, self.engine, self.gender)
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _on_done(self, f1, f2, f1_sd, f2_sd, mfcc_chunks, proto_chunks):
        vowel = VOWEL_ORDER[self._idx]
        if f1 > 0 and f2 > 0:
            self._data[vowel] = {
                "f1": round(f1, 1), "f2": round(f2, 1),
                "f1_sd": round(f1_sd, 1), "f2_sd": round(f2_sd, 1),
            }
            # SVM용 MFCC 수집
            if self.svm_clf is not None and mfcc_chunks:
                for ch in mfcc_chunks:
                    self.svm_clf.calib_feed(vowel, ch)
            # wav2vec2 레이어-8 프로토타입 수집
            if self.wav2vec_clf is not None and proto_chunks:
                for ch in proto_chunks:
                    self.wav2vec_clf.add_prototype(vowel, ch)

            n_proto = len(proto_chunks)
            self.lbl_status.setText(
                f"측정 완료  F1={f1:.0f}  F2={f2:.0f}  AI샘플={n_proto}개"
            )
            self.lbl_status.setStyleSheet("color:#88FF88;")
            self.btn_rec.setText("↺ 다시 녹음")
        else:
            self.lbl_status.setText("인식 실패 — 더 크게 말해주세요")
            self.lbl_status.setStyleSheet("color:#FF8888;")
            self.btn_rec.setText("▶ 다시 시도")

        self.btn_rec.setEnabled(True)
        self.btn_next.setEnabled(True)

    def _advance(self):
        self._idx += 1
        self._show_step()

    def _all_done(self):
        self.lbl_guide.setText("모든 모음 보정 완료!")
        self.lbl_vowel.setText("✓")
        self.lbl_vowel.setStyleSheet("color:#88FF88;")
        self.bar.setValue(len(VOWEL_ORDER))
        self.btn_rec.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_finish.setEnabled(True)

    def _save_and_close(self):
        if self._data:
            CALIB_FILE.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # SVM 학습 + 저장
            if self.svm_clf is not None:
                ok = self.svm_clf.train()
                if ok:
                    self.svm_clf.save()
            # wav2vec2 프로토타입 확정 + 저장
            if self.wav2vec_clf is not None:
                self.wav2vec_clf.fit_prototypes()
                self.wav2vec_clf.save_prototypes()
            self.calibration_done.emit(self._data)
        self.accept()


# ── 저장/불러오기 유틸 ──────────────────────────────────────────
def load_calibration() -> dict:
    if CALIB_FILE.exists():
        try:
            return json.loads(CALIB_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}
