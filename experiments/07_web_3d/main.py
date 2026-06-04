"""VoiceTypo — 마이크 입력으로 한글 타이포그래피를 실시간 변형.

실행:
    python main.py          # 캘리브레이션 → 메인
    python main.py --sim    # 마이크 없이 시뮬레이션
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer


def main():
    parser = argparse.ArgumentParser(description='VoiceTypo')
    parser.add_argument('--sim', action='store_true',
                        help='시뮬레이션 모드 (마이크 없이)')
    args = parser.parse_args()

    app = QApplication(sys.argv)

    print("Loading glyphs...", flush=True)

    simulate = args.sim

    # 1단계: 캘리브레이션
    from voicetypo.ui.calibration_window import CalibrationWindow

    cal_win = CalibrationWindow(simulate=simulate)

    def on_cal_done(centroids):
        cal_win.close()

        if not simulate:
            save_path = os.path.join(os.path.dirname(__file__), 'calibration.npz')
            np.savez(save_path, **{v: c for v, c in centroids.items()})
            print(f"캘리브레이션 저장: {save_path}", flush=True)

        # 2단계: 메인 윈도우
        from voicetypo.sdf.renderer import SDFRenderer
        renderer = SDFRenderer()

        if simulate:
            _launch_sim_main(app, renderer, centroids)
        else:
            from voicetypo.ui.main_window import MainWindow
            win = MainWindow(renderer=renderer, centroids=centroids)
            win.show()
            win.start()
            app._main_win = win

    cal_win.calibration_complete.connect(on_cal_done)
    cal_win.show()
    cal_win.start()
    app._win = cal_win

    print("Ready.", flush=True)
    sys.exit(app.exec())


def _launch_sim_main(app, renderer, centroids):
    """시뮬레이션 메인 윈도우."""
    from PySide6.QtWidgets import QWidget, QHBoxLayout
    from voicetypo.ui.main_window import VoiceTypoCanvas
    from voicetypo.audio.formant import FormantSimulator
    from voicetypo.vowel.space import formant_to_xyz
    from voicetypo.vowel.tracker import VowelTracker

    win = QWidget()
    win.setWindowTitle("VoiceTypo — Simulation")
    win.resize(900, 700)
    win.setStyleSheet("background: #000;")

    canvas = VoiceTypoCanvas(renderer)
    if centroids:
        canvas.set_centroids(centroids)

    layout = QHBoxLayout(win)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(canvas, 1)

    sim = FormantSimulator(
        vowel_sequence=['아', '이', '우', '에', '오', '으', '어'],
        hold_frames=40, transition_frames=15,
    )
    tracker = VowelTracker(centroids=centroids)

    def sim_tick():
        f1, f2, f3 = sim.next_frame()
        xyz = formant_to_xyz(f1, f2, f3)
        pos = tracker.update(xyz)
        canvas.on_formant_direct(f1, f2, f3, pos)
        canvas.on_voice_data(220.0, 0.06, 0.0, 0.0, True)

    sim_timer = QTimer()
    sim_timer.timeout.connect(sim_tick)
    sim_timer.setInterval(20)
    sim_timer.start()

    win._sim_timer = sim_timer
    win.show()
    app._main_win = win


if __name__ == '__main__':
    main()
