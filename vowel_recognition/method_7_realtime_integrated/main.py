"""
main.py — 실행 진입점

실행:
    cd C:/Users/admin/Desktop/realtime_formant
    python main.py

의존 패키지:
    pip install praat-parselmouth sounddevice pyqtgraph PySide6
"""

import sys
from PySide6.QtWidgets import QApplication
from ui_window import RealtimePraatWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RealtimePraatWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
