"""main.py — 실행 진입점

    python main.py

UI 시작 → 처음 실행이면 자동으로 캘리브레이션 다이얼로그 표시.
cal 다시 받기: python cal_setup.py --reset
cal 비활성화: rm user_refs.pkl
"""

import sys
from PySide6.QtWidgets import QApplication

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ui_window import RealtimePraatWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RealtimePraatWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
