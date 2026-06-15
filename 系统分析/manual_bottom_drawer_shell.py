from __future__ import annotations

import os
import sys

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

from sweep_worker import SweepWorker
from ui_components.bottom_drawer import BottomDrawer


class TestShell(QMainWindow):
    """Manual smoke shell for BottomDrawer and SweepWorker wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.thread: QThread | None = None
        self.worker: SweepWorker | None = None
        self.setWindowTitle("Bottom Drawer & Worker Test")
        self.resize(680, 420)
        self.setStyleSheet("background-color: #FFFFFF;")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        self.btn_start = QPushButton("开始 Mock 扫频")
        self.btn_start.setStyleSheet(
            """
            QPushButton {
                min-height: 38px;
                padding: 0 14px;
                background: #1A73E8;
                color: white;
                font-weight: 600;
                border: none;
                border-radius: 6px;
            }
            QPushButton:disabled {
                background: #AECBFA;
            }
            """
        )
        self.btn_start.clicked.connect(self.start_worker)

        self.drawer = BottomDrawer()
        layout.addWidget(self.btn_start)
        layout.addStretch(1)
        layout.addWidget(self.drawer)
        self.setCentralWidget(root)

    def start_worker(self) -> None:
        if self.thread is not None:
            self.drawer.append_log("WARN", "已有 Mock 扫频任务正在执行。")
            return
        self.btn_start.setEnabled(False)
        self.drawer.clear_logs()
        self.drawer.append_log("INFO", "初始化 Worker 线程...")

        self.thread = QThread()
        self.worker = SweepWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(
            lambda: self.worker.run(
                {
                    "start": 100,
                    "stop": 20_000,
                    "step": 250,
                    "amplitude": 1.2,
                    "port": "COM13",
                    "baudrate": 115200,
                }
            )
        )
        self.worker.log_emitted.connect(self.drawer.append_log)
        self.worker.progress_updated.connect(self.drawer.update_progress)
        self.worker.session_completed.connect(self.on_session_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

    def on_session_done(self, session: object) -> None:
        is_valid = getattr(session, "is_valid", False)
        point_count = getattr(session, "point_count", 0)
        if is_valid:
            self.drawer.append_log("SUCCESS", f"Mock 扫频完成，Session Valid: True，点数: {point_count}")
        else:
            error = getattr(session, "error", "未知错误")
            self.drawer.append_log("ERR", f"Mock 扫频失败：{error}")

    def on_thread_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.worker = None
        self.thread = None


def main() -> int:
    app = QApplication(sys.argv)
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    window = TestShell()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
