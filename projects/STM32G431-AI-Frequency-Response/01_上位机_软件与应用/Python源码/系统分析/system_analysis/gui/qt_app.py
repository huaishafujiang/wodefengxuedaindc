from __future__ import annotations

import sys

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from system_analysis.gui.sweep_worker import SweepWorker
from system_analysis.gui.ui_components.bottom_drawer import BottomDrawer
from system_analysis.gui.ui_components.diagnosis_panel import DiagnosisPanel
from system_analysis.gui.ui_components.plot_panel import PlotPanel
from system_analysis.gui.ui_components.sweep_panel import SweepPanel
from system_analysis.gui.ui_components.top_bar import TopBar


class InstrumentWindow(QMainWindow):
    """Material-style PySide6 instrument shell for validating the new UI route."""

    sig_stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.thread: QThread | None = None
        self.worker: SweepWorker | None = None
        self.setWindowTitle("STM32G431 智能频响分析仪")
        self.resize(1440, 900)
        self.setStyleSheet("QMainWindow { background: #FFFFFF; }")
        self._setup_ui()
        self._wire_signals()

    def _setup_ui(self) -> None:
        root = QWidget()
        root.setStyleSheet("background: #FFFFFF;")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        self.top_bar = TopBar()
        layout.addWidget(self.top_bar)

        body = QWidget()
        body.setStyleSheet("background: #FFFFFF;")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(18, 0, 18, 0)
        body_layout.setSpacing(18)

        self.sweep_panel = SweepPanel()
        self.plot_panel = PlotPanel()
        self.diagnosis_panel = DiagnosisPanel()

        body_layout.addWidget(self.sweep_panel)
        body_layout.addWidget(self.plot_panel, 1)
        body_layout.addWidget(self.diagnosis_panel)
        layout.addWidget(body, 1)

        self.bottom_drawer = BottomDrawer()
        layout.addWidget(self.bottom_drawer)
        self.setCentralWidget(root)

    def _wire_signals(self) -> None:
        self.sweep_panel.sig_start_sweep.connect(self.start_sweep_task)
        self.top_bar.sig_stop_clicked.connect(self.request_stop)
        self.top_bar.sig_refresh_clicked.connect(lambda: self.bottom_drawer.append_log("INFO", "刷新串口列表：当前为 Mock 模式"))
        self.top_bar.sig_ping_clicked.connect(lambda: self.bottom_drawer.append_log("INFO", "PING：Mock Worker 未连接真实设备"))

    def start_sweep_task(self, params: dict | None = None) -> None:
        if self.thread is not None:
            self.bottom_drawer.append_log("WARN", "扫频任务仍在执行，请等待完成或按 STOP。")
            return

        self.sweep_panel.set_busy(True)
        self.top_bar.set_status("扫频执行中...")
        self.top_bar.set_identification("采集中")
        self.bottom_drawer.clear_logs()
        self.plot_panel.show_empty_state()
        self.diagnosis_panel.show_empty_state()
        self.bottom_drawer.append_log("INFO", "初始化 Worker 线程并分配任务...")

        self.thread = QThread()
        self.worker = SweepWorker()
        self.worker.moveToThread(self.thread)

        sweep_params = dict(params or {})
        sweep_params.update(
            {
                "port": self.top_bar.selected_port(),
                "baudrate": self.top_bar.selected_baudrate(),
            }
        )
        self.thread.started.connect(lambda: self.worker.run(sweep_params))
        self.worker.log_emitted.connect(self.bottom_drawer.append_log)
        self.worker.progress_updated.connect(self.bottom_drawer.update_progress)
        self.worker.session_completed.connect(self.plot_panel.update_plots)
        self.worker.session_completed.connect(self.diagnosis_panel.update_from_session)
        self.worker.session_completed.connect(self.on_session_completed)
        self.sig_stop_requested.connect(self.worker.request_stop)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_thread_cleanup)
        self.thread.start()

    def on_session_completed(self, session: object) -> None:
        is_valid = bool(getattr(session, "is_valid", False))
        point_count = int(getattr(session, "point_count", 0))
        health_score = str(getattr(session, "health_score", "Pending"))
        diagnosis = getattr(session, "ai_diagnosis", None)
        label = "等待诊断"
        if isinstance(diagnosis, dict):
            label = str(diagnosis.get("system_label", label))
        elif diagnosis is not None:
            label = str(getattr(diagnosis, "system_label", label))

        if is_valid:
            self.top_bar.set_identification(label)
            self.sweep_panel.add_session(session)
            self.bottom_drawer.append_log("SUCCESS", f"测量完成：点数 {point_count}，健康度 {health_score}，判定 {label}")
        else:
            self.top_bar.set_identification("无效测量")
            error = getattr(session, "error", "未知错误")
            self.bottom_drawer.append_log("ERR", f"测量失败：{error}")

    def on_thread_cleanup(self) -> None:
        self.sweep_panel.set_busy(False)
        self.top_bar.set_status("就绪")
        self.worker = None
        self.thread = None

    def request_stop(self) -> None:
        if self.worker is None:
            self.bottom_drawer.append_log("WARN", "当前没有正在执行的扫频任务。")
            return
        self.sig_stop_requested.emit()
        self.top_bar.set_status("停止请求已发送")
        self.bottom_drawer.append_log("WARN", "发送紧急停止指令。")

    def closeEvent(self, event) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.request_stop()
            if not self.thread.wait(1500):
                self.bottom_drawer.append_log("ERR", "后台任务未能在关闭前及时退出。")
                event.ignore()
                return
        event.accept()


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = InstrumentWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
