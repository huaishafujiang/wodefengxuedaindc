from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget


class TopBar(QWidget):
    """Top app bar for global controls and compact instrument status."""

    sig_stop_clicked = Signal()
    sig_refresh_clicked = Signal()
    sig_ping_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(56)
        self.setStyleSheet("background: #FFFFFF; border-bottom: 1px solid #E0E0E0;")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(12)

        title = QLabel("○  STM32G431 智能频响分析仪")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #202124; border: none;")
        layout.addWidget(title)
        layout.addSpacing(20)

        self.com_box = QComboBox()
        self.com_box.addItems(["COM13", "未检测到串口"])
        self.baud_box = QComboBox()
        self.baud_box.addItems(["115200", "9600"])

        combo_style = """
            QComboBox {
                min-height: 30px;
                padding: 3px 8px;
                border: 1px solid #DADCE0;
                border-radius: 6px;
                background: #FFFFFF;
                color: #202124;
            }
            QComboBox:hover { background: #F8F9FA; }
        """
        self.com_box.setStyleSheet(combo_style)
        self.baud_box.setStyleSheet(combo_style)
        layout.addWidget(self.com_box)
        layout.addWidget(self.baud_box)

        button_style = """
            QPushButton {
                min-height: 30px;
                padding: 0 12px;
                border: 1px solid #DADCE0;
                border-radius: 6px;
                background: #FFFFFF;
                color: #1A73E8;
                font-weight: 500;
            }
            QPushButton:hover { background: #F8F9FA; }
            QPushButton:pressed { background: #F1F3F4; }
        """
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.setStyleSheet(button_style)
        self.btn_refresh.clicked.connect(self.sig_refresh_clicked)
        layout.addWidget(self.btn_refresh)

        self.btn_ping = QPushButton("PING")
        self.btn_ping.setStyleSheet(button_style)
        self.btn_ping.clicked.connect(self.sig_ping_clicked)
        layout.addWidget(self.btn_ping)

        layout.addStretch(1)

        self.lbl_status = QLabel("状态：就绪")
        self.lbl_status.setStyleSheet("color: #5F6368; font-size: 13px; border: none;")
        layout.addWidget(self.lbl_status)

        self.lbl_identification = QLabel("识别：等待测量")
        self.lbl_identification.setStyleSheet("color: #5F6368; font-size: 13px; border: none;")
        layout.addWidget(self.lbl_identification)

        self.btn_stop = QPushButton("紧急停止 STOP")
        self.btn_stop.setStyleSheet(
            """
            QPushButton {
                min-height: 34px;
                padding: 0 16px;
                background: #D93025;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 700;
                font-size: 13px;
            }
            QPushButton:hover { background: #B3261E; }
            QPushButton:pressed { background: #8C1D18; }
            """
        )
        self.btn_stop.clicked.connect(self.sig_stop_clicked)
        layout.addWidget(self.btn_stop)

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(f"状态：{text}")

    def set_identification(self, text: str) -> None:
        self.lbl_identification.setText(f"识别：{text}")

    def selected_port(self) -> str:
        return self.com_box.currentText().strip()

    def selected_baudrate(self) -> int | None:
        try:
            return int(self.baud_box.currentText().strip())
        except ValueError:
            return None
