from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class SweepParams:
    start_hz: float
    stop_hz: float
    step_hz: float
    amplitude_vpp: float


class SweepPanel(QWidget):
    """Left sidebar for manual sweep controls and recent session history."""

    sig_start_sweep = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(304)
        self._setup_ui()
        self._update_point_hint()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        controls = self._card("手动扫频设置")
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)
        controls.layout().addLayout(form)

        self.start_edit = self._line_edit("100")
        self.stop_edit = self._line_edit("20000")
        self.step_edit = self._line_edit("100")
        self.amp_edit = self._line_edit("1.2")
        self._add_field(form, 0, "起始", self.start_edit, "Hz")
        self._add_field(form, 1, "终止", self.stop_edit, "Hz")
        self._add_field(form, 2, "步进", self.step_edit, "Hz")
        self._add_field(form, 3, "幅值", self.amp_edit, "Vpp")

        self.point_hint = QLabel()
        self.point_hint.setStyleSheet("color: #5F6368; font-size: 12px;")
        controls.layout().addWidget(self.point_hint)

        self.btn_start = QPushButton("开始扫频分析")
        self.btn_start.setObjectName("primarySweepButton")
        self.btn_start.setStyleSheet(
            """
            QPushButton#primarySweepButton {
                min-height: 42px;
                background: #1A73E8;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-weight: 700;
                font-size: 14px;
            }
            QPushButton#primarySweepButton:hover { background: #1765CC; }
            QPushButton#primarySweepButton:pressed { background: #1558B0; }
            QPushButton#primarySweepButton:disabled { background: #AECBFA; }
            """
        )
        self.btn_start.clicked.connect(self._emit_start)
        controls.layout().addWidget(self.btn_start)

        for edit in (self.start_edit, self.stop_edit, self.step_edit, self.amp_edit):
            edit.textChanged.connect(self._update_point_hint)

        history = self._card("测量历史")
        self.history_list = QListWidget()
        self.history_list.setStyleSheet(
            """
            QListWidget {
                background: #FFFFFF;
                border: none;
                outline: none;
            }
            QListWidget::item {
                min-height: 44px;
                padding: 5px 8px;
                border-radius: 6px;
                color: #202124;
            }
            QListWidget::item:hover { background: #F8F9FA; }
            QListWidget::item:selected {
                background: #F8FAFE;
                color: #202124;
                border-left: 1px solid #1A73E8;
            }
            """
        )
        self._add_history_placeholder()
        history.layout().addWidget(self.history_list, 1)

        layout.addWidget(controls, 0)
        layout.addWidget(history, 1)

    def _card(self, title: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("outlinedCard")
        frame.setStyleSheet(
            """
            QFrame#outlinedCard {
                background: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
            """
        )
        card_layout = QVBoxLayout(frame)
        card_layout.setContentsMargins(16, 14, 16, 16)
        card_layout.setSpacing(12)
        label = QLabel(title)
        label.setStyleSheet("color: #202124; font-size: 14px; font-weight: 700;")
        card_layout.addWidget(label)
        return frame

    def _line_edit(self, value: str) -> QLineEdit:
        edit = QLineEdit(value)
        edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        edit.setStyleSheet(
            """
            QLineEdit {
                min-height: 32px;
                background: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                padding: 0 8px;
                color: #202124;
            }
            QLineEdit:focus {
                border: 1px solid #1A73E8;
                background: #FFFFFF;
            }
            """
        )
        return edit

    def _add_field(self, form: QGridLayout, row: int, label: str, edit: QLineEdit, unit: str) -> None:
        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: #5F6368; font-size: 12px;")
        unit_widget = QLabel(unit)
        unit_widget.setStyleSheet("color: #5F6368; font-size: 12px;")
        form.addWidget(label_widget, row, 0)
        form.addWidget(edit, row, 1)
        form.addWidget(unit_widget, row, 2)

    def _params_or_none(self) -> SweepParams | None:
        try:
            params = SweepParams(
                start_hz=float(self.start_edit.text().strip()),
                stop_hz=float(self.stop_edit.text().strip()),
                step_hz=float(self.step_edit.text().strip()),
                amplitude_vpp=float(self.amp_edit.text().strip()),
            )
        except ValueError:
            return None
        if params.start_hz <= 0 or params.stop_hz <= params.start_hz or params.step_hz <= 0 or params.amplitude_vpp <= 0:
            return None
        return params

    def _update_point_hint(self) -> None:
        params = self._params_or_none()
        if params is None:
            self.point_hint.setText("请输入有效扫频参数")
            self.point_hint.setStyleSheet("color: #D93025; font-size: 12px;")
            return
        points = int((params.stop_hz - params.start_hz) // params.step_hz) + 1
        if points > 200:
            self.point_hint.setText(f"预估点数：{points}，建议降低密度")
            self.point_hint.setStyleSheet("color: #F29900; font-size: 12px;")
        else:
            self.point_hint.setText(f"预估点数：{points}")
            self.point_hint.setStyleSheet("color: #5F6368; font-size: 12px;")

    def _emit_start(self) -> None:
        params = self._params_or_none()
        if params is None:
            self._update_point_hint()
            return
        self.sig_start_sweep.emit(
            {
                "start_hz": params.start_hz,
                "stop_hz": params.stop_hz,
                "step_hz": params.step_hz,
                "amplitude_vpp": params.amplitude_vpp,
            }
        )

    def set_busy(self, busy: bool) -> None:
        self.btn_start.setDisabled(busy)
        for edit in (self.start_edit, self.stop_edit, self.step_edit, self.amp_edit):
            edit.setDisabled(busy)

    def add_session(self, session: Any) -> None:
        diagnosis = getattr(session, "decision_label", "已测量")
        confidence = getattr(session, "confidence_text", "")
        created_at = getattr(session, "created_at", None)
        time_text = created_at.strftime("%H:%M:%S") if created_at is not None else "刚刚"
        suffix = f"  {confidence}" if confidence else ""
        item = QListWidgetItem(f"{time_text}    {diagnosis}{suffix}")
        self.history_list.insertItem(0, item)
        self.history_list.setCurrentItem(item)

    def _add_history_placeholder(self) -> None:
        for text in ("等待首次扫频", "Mock 数据将显示在这里"):
            item = QListWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.history_list.addItem(item)
