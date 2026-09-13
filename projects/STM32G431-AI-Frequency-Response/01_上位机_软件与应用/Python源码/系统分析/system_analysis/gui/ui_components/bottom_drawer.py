from __future__ import annotations

import html

from PySide6.QtCore import Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QLabel, QFrame, QProgressBar, QTabWidget, QTextEdit, QVBoxLayout, QWidget


class BottomDrawer(QWidget):
    """Bottom expert drawer for logs, reports, settings, and import/export tools."""

    _LEVEL_COLORS = {
        "INFO": "#5F6368",
        "WARN": "#F29900",
        "ERR": "#D93025",
        "ERROR": "#D93025",
        "SUCCESS": "#1E8E3E",
    }

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("bottomDrawerTabs")
        self.tabs.setMaximumHeight(220)
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane {
                background: #FFFFFF;
                border-top: 1px solid #E0E0E0;
            }
            QTabBar::tab {
                background: #FFFFFF;
                color: #5F6368;
                padding: 8px 14px;
                border: none;
                min-width: 92px;
            }
            QTabBar::tab:selected {
                color: #1A73E8;
                font-weight: 600;
                border-bottom: 2px solid #1A73E8;
            }
            QTabBar::tab:hover {
                background: #F8F9FA;
            }
            """
        )

        self._init_log_tab()
        self.tabs.addTab(self.log_tab_widget, "流水日志")
        self.tabs.addTab(self._placeholder("详细报告内容预留区..."), "详细报告")
        self.tabs.addTab(self._placeholder("高级参数预留区..."), "高级参数")
        self.tabs.addTab(self._placeholder("数据导入/导出预留区..."), "数据 I/O")
        layout.addWidget(self.tabs)

    def _init_log_tab(self) -> None:
        self.log_tab_widget = QWidget()
        layout = QVBoxLayout(self.log_tab_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setFrameShape(QFrame.Shape.NoFrame)
        self.text_log.setStyleSheet(
            """
            QTextEdit {
                background: #FAFAFA;
                border: 1px solid #EEEEEE;
                border-radius: 6px;
                padding: 6px;
                color: #202124;
            }
            """
        )
        mono_font = QFont("Consolas", 10)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self.text_log.setFont(mono_font)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                background-color: #E0E0E0;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #1A73E8;
                border-radius: 2px;
            }
            """
        )

        layout.addWidget(self.text_log, 1)
        layout.addWidget(self.progress_bar)

    def _placeholder(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("background: #FFFFFF; color: #5F6368; padding: 16px;")
        return label

    @Slot(str, str)
    def append_log(self, level: str, message: str) -> None:
        normalized_level = (level or "INFO").upper()
        color = self._LEVEL_COLORS.get(normalized_level, "#5F6368")
        escaped_level = html.escape(normalized_level)
        escaped_message = html.escape(str(message))

        cursor = self.text_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_log.setTextCursor(cursor)
        self.text_log.insertHtml(
            f"<span style='color:{color};'><b>[{escaped_level}]</b> {escaped_message}</span><br>"
        )
        self.text_log.ensureCursorVisible()

    @Slot(int)
    def update_progress(self, value: int) -> None:
        self.progress_bar.setValue(max(0, min(100, int(value))))

    def clear_logs(self) -> None:
        self.text_log.clear()
        self.progress_bar.setValue(0)
