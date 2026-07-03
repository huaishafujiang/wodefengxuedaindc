from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class DiagnosisPanel(QWidget):
    """Right sidebar that presents the AI diagnosis as a concise health card."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(328)
        self._setup_ui()
        self.show_empty_state()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.card = QFrame()
        self.card.setObjectName("diagnosisCard")
        self.card.setStyleSheet(
            """
            QFrame#diagnosisCard {
                background: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
            }
            """
        )
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(18, 16, 18, 18)
        card_layout.setSpacing(14)

        title = QLabel("AI 诊断结论")
        title.setStyleSheet("color: #202124; font-size: 14px; font-weight: 700;")
        card_layout.addWidget(title)

        self.result_label = QLabel("等待测量")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #202124; font-size: 26px; font-weight: 800;")
        card_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("置信度：-")
        self.confidence_label.setStyleSheet("color: #5F6368; font-size: 13px;")
        card_layout.addWidget(self.confidence_label)

        health_row = QHBoxLayout()
        self.health_dot = QLabel("●")
        self.health_dot.setStyleSheet("color: #9AA0A6; font-size: 18px;")
        self.health_label = QLabel("健康度：等待分析")
        self.health_label.setStyleSheet("color: #202124; font-size: 13px;")
        health_row.addWidget(self.health_dot)
        health_row.addWidget(self.health_label, 1)
        card_layout.addLayout(health_row)

        self.frequency_badge = QLabel("关键频率：-")
        self.frequency_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frequency_badge.setStyleSheet(
            """
            QLabel {
                background: #E8F0FE;
                color: #174EA6;
                border-radius: 8px;
                padding: 10px 12px;
                font-size: 18px;
                font-weight: 800;
            }
            """
        )
        card_layout.addWidget(self.frequency_badge)

        self.fault_label = QLabel("知识库故障：未触发")
        self.fault_label.setWordWrap(True)
        self.fault_label.setStyleSheet("color: #5F6368; font-size: 13px;")
        card_layout.addWidget(self.fault_label)

        card_layout.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.refresh_button = QPushButton("刷新诊断")
        self.refresh_button.setStyleSheet(
            """
            QPushButton {
                min-height: 36px;
                background: #FFFFFF;
                color: #1A73E8;
                border: 1px solid #DADCE0;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover { background: #F8F9FA; }
            """
        )
        self.smart_button = QPushButton("✨ 智能补扫")
        self.smart_button.setStyleSheet(
            """
            QPushButton {
                min-height: 36px;
                background: #1A73E8;
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                font-weight: 700;
            }
            QPushButton:hover { background: #1765CC; }
            """
        )
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.smart_button)
        card_layout.addLayout(actions)

        layout.addWidget(self.card, 1)

    def show_empty_state(self) -> None:
        self.result_label.setText("等待测量")
        self.confidence_label.setText("置信度：-")
        self.health_dot.setStyleSheet("color: #9AA0A6; font-size: 18px;")
        self.health_label.setText("健康度：等待分析")
        self.frequency_badge.setText("关键频率：-")
        self.fault_label.setText("知识库故障：未触发")

    def update_from_session(self, session: Any) -> None:
        if not bool(getattr(session, "is_valid", False)):
            self.result_label.setText("无效测量")
            self.confidence_label.setText("置信度：-")
            self.health_dot.setStyleSheet("color: #D93025; font-size: 18px;")
            self.health_label.setText("健康度：需复测")
            self.fault_label.setText(str(getattr(session, "error", "测量数据无效")))
            return

        diagnosis = getattr(session, "ai_diagnosis", None)
        if isinstance(diagnosis, dict):
            result = str(diagnosis.get("system_label", "已完成测量"))
            confidence = float(diagnosis.get("confidence", getattr(session, "confidence", 0.0)) or 0.0)
            health = str(diagnosis.get("health_score", getattr(session, "health_score", "A")))
        else:
            result = str(getattr(diagnosis, "system_label", getattr(session, "decision_label", "已完成测量")))
            confidence = float(getattr(diagnosis, "confidence", getattr(session, "confidence", 0.0)) or 0.0)
            health = str(getattr(session, "health_score", "A"))

        self.result_label.setText(result)
        self.confidence_label.setText(f"置信度：{confidence:.2f}")
        self.health_dot.setStyleSheet("color: #1E8E3E; font-size: 18px;")
        self.health_label.setText(f"健康度：{health} 可直接验收")
        self.frequency_badge.setText(f"截止频率：{_estimate_corner_hz(session)}")
        self.fault_label.setText("知识库故障：未触发严重故障")


def _estimate_corner_hz(session: Any) -> str:
    omega = np.asarray(getattr(session, "omega", []), dtype=float)
    magnitude = np.asarray(getattr(session, "magnitude_data", []), dtype=float)
    if len(omega) == 0 or len(magnitude) == 0:
        return "-"
    n = min(len(omega), len(magnitude))
    omega = omega[:n]
    magnitude = magnitude[:n]
    valid = np.isfinite(omega) & (omega > 0.0) & np.isfinite(magnitude) & (magnitude > 0.0)
    if not np.any(valid):
        return "-"
    omega = omega[valid]
    magnitude = magnitude[valid]
    target = float(np.max(magnitude) / np.sqrt(2.0))
    idx = int(np.argmin(np.abs(magnitude - target)))
    hz = float(omega[idx] / (2.0 * np.pi))
    if hz >= 1000:
        return f"{hz / 1000.0:.2f} kHz"
    return f"{hz:.1f} Hz"
