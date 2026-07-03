from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import numpy as np


def _empty_array() -> np.ndarray:
    return np.asarray([], dtype=float)


@dataclass
class MeasurementSession:
    id: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    source: str = ""
    omega: np.ndarray = field(default_factory=_empty_array)
    magnitude: np.ndarray = field(default_factory=_empty_array)
    phase: np.ndarray = field(default_factory=_empty_array)
    raw_omega: np.ndarray = field(default_factory=_empty_array)
    raw_magnitude: np.ndarray = field(default_factory=_empty_array)
    raw_phase: np.ndarray = field(default_factory=_empty_array)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    expected_circuit: str = "Auto"
    serial_port: str = ""
    baudrate: int | None = None
    sweep_command: str = ""
    analysis_settings: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Any | None = None
    ai_diagnosis: Any | None = None
    report_lines: list[str] = field(default_factory=list)
    error: str | None = None
    remark: str = ""
    session_id: str = ""
    timestamp: float | None = None
    health_score: str = "Pending"
    confidence: float = 0.0
    raw_text: str = ""
    logs: list[str] = field(default_factory=list)
    control_compensation_report: Any | None = None

    def __post_init__(self) -> None:
        self.omega = np.asarray(self.omega, dtype=float).flatten()
        self.magnitude = np.asarray(self.magnitude, dtype=float).flatten()
        self.phase = np.asarray(self.phase, dtype=float).flatten()
        self.raw_omega = np.asarray(self.raw_omega, dtype=float).flatten()
        self.raw_magnitude = np.asarray(self.raw_magnitude, dtype=float).flatten()
        self.raw_phase = np.asarray(self.raw_phase, dtype=float).flatten()
        self.diagnostics = {
            str(key): np.asarray(value, dtype=float).flatten()
            for key, value in (self.diagnostics or {}).items()
        }
        if not self.session_id:
            self.session_id = str(self.id) if self.id else uuid4().hex[:12]
        if self.timestamp is None:
            self.timestamp = float(self.created_at.timestamp())
        self.logs = list(self.logs or [])

        if len(self.raw_omega) == 0:
            self.raw_omega = self.omega.copy()
        if len(self.raw_magnitude) == 0:
            self.raw_magnitude = self.magnitude.copy()
        if len(self.raw_phase) == 0:
            self.raw_phase = self.phase.copy()

    @property
    def point_count(self) -> int:
        return int(min(len(self.omega), len(self.magnitude), len(self.phase)))

    @property
    def magnitude_data(self) -> np.ndarray:
        return self.magnitude

    @magnitude_data.setter
    def magnitude_data(self, value: np.ndarray) -> None:
        self.magnitude = np.asarray(value, dtype=float).flatten()

    @property
    def phase_data_rad(self) -> np.ndarray:
        return self.phase

    @phase_data_rad.setter
    def phase_data_rad(self, value: np.ndarray) -> None:
        self.phase = np.asarray(value, dtype=float).flatten()

    @property
    def is_valid(self) -> bool:
        return self.error is None and len(self.omega) > 0

    @property
    def has_complete_arrays(self) -> bool:
        if self.error is not None:
            return False
        lengths = {len(self.omega), len(self.magnitude), len(self.phase)}
        return len(self.omega) > 0 and len(lengths) == 1

    @property
    def decision_label(self) -> str:
        if self.error:
            return "无效"
        if self.ai_diagnosis is not None:
            if isinstance(self.ai_diagnosis, dict):
                return str(self.ai_diagnosis.get("system_label", "已诊断"))
            return str(getattr(self.ai_diagnosis, "system_label", "已诊断"))
        if self.result is not None:
            return str(getattr(self.result, "system_order", "已分析"))
        return "分析中"

    @property
    def confidence_text(self) -> str:
        if self.ai_diagnosis is None:
            return f"{float(self.confidence):.2f}" if self.confidence > 0 else ""
        confidence = getattr(self.ai_diagnosis, "confidence", None)
        if confidence is None and isinstance(self.ai_diagnosis, dict):
            confidence = self.ai_diagnosis.get("confidence")
        if confidence is None:
            return ""
        return f"{float(confidence):.2f}"

    @property
    def remark_summary(self) -> str:
        text = " ".join((self.remark or "").split())
        if len(text) <= 24:
            return text
        return text[:24] + "..."

    def as_legacy_tuple(self):
        return self.omega, self.magnitude, self.phase, self.diagnostics
