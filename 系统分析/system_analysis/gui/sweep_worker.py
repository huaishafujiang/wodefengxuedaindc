from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

import numpy as np

from system_analysis.core.measurement_session import MeasurementSession

try:
    from PySide6.QtCore import QObject, Signal, Slot
except Exception:
    class _BoundSignal:
        """Small fallback used only for console tests when PySide6 is absent."""

        def __init__(self) -> None:
            self._slots: list[Callable[..., Any]] = []

        def connect(self, slot: Callable[..., Any]) -> None:
            self._slots.append(slot)

        def emit(self, *args: Any) -> None:
            for slot in list(self._slots):
                slot(*args)

    class _SignalDescriptor:
        def __set_name__(self, owner: type, name: str) -> None:
            self._name = f"__signal_{name}"

        def __get__(self, instance: object | None, owner: type) -> Any:
            if instance is None:
                return self
            signal = instance.__dict__.get(self._name)
            if signal is None:
                signal = _BoundSignal()
                instance.__dict__[self._name] = signal
            return signal

    class QObject:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            super().__init__()

    def Signal(*_types: Any) -> _SignalDescriptor:
        return _SignalDescriptor()

    def Slot(*_types: Any, **_kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


class SweepWorker(QObject):
    """Run a sweep task away from the GUI thread and return a MeasurementSession."""

    log_emitted = Signal(str, str)
    progress_updated = Signal(int)
    session_completed = Signal(object)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._stop_requested = threading.Event()

    @Slot()
    def request_stop(self) -> None:
        self._stop_requested.set()

    @Slot(dict)
    def run(self, params: dict[str, Any]) -> None:
        session = MeasurementSession(
            source=str(params.get("source", "mock sweep")),
            serial_port=str(params.get("port", "")),
            baudrate=_optional_int(params.get("baudrate")),
            sweep_command=str(params.get("sweep_command", "SWEEP")),
            raw_text="",
            status="running",
        )
        self._stop_requested.clear()

        try:
            self._log(session, "INFO", "开始扫频任务")
            self.progress_updated.emit(0)

            for percent in (10, 24, 38, 52, 66, 80):
                if self._stop_requested.is_set():
                    raise RuntimeError("用户已请求停止扫频")
                time.sleep(0.08)
                self.progress_updated.emit(percent)
                self._log(session, "INFO", f"采集进度 {percent}%")

            omega, magnitude, phase = _build_mock_response(params)
            session.omega = omega
            session.magnitude_data = magnitude
            session.phase_data_rad = phase
            session.raw_text = _format_three_array_frame(omega, magnitude, phase)
            session.health_score = "A"
            session.confidence = 0.97
            session.ai_diagnosis = {
                "system_label": "三阶高通系统",
                "confidence": session.confidence,
                "health_score": session.health_score,
            }
            session.status = "done"
            self.progress_updated.emit(100)
            self._log(session, "INFO", f"扫频完成，采样点数 {session.point_count}")
        except Exception as exc:
            session.error = str(exc)
            session.status = "error"
            self._log(session, "ERR", session.error)
        finally:
            self.session_completed.emit(session)
            self.finished.emit()

    def _log(self, session: MeasurementSession, level: str, message: str) -> None:
        session.logs.append(f"{level}: {message}")
        self.log_emitted.emit(level, message)


def _optional_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_param(params: dict[str, Any], *names: str, default: float) -> float:
    for name in names:
        if name not in params:
            continue
        try:
            return float(params[name])
        except (TypeError, ValueError):
            continue
    return default


def _build_mock_response(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_hz = max(_float_param(params, "start", "start_hz", "f_start", default=10.0), 0.1)
    stop_hz = max(_float_param(params, "stop", "stop_hz", "f_stop", default=100_000.0), start_hz)
    step_hz = max(_float_param(params, "step", "step_hz", "f_step", default=(stop_hz - start_hz) / 120.0), 0.1)
    amplitude = max(_float_param(params, "amplitude", "amplitude_vpp", "amp", default=1.0), 0.001)

    requested_points = int(math.floor((stop_hz - start_hz) / step_hz)) + 1
    point_count = min(max(requested_points, 16), 240)
    freq_hz = np.geomspace(start_hz, stop_hz, point_count)
    omega = 2.0 * np.pi * freq_hz

    corner_hz = math.sqrt(start_hz * stop_hz)
    x = np.maximum(freq_hz / corner_hz, 1e-9)
    magnitude = amplitude * (x**3 / np.sqrt(1.0 + x**6))
    phase = (1.5 * np.pi) - (3.0 * np.arctan(x))
    return omega, magnitude, phase


def _format_three_array_frame(omega: np.ndarray, magnitude: np.ndarray, phase: np.ndarray) -> str:
    def values(array: np.ndarray) -> str:
        return ",".join(f"{float(item):.9g}" for item in array)

    return "\n".join(
        [
            f"omega=[{values(omega)}]",
            f"Magnitude_data=[{values(magnitude)}]",
            f"Phase_data_rad=[{values(phase)}]",
        ]
    )
