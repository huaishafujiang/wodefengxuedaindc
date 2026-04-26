from __future__ import annotations

import re
import time
from typing import Protocol

import numpy as np

from measurement_model import MeasurementFrame


NUMBER_PATTERN = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")

NAME_PATTERNS = {
    "omega": re.compile(r"^\s*omega\s*=\s*\[.*\]\s*$", re.IGNORECASE),
    "magnitude": re.compile(r"^\s*Magnitude_data\s*=\s*\[.*\]\s*$", re.IGNORECASE),
    "phase": re.compile(r"^\s*Phase_data_rad\s*=\s*\[.*\]\s*$", re.IGNORECASE),
}

SEARCH_PATTERNS = {
    "omega": re.compile(r"omega\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "magnitude": re.compile(r"Magnitude_data\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "phase": re.compile(r"Phase_data_rad\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
}

DIAGNOSTIC_PATTERNS = {
    "input_rms_v": re.compile(r"Input_rms_v\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "output_rms_v": re.compile(r"Output_rms_v\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "input_dc_v": re.compile(r"Input_dc_v\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "output_dc_v": re.compile(r"Output_dc_v\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "clip_flags": re.compile(r"Clip_flags\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "valid_capture_count": re.compile(r"Valid_capture_count\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "actual_freq_hz": re.compile(r"Actual_freq_hz\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "adc_sample_rate_hz": re.compile(r"Adc_sample_rate_hz\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "dac_sample_rate_hz": re.compile(r"Dac_sample_rate_hz\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "magnitude_repeat_span_db": re.compile(
        r"Magnitude_repeat_span_db\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL
    ),
    "phase_repeat_span_rad": re.compile(
        r"Phase_repeat_span_rad\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL
    ),
    "input_pp_v": re.compile(r"Input_pp_v\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
    "output_pp_v": re.compile(r"Output_pp_v\s*=\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL),
}

FRAME_TRAFFIC_PREFIXES = ("OK", "DONE", "ERR ", "WARN ", "READY", "Commands:")


class SerialLike(Protocol):
    timeout: float | None

    def readline(self) -> bytes:
        ...


def parse_array_from_line(line: str) -> np.ndarray:
    nums = NUMBER_PATTERN.findall(line)
    if not nums:
        raise ValueError(f"未能从行中解析数字: {line[:120]}")
    return np.array([float(x) for x in nums], dtype=float)


def parse_diagnostics_from_text(text: str) -> dict[str, np.ndarray]:
    diagnostics: dict[str, np.ndarray] = {}
    for key, pattern in DIAGNOSTIC_PATTERNS.items():
        match = pattern.search(text)
        if match:
            diagnostics[key] = parse_array_from_line(match.group(1))
    return diagnostics


def parse_measurement_frame_from_text(text: str, source: str = "text") -> MeasurementFrame:
    out = {}
    raw_lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    for key, pattern in SEARCH_PATTERNS.items():
        match = pattern.search(text)
        if match:
            out[key] = parse_array_from_line(match.group(1))

    missing = {"omega", "magnitude", "phase"} - set(out)
    if missing:
        raise ValueError("未找到完整的 omega / Magnitude_data / Phase_data_rad 三组数组。")

    frame = MeasurementFrame(
        omega=out["omega"],
        magnitude=out["magnitude"],
        phase=out["phase"],
        diagnostics=parse_diagnostics_from_text(text),
        source=source,
        raw_lines=raw_lines,
    )
    frame.validate()
    return frame


def parse_arrays_from_text(text: str):
    frame = parse_measurement_frame_from_text(text)
    return frame.omega, frame.magnitude, frame.phase


def build_sweep_command(f_start: float, f_stop: float, f_step: float, amplitude_vpp: float) -> str:
    return f"SWEEP {f_start:g} {f_stop:g} {f_step:g} {amplitude_vpp:g}\n"


def decode_serial_line(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        return raw.decode(errors="ignore").strip()
    return str(raw).strip()


def is_frame_noise(line: str) -> bool:
    if not line:
        return True
    return line in ("OK", "DONE") or line.startswith(FRAME_TRAFFIC_PREFIXES)


def read_optional_diagnostics_from_serial(ser: SerialLike) -> dict[str, np.ndarray]:
    diagnostics: dict[str, np.ndarray] = {}
    original_timeout = ser.timeout
    try:
        ser.timeout = 0.8
        for _ in range(len(DIAGNOSTIC_PATTERNS) + 2):
            line = decode_serial_line(ser.readline())
            if not line:
                break
            parsed = parse_diagnostics_from_text(line)
            if parsed:
                diagnostics.update(parsed)
                ser.timeout = 0.8
                if len(diagnostics) >= len(DIAGNOSTIC_PATTERNS):
                    break
                continue
            if is_frame_noise(line):
                continue
            break
    finally:
        ser.timeout = original_timeout
    return diagnostics


def read_measurement_frame_from_serial(
    ser: SerialLike,
    stop_event,
    timeout_sec: float,
    *,
    source: str = "serial",
) -> MeasurementFrame:
    deadline = time.monotonic() + max(float(timeout_sec), 1.0)
    while not stop_event.is_set() and time.monotonic() < deadline:
        line1 = decode_serial_line(ser.readline())
        if not line1 or not NAME_PATTERNS["omega"].match(line1):
            continue

        line2 = decode_serial_line(ser.readline())
        line3 = decode_serial_line(ser.readline())
        if not NAME_PATTERNS["magnitude"].match(line2) or not NAME_PATTERNS["phase"].match(line3):
            continue

        frame = MeasurementFrame(
            omega=parse_array_from_line(line1),
            magnitude=parse_array_from_line(line2),
            phase=parse_array_from_line(line3),
            diagnostics=read_optional_diagnostics_from_serial(ser),
            source=source,
            raw_lines=(line1, line2, line3),
        )
        frame.validate()
        return frame

    raise TimeoutError("等待测量帧超时。")
