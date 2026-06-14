from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from serial_protocol import build_sweep_command

try:
    from scipy.optimize import least_squares

    HAS_SCIPY_OPTIMIZE = True
except Exception:
    least_squares = None
    HAS_SCIPY_OPTIMIZE = False


CIRCUIT_TYPE_LABELS = {
    "lowpass": "低通",
    "highpass": "高通",
    "bandpass": "带通",
    "bandstop": "带阻",
    "unknown": "未知",
    "other": "未知",
}

ORDER_LABELS = {
    1: "一阶",
    2: "二阶",
    3: "三阶",
}

DEFAULT_KB_PATH = Path(__file__).with_name("diagnosis_knowledge_base.json")
NOISE_FLOOR_RMS_V = 0.004
LOW_INPUT_RMS_V = 0.03
ADC_BIAS_CENTER_V = 1.65


@dataclass(frozen=True)
class DiagnosisFeatures:
    point_count: int
    freq_min_hz: float
    freq_max_hz: float
    low_gain_db: float
    high_gain_db: float
    max_gain_db: float
    min_gain_db: float
    pass_stop_span_db: float
    peak_margin_db: float
    notch_margin_db: float
    head_slope_db_dec: float
    tail_slope_db_dec: float
    phase_span_deg: float
    minus3_crossing_count: int
    nyquist_min_distance: float
    peak_freq_hz: float
    notch_freq_hz: float
    input_rms_median_v: float | None = None
    output_rms_median_v: float | None = None
    input_dc_median_v: float | None = None
    output_dc_median_v: float | None = None
    input_clip_points: int = 0
    output_clip_points: int = 0
    clipped_points: int = 0
    min_valid_capture_count: int | None = None
    sweep_range_decades: float = 0.0


@dataclass(frozen=True)
class EquivalentTransferFunction:
    model_family: str
    order: int
    gain: float
    numerator: list[float]
    denominator: list[float]
    expression: str
    parameter_summary: str


@dataclass(frozen=True)
class TransferModelFit:
    model_family: str
    order: int
    parameters: dict[str, float]
    r_squared: float
    rmse_db: float
    residual_summary: str
    evidence: str
    success: bool = True
    transfer_function: EquivalentTransferFunction | None = None


@dataclass(frozen=True)
class DiagnosisCandidate:
    circuit_type: str
    order_estimate: int
    confidence: float
    key_frequency_hz: float | None
    evidence: str
    model_fit: TransferModelFit | None = None


@dataclass(frozen=True)
class FaultRule:
    rule_id: str
    name: str
    severity: str
    condition: str
    evidence_template: str
    suggestion: str
    confidence_penalty: float = 0.0


@dataclass(frozen=True)
class FaultFinding:
    rule_id: str
    name: str
    severity: str
    evidence: str
    suggestion: str
    confidence_penalty: float = 0.0


@dataclass(frozen=True)
class ActiveSweepStep:
    command: str
    reason: str


@dataclass(frozen=True)
class IntelligentDiagnosis:
    circuit_type: str
    circuit_label: str
    order_estimate: int
    system_label: str
    cutoff_frequency_hz: float | None
    center_frequency_hz: float | None
    bandwidth_hz: float | None
    left_cutoff_hz: float | None
    right_cutoff_hz: float | None
    confidence: float
    measurement_quality: list[str] = field(default_factory=list)
    possible_faults: list[str] = field(default_factory=list)
    next_test_suggestions: list[str] = field(default_factory=list)
    adaptive_sweep_commands: list[str] = field(default_factory=list)
    features: DiagnosisFeatures | None = None
    candidates: list[DiagnosisCandidate] = field(default_factory=list)
    best_fit: TransferModelFit | None = None
    model_fits: list[TransferModelFit] = field(default_factory=list)
    fault_findings: list[FaultFinding] = field(default_factory=list)
    active_sweep_plan: list[ActiveSweepStep] = field(default_factory=list)
    equivalent_transfer_function: EquivalentTransferFunction | None = None
    needs_resweep_confirmation: bool = False
    decision_strategy: str = ""


def _as_float_array(values) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    return np.asarray(values, dtype=float).flatten()


def _edge_count(length: int) -> int:
    return max(3, min(12, max(1, length // 8)))


def _db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.clip(np.asarray(values, dtype=float), 1.0e-12, None))


def _median_or_none(values: np.ndarray | None) -> float | None:
    if values is None or len(values) == 0:
        return None
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    return float(np.median(finite))


def _aligned_diagnostics(
    diagnostics: dict[str, np.ndarray] | None,
    valid_mask: np.ndarray,
    sort_idx: np.ndarray,
    original_len: int,
) -> dict[str, np.ndarray]:
    if not diagnostics:
        return {}

    aligned: dict[str, np.ndarray] = {}
    for name, values in diagnostics.items():
        arr = _as_float_array(values)
        if len(arr) >= original_len:
            aligned[name] = arr[:original_len][valid_mask][sort_idx]
        else:
            aligned[name] = arr
    return aligned


def _prepare_arrays(
    omega,
    magnitude,
    phase_rad,
    diagnostics: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    omega = _as_float_array(omega)
    magnitude = _as_float_array(magnitude)
    phase_rad = _as_float_array(phase_rad)
    n = min(len(omega), len(magnitude), len(phase_rad))
    if n == 0:
        return omega[:0], magnitude[:0], phase_rad[:0], {}

    omega = omega[:n]
    magnitude = magnitude[:n]
    phase_rad = phase_rad[:n]
    valid = (
        np.isfinite(omega)
        & np.isfinite(magnitude)
        & np.isfinite(phase_rad)
        & (omega > 0.0)
        & (magnitude > 0.0)
    )
    if not np.any(valid):
        return omega[:0], magnitude[:0], phase_rad[:0], {}

    sort_idx = np.argsort(omega[valid])
    aligned_diag = _aligned_diagnostics(diagnostics, valid, sort_idx, n)
    return (
        omega[valid][sort_idx],
        magnitude[valid][sort_idx],
        np.unwrap(phase_rad[valid][sort_idx]),
        aligned_diag,
    )


def _slope_db_per_dec(freq_hz: np.ndarray, mag_db: np.ndarray) -> np.ndarray:
    if len(freq_hz) < 2:
        return np.array([], dtype=float)
    log_f = np.log10(np.clip(freq_hz, 1.0e-12, None))
    delta_x = np.diff(log_f)
    delta_y = np.diff(mag_db)
    valid = np.abs(delta_x) > 1.0e-12
    if not np.any(valid):
        return np.array([], dtype=float)
    return delta_y[valid] / delta_x[valid]


def _find_crossings(freq_hz: np.ndarray, values_db: np.ndarray, target_db: float) -> list[tuple[float, int]]:
    crossings: list[tuple[float, int]] = []
    if len(freq_hz) < 2:
        return crossings

    for idx in range(len(freq_hz) - 1):
        y0 = float(values_db[idx])
        y1 = float(values_db[idx + 1])
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        e0 = y0 - target_db
        e1 = y1 - target_db
        if e0 == 0.0:
            crossings.append((float(freq_hz[idx]), idx))
            continue
        if e0 * e1 > 0.0 and e1 != 0.0:
            continue
        if y1 == y0:
            crossings.append((float(freq_hz[idx]), idx))
            continue
        frac = float(np.clip((target_db - y0) / (y1 - y0), 0.0, 1.0))
        log_f = math.log(max(float(freq_hz[idx]), 1.0e-12)) + frac * (
            math.log(max(float(freq_hz[idx + 1]), 1.0e-12))
            - math.log(max(float(freq_hz[idx]), 1.0e-12))
        )
        crossings.append((float(math.exp(log_f)), idx if frac < 0.5 else idx + 1))
    return crossings


def _order_from_slope(slope_db_dec: float) -> int:
    if not np.isfinite(slope_db_dec):
        return 1
    order = int(round(abs(float(slope_db_dec)) / 20.0))
    return int(np.clip(order, 1, 6))


def _nice_step_hz(raw_step: float) -> float:
    if not np.isfinite(raw_step) or raw_step <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    base = 10.0 ** exponent
    frac = raw_step / base
    for nice in (1.0, 2.0, 5.0, 10.0):
        if frac <= nice:
            return nice * base
    return 10.0 * base


def _round_frequency_hz(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    if value < 100.0:
        return round(value, 1)
    if value < 1000.0:
        return round(value / 5.0) * 5.0
    return round(value / 10.0) * 10.0


def _make_sweep_command(f_start: float, f_stop: float, step: float, amplitude_vpp: float = 0.6) -> str:
    f_start = max(1.0, _round_frequency_hz(f_start))
    f_stop = max(f_start + 1.0, _round_frequency_hz(f_stop))
    min_step = max((f_stop - f_start) / 180.0, 0.1)
    step = _nice_step_hz(max(step, min_step))
    return build_sweep_command(f_start, f_stop, step, amplitude_vpp).strip()


def _omega_attr_to_hz(obj: Any, attr: str) -> float | None:
    value = getattr(obj, attr, None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value / (2.0 * math.pi)


def _system_label(circuit_type: str, order: int) -> str:
    if circuit_type in ("unknown", "other"):
        return "未知系统"
    order_label = ORDER_LABELS.get(int(order), f"{int(order)}阶")
    return f"{order_label}{CIRCUIT_TYPE_LABELS.get(circuit_type, '未知')}系统"


def extract_diagnosis_features(
    omega,
    magnitude,
    phase_rad,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> DiagnosisFeatures:
    omega, magnitude, phase_rad, diag = _prepare_arrays(omega, magnitude, phase_rad, diagnostics)
    if len(omega) == 0:
        return DiagnosisFeatures(
            point_count=0,
            freq_min_hz=0.0,
            freq_max_hz=0.0,
            low_gain_db=0.0,
            high_gain_db=0.0,
            max_gain_db=0.0,
            min_gain_db=0.0,
            pass_stop_span_db=0.0,
            peak_margin_db=0.0,
            notch_margin_db=0.0,
            head_slope_db_dec=0.0,
            tail_slope_db_dec=0.0,
            phase_span_deg=0.0,
            minus3_crossing_count=0,
            nyquist_min_distance=0.0,
            peak_freq_hz=0.0,
            notch_freq_hz=0.0,
        )

    freq_hz = omega / (2.0 * math.pi)
    mag_db = _db(magnitude)
    edge = _edge_count(len(mag_db))
    low_gain_db = float(np.median(mag_db[:edge]))
    high_gain_db = float(np.median(mag_db[-edge:]))
    max_idx = int(np.argmax(mag_db))
    min_idx = int(np.argmin(mag_db))
    max_gain_db = float(mag_db[max_idx])
    min_gain_db = float(mag_db[min_idx])
    slope = _slope_db_per_dec(freq_hz, mag_db)
    slope_edge = _edge_count(len(slope)) if len(slope) else 1
    head_slope = float(np.median(slope[:slope_edge])) if len(slope) else 0.0
    tail_slope = float(np.median(slope[-slope_edge:])) if len(slope) else 0.0
    phase_span_deg = float(abs(np.degrees(phase_rad[-1] - phase_rad[0]))) if len(phase_rad) else 0.0
    h_jw = magnitude * np.exp(1j * phase_rad)
    nyquist_min_distance = float(np.min(np.abs(h_jw + 1.0 + 0.0j))) if len(h_jw) else 0.0
    pass_stop_span_db = float(max_gain_db - min_gain_db)
    peak_margin_db = float(min(max_gain_db - low_gain_db, max_gain_db - high_gain_db))
    notch_margin_db = float(min(low_gain_db - min_gain_db, high_gain_db - min_gain_db))
    minus3_crossings = _find_crossings(freq_hz, mag_db, max_gain_db - 3.0)
    sweep_range_decades = float(np.log10(freq_hz[-1] / max(freq_hz[0], 1.0e-12))) if len(freq_hz) > 1 else 0.0

    clip_flags = diag.get("clip_flags")
    input_clip_points = 0
    output_clip_points = 0
    clipped_points = 0
    if clip_flags is not None and len(clip_flags):
        flags = np.asarray(clip_flags, dtype=float)
        flags = flags[np.isfinite(flags)].astype(int)
        input_clip_points = int(np.count_nonzero((flags & 0x01) != 0))
        output_clip_points = int(np.count_nonzero((flags & 0x02) != 0))
        clipped_points = int(np.count_nonzero(flags))
    clip_point_count = diag.get("clip_point_count")
    if clip_point_count is not None and len(clip_point_count) >= 2:
        compact = np.asarray(clip_point_count, dtype=float).flatten()
        if np.all(np.isfinite(compact[:2])):
            input_clip_points = max(input_clip_points, int(round(float(compact[0]))))
            output_clip_points = max(output_clip_points, int(round(float(compact[1]))))
            clipped_points = max(clipped_points, input_clip_points + output_clip_points)

    valid_count = diag.get("valid_capture_count")
    min_valid_capture_count = None
    if valid_count is not None and len(valid_count):
        finite_valid = np.asarray(valid_count, dtype=float)
        finite_valid = finite_valid[np.isfinite(finite_valid)]
        if len(finite_valid):
            min_valid_capture_count = int(np.min(finite_valid))

    return DiagnosisFeatures(
        point_count=int(len(omega)),
        freq_min_hz=float(freq_hz[0]),
        freq_max_hz=float(freq_hz[-1]),
        low_gain_db=low_gain_db,
        high_gain_db=high_gain_db,
        max_gain_db=max_gain_db,
        min_gain_db=min_gain_db,
        pass_stop_span_db=pass_stop_span_db,
        peak_margin_db=peak_margin_db,
        notch_margin_db=notch_margin_db,
        head_slope_db_dec=head_slope,
        tail_slope_db_dec=tail_slope,
        phase_span_deg=phase_span_deg,
        minus3_crossing_count=len(minus3_crossings),
        nyquist_min_distance=nyquist_min_distance,
        peak_freq_hz=float(freq_hz[max_idx]),
        notch_freq_hz=float(freq_hz[min_idx]),
        input_rms_median_v=_median_or_none(diag.get("input_rms_v")),
        output_rms_median_v=_median_or_none(diag.get("output_rms_v")),
        input_dc_median_v=_median_or_none(diag.get("input_dc_v")),
        output_dc_median_v=_median_or_none(diag.get("output_dc_v")),
        input_clip_points=input_clip_points,
        output_clip_points=output_clip_points,
        clipped_points=clipped_points,
        min_valid_capture_count=min_valid_capture_count,
        sweep_range_decades=sweep_range_decades,
    )


def _classify_from_features(features: DiagnosisFeatures) -> tuple[str, int, float]:
    if features.point_count < 5:
        return "unknown", 1, 0.20

    low_to_high_db = features.low_gain_db - features.high_gain_db
    high_to_low_db = features.high_gain_db - features.low_gain_db
    peak_interior = (
        features.freq_min_hz * 1.25 < features.peak_freq_hz
        and features.peak_freq_hz < features.freq_max_hz / 1.25
    )
    notch_interior = (
        features.freq_min_hz * 1.25 < features.notch_freq_hz
        and features.notch_freq_hz < features.freq_max_hz / 1.25
    )

    if (
        peak_interior
        and features.peak_margin_db >= 3.0
        and features.head_slope_db_dec > 5.0
        and features.tail_slope_db_dec < -5.0
    ):
        order = max(
            2,
            _order_from_slope(features.head_slope_db_dec) + _order_from_slope(features.tail_slope_db_dec),
        )
        confidence = 0.55 + min(features.peak_margin_db / 24.0, 0.30)
        confidence += min(abs(features.head_slope_db_dec) + abs(features.tail_slope_db_dec), 80.0) / 400.0
        return "bandpass", int(np.clip(order, 1, 6)), float(np.clip(confidence, 0.0, 0.98))

    if notch_interior and features.notch_margin_db >= 3.0:
        order = 2
        confidence = 0.58 + min(features.notch_margin_db / 30.0, 0.32)
        if features.minus3_crossing_count >= 2:
            confidence += 0.06
        return "bandstop", order, float(np.clip(confidence, 0.0, 0.98))

    if low_to_high_db >= 6.0 or (low_to_high_db >= 3.0 and features.tail_slope_db_dec < -8.0):
        order = _order_from_slope(features.tail_slope_db_dec)
        confidence = 0.58 + min(low_to_high_db / 55.0, 0.28)
        confidence += min(abs(features.tail_slope_db_dec), 80.0) / 500.0
        return "lowpass", order, float(np.clip(confidence, 0.0, 0.98))

    if high_to_low_db >= 6.0 or (high_to_low_db >= 3.0 and features.head_slope_db_dec > 8.0):
        order = _order_from_slope(features.head_slope_db_dec)
        confidence = 0.58 + min(high_to_low_db / 55.0, 0.28)
        confidence += min(abs(features.head_slope_db_dec), 80.0) / 500.0
        return "highpass", order, float(np.clip(confidence, 0.0, 0.98))

    return "unknown", 1, 0.30


def _model_db_response(family: str, order: int, freq_hz: np.ndarray, params: np.ndarray) -> np.ndarray:
    gain_db = params[0]
    freq_hz = np.clip(freq_hz, 1.0e-12, None)
    if family == "lowpass":
        fc = max(float(params[1]), 1.0e-9)
        ratio = freq_hz / fc
        return gain_db - 10.0 * np.log10(1.0 + ratio ** (2 * order))
    if family == "highpass":
        fc = max(float(params[1]), 1.0e-9)
        ratio = fc / freq_hz
        return gain_db - 10.0 * np.log10(1.0 + ratio ** (2 * order))
    if family == "bandpass":
        f0 = max(float(params[1]), 1.0e-9)
        q = max(float(params[2]), 0.05)
        x = freq_hz / f0
        denom = np.sqrt(1.0 + (q * (x - 1.0 / x)) ** 2)
        return gain_db - 20.0 * np.log10(np.clip(denom, 1.0e-12, None))
    if family == "bandstop":
        f0 = max(float(params[1]), 1.0e-9)
        q = max(float(params[2]), 0.05)
        floor_db = float(params[3]) if len(params) >= 4 else -80.0
        floor_mag = float(np.clip(10.0 ** (floor_db / 20.0), 1.0e-6, 0.95))
        x = freq_hz / f0
        bandpass_mag = 1.0 / np.sqrt(1.0 + (q * (x - 1.0 / x)) ** 2)
        notch_mag = np.sqrt(np.clip(1.0 - bandpass_mag**2, 1.0e-12, None))
        notch_mag = np.sqrt(floor_mag**2 + (1.0 - floor_mag**2) * notch_mag**2)
        return gain_db + 20.0 * np.log10(np.clip(notch_mag, 1.0e-12, None))
    return np.full_like(freq_hz, gain_db, dtype=float)


def _poly_mul(left: list[float], right: list[float]) -> list[float]:
    result = [0.0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += float(a) * float(b)
    return result


def _format_coeff(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1.0e4 or (0.0 < abs(value) < 1.0e-3):
        return f"{value:.3e}"
    return f"{value:.4g}"


def _format_polynomial(coefficients: list[float]) -> str:
    degree = len(coefficients) - 1
    terms: list[str] = []
    for index, coeff in enumerate(coefficients):
        power = degree - index
        if abs(coeff) < 1.0e-12:
            continue
        coeff_text = _format_coeff(abs(coeff))
        if power == 0:
            term = coeff_text
        elif power == 1:
            term = "s" if math.isclose(abs(coeff), 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12) else f"{coeff_text}s"
        else:
            term = f"s^{power}" if math.isclose(abs(coeff), 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12) else f"{coeff_text}s^{power}"
        if not terms:
            terms.append(term if coeff >= 0 else f"-{term}")
        else:
            terms.append(("+ " if coeff >= 0 else "- ") + term)
    return " ".join(terms) if terms else "0"


def _low_high_denominator(order: int, wc: float) -> list[float]:
    if order == 1:
        return [1.0, wc]
    if order == 2:
        sqrt2 = math.sqrt(2.0)
        return [1.0, sqrt2 * wc, wc**2]
    if order == 3:
        # Third-order Butterworth denominator: (s + wc)(s^2 + wc*s + wc^2).
        return _poly_mul([1.0, wc], [1.0, wc, wc**2])
    denominator = [1.0]
    for _ in range(max(1, int(order))):
        denominator = _poly_mul(denominator, [1.0, wc])
    return denominator


def build_equivalent_transfer_function(fit: TransferModelFit) -> EquivalentTransferFunction | None:
    family = fit.model_family
    order = int(fit.order)
    gain_db = float(fit.parameters.get("gain_db", 0.0))
    gain = float(10.0 ** (gain_db / 20.0))

    if family in ("lowpass", "highpass"):
        fc_hz = fit.parameters.get("fc_hz")
        if fc_hz is None or fc_hz <= 0.0:
            return None
        wc = 2.0 * math.pi * float(fc_hz)
        denominator = _low_high_denominator(order, wc)
        if family == "lowpass":
            numerator = [gain * wc**order]
            summary = f"K={_format_coeff(gain)}, fc={fc_hz:.3g} Hz, wc={_format_coeff(wc)} rad/s"
        else:
            numerator = [gain] + [0.0] * order
            summary = f"K={_format_coeff(gain)}, fc={fc_hz:.3g} Hz, wc={_format_coeff(wc)} rad/s"
        expression = f"H(s) = ({_format_polynomial(numerator)}) / ({_format_polynomial(denominator)})"
        return EquivalentTransferFunction(
            model_family=family,
            order=order,
            gain=gain,
            numerator=[float(item) for item in numerator],
            denominator=[float(item) for item in denominator],
            expression=expression,
            parameter_summary=summary,
        )

    if family in ("bandpass", "bandstop"):
        f0_hz = fit.parameters.get("f0_hz")
        q = fit.parameters.get("q")
        if f0_hz is None or q is None or f0_hz <= 0.0 or q <= 0.0:
            return None
        w0 = 2.0 * math.pi * float(f0_hz)
        q = float(q)
        denominator = [1.0, w0 / q, w0**2]
        if family == "bandpass":
            numerator = [gain * w0 / q, 0.0]
            summary = (
                f"K={_format_coeff(gain)}, f0={f0_hz:.3g} Hz, "
                f"w0={_format_coeff(w0)} rad/s, Q={q:.3g}"
            )
        else:
            numerator = [gain, 0.0, gain * w0**2]
            summary = (
                f"K={_format_coeff(gain)}, f0={f0_hz:.3g} Hz, "
                f"w0={_format_coeff(w0)} rad/s, Q={q:.3g}"
            )
            floor_db = fit.parameters.get("notch_floor_db")
            if floor_db is not None:
                summary += f", notch_floor={floor_db:.3g} dB"
        expression = f"H(s) = ({_format_polynomial(numerator)}) / ({_format_polynomial(denominator)})"
        return EquivalentTransferFunction(
            model_family=family,
            order=order,
            gain=gain,
            numerator=[float(item) for item in numerator],
            denominator=[float(item) for item in denominator],
            expression=expression,
            parameter_summary=summary,
        )

    return None


def _fit_metrics(measured_db: np.ndarray, fitted_db: np.ndarray) -> tuple[float, float, np.ndarray]:
    residual = fitted_db - measured_db
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((measured_db - np.mean(measured_db)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-12 else 0.0
    rmse_db = float(np.sqrt(np.mean(residual**2)))
    return float(np.clip(r_squared, -1.0, 1.0)), rmse_db, residual


def _build_transfer_fit(
    family: str,
    order: int,
    params: np.ndarray,
    freq_hz: np.ndarray,
    measured_db: np.ndarray,
    success: bool = True,
) -> TransferModelFit:
    fitted_db = _model_db_response(family, order, freq_hz, params)
    r_squared, rmse_db, residual = _fit_metrics(measured_db, fitted_db)
    parameters = {"gain_db": float(params[0])}
    if family in ("lowpass", "highpass"):
        parameters["fc_hz"] = float(params[1])
        evidence = f"{CIRCUIT_TYPE_LABELS[family]}{order}阶模板: fc={params[1]:.1f} Hz, R2={r_squared:.3f}"
    else:
        parameters["f0_hz"] = float(params[1])
        parameters["q"] = float(params[2])
        parameters["bandwidth_hz"] = float(params[1] / max(params[2], 1.0e-9))
        if family == "bandstop" and len(params) >= 4:
            parameters["notch_floor_db"] = float(params[3])
        evidence = (
            f"{CIRCUIT_TYPE_LABELS[family]}二阶模板: f0={params[1]:.1f} Hz, "
            f"Q={params[2]:.2f}, R2={r_squared:.3f}"
        )

    residual_summary = f"RMSE={rmse_db:.2f} dB, 最大残差={float(np.max(np.abs(residual))):.2f} dB"
    fit = TransferModelFit(
        model_family=family,
        order=int(order),
        parameters=parameters,
        r_squared=r_squared,
        rmse_db=rmse_db,
        residual_summary=residual_summary,
        evidence=evidence,
        success=success,
    )
    return TransferModelFit(
        model_family=fit.model_family,
        order=fit.order,
        parameters=fit.parameters,
        r_squared=fit.r_squared,
        rmse_db=fit.rmse_db,
        residual_summary=fit.residual_summary,
        evidence=fit.evidence,
        success=fit.success,
        transfer_function=build_equivalent_transfer_function(fit),
    )


def _best_gain_for_shape(measured_db: np.ndarray, shape_db: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(measured_db) & np.isfinite(shape_db)
    if not np.any(finite):
        return 0.0, float("inf")
    gain_db = float(np.mean(measured_db[finite] - shape_db[finite]))
    residual = gain_db + shape_db[finite] - measured_db[finite]
    return gain_db, float(np.sqrt(np.mean(residual**2)))


def _log_grid(low: float, high: float, count: int) -> np.ndarray:
    low = max(float(low), 1.0e-6)
    high = max(float(high), low * 1.01)
    return np.geomspace(low, high, max(2, int(count)))


def _initial_cutoff_guess(family: str, features: DiagnosisFeatures, freq_hz: np.ndarray, measured_db: np.ndarray) -> float:
    edge = _edge_count(len(measured_db))
    if family == "lowpass":
        target = float(np.median(measured_db[:edge])) - 3.0
        crossings = _find_crossings(freq_hz, measured_db, target)
        if crossings:
            return float(crossings[0][0])
    if family == "highpass":
        target = float(np.median(measured_db[-edge:])) - 3.0
        crossings = _find_crossings(freq_hz, measured_db, target)
        if crossings:
            return float(crossings[-1][0])
    return float(np.median(freq_hz))


def _grid_fit_one_model(
    family: str,
    order: int,
    freq_hz: np.ndarray,
    measured_db: np.ndarray,
    features: DiagnosisFeatures,
) -> TransferModelFit | None:
    if len(freq_hz) < 8:
        return None

    if family in ("lowpass", "highpass"):
        guess = _initial_cutoff_guess(family, features, freq_hz, measured_db)
        fc_candidates = np.unique(
            np.r_[
                _log_grid(features.freq_min_hz * 0.15, features.freq_max_hz * 8.0, 72),
                _log_grid(max(guess / 2.8, features.freq_min_hz * 0.08), guess * 2.8, 48),
                np.array([guess, features.peak_freq_hz, features.notch_freq_hz], dtype=float),
            ]
        )
        best: tuple[float, np.ndarray] | None = None
        for fc_hz in fc_candidates:
            if not np.isfinite(fc_hz) or fc_hz <= 0.0:
                continue
            shape = _model_db_response(family, order, freq_hz, np.array([0.0, fc_hz], dtype=float))
            gain_db, rmse_db = _best_gain_for_shape(measured_db, shape)
            params = np.array([gain_db, fc_hz], dtype=float)
            if best is None or rmse_db < best[0]:
                best = (rmse_db, params)
        if best is None:
            return None
        return _build_transfer_fit(family, order, best[1], freq_hz, measured_db, success=True)

    if family in ("bandpass", "bandstop"):
        center_guess = features.peak_freq_hz if family == "bandpass" else features.notch_freq_hz
        center_guess = float(center_guess if center_guess > 0.0 else np.median(freq_hz))
        coarse_f0_candidates = np.unique(
            np.r_[
                _log_grid(features.freq_min_hz * 0.5, features.freq_max_hz * 2.0, 36),
                _log_grid(max(center_guess / 2.5, features.freq_min_hz * 0.3), center_guess * 2.5, 28),
                np.array([center_guess], dtype=float),
            ]
        )
        coarse_q_candidates = _log_grid(0.12, 25.0, 28)
        coarse_floor_candidates = np.array([-60.0, -45.0, -34.0, -25.0, -18.0, -11.0], dtype=float)
        best: tuple[float, np.ndarray] | None = None

        def search_band(
            f0_candidates: np.ndarray,
            q_candidates: np.ndarray,
            floor_candidates: np.ndarray,
            current_best: tuple[float, np.ndarray] | None,
        ) -> tuple[float, np.ndarray] | None:
            best_local = current_best
            for f0_hz in f0_candidates:
                if not np.isfinite(f0_hz) or f0_hz <= 0.0:
                    continue
                for q in q_candidates:
                    if family == "bandstop":
                        for floor_db in floor_candidates:
                            shape = _model_db_response(
                                family,
                                order,
                                freq_hz,
                                np.array([0.0, f0_hz, q, floor_db], dtype=float),
                            )
                            gain_db, rmse_db = _best_gain_for_shape(measured_db, shape)
                            params = np.array([gain_db, f0_hz, q, floor_db], dtype=float)
                            if best_local is None or rmse_db < best_local[0]:
                                best_local = (rmse_db, params)
                    else:
                        shape = _model_db_response(family, order, freq_hz, np.array([0.0, f0_hz, q], dtype=float))
                        gain_db, rmse_db = _best_gain_for_shape(measured_db, shape)
                        params = np.array([gain_db, f0_hz, q], dtype=float)
                        if best_local is None or rmse_db < best_local[0]:
                            best_local = (rmse_db, params)
            return best_local

        best = search_band(coarse_f0_candidates, coarse_q_candidates, coarse_floor_candidates, best)
        if best is not None:
            best_params = best[1]
            fine_f0_candidates = _log_grid(best_params[1] / 1.45, best_params[1] * 1.45, 22)
            fine_q_candidates = _log_grid(max(best_params[2] / 1.65, 0.05), min(best_params[2] * 1.65, 50.0), 22)
            if family == "bandstop":
                fine_floor_candidates = np.unique(
                    np.clip(np.r_[coarse_floor_candidates, np.linspace(best_params[3] - 5.0, best_params[3] + 5.0, 7)], -80.0, -3.0)
                )
            else:
                fine_floor_candidates = coarse_floor_candidates[:1]
            best = search_band(fine_f0_candidates, fine_q_candidates, fine_floor_candidates, best)
        if best is None:
            return None
        return _build_transfer_fit(family, order, best[1], freq_hz, measured_db, success=True)

    return None


def _fit_one_model(
    family: str,
    order: int,
    freq_hz: np.ndarray,
    measured_db: np.ndarray,
    features: DiagnosisFeatures,
) -> TransferModelFit | None:
    if len(freq_hz) < 8:
        return None

    finite = np.isfinite(freq_hz) & np.isfinite(measured_db) & (freq_hz > 0.0)
    if np.count_nonzero(finite) < 8:
        return None
    freq_hz = freq_hz[finite]
    measured_db = measured_db[finite]

    gain_guess = float(np.nanmax(measured_db))
    fc_guess = float(np.clip(features.peak_freq_hz, features.freq_min_hz, features.freq_max_hz))
    if family == "lowpass":
        gain_guess = features.low_gain_db
        fc_guess = max(features.freq_min_hz, min(features.freq_max_hz, features.peak_freq_hz or np.median(freq_hz)))
        x0 = np.array([gain_guess, max(fc_guess, np.median(freq_hz))], dtype=float)
        lower = np.array([features.min_gain_db - 20.0, max(features.freq_min_hz * 0.05, 0.1)])
        upper = np.array([features.max_gain_db + 20.0, features.freq_max_hz * 20.0])
    elif family == "highpass":
        gain_guess = features.high_gain_db
        fc_guess = max(features.freq_min_hz, min(features.freq_max_hz, features.peak_freq_hz or np.median(freq_hz)))
        x0 = np.array([gain_guess, max(fc_guess, np.median(freq_hz))], dtype=float)
        lower = np.array([features.min_gain_db - 20.0, max(features.freq_min_hz * 0.05, 0.1)])
        upper = np.array([features.max_gain_db + 20.0, features.freq_max_hz * 20.0])
    elif family == "bandpass":
        f0_guess = max(features.peak_freq_hz, np.median(freq_hz))
        x0 = np.array([gain_guess, f0_guess, 1.0], dtype=float)
        lower = np.array([features.min_gain_db - 20.0, max(features.freq_min_hz * 0.05, 0.1), 0.05])
        upper = np.array([features.max_gain_db + 20.0, features.freq_max_hz * 20.0, 50.0])
    elif family == "bandstop":
        f0_guess = max(features.notch_freq_hz, np.median(freq_hz))
        gain_guess = max(features.low_gain_db, features.high_gain_db)
        floor_guess = float(np.clip(features.min_gain_db - gain_guess, -55.0, -6.0))
        x0 = np.array([gain_guess, f0_guess, 1.0, floor_guess], dtype=float)
        lower = np.array([features.min_gain_db - 20.0, max(features.freq_min_hz * 0.05, 0.1), 0.05, -80.0])
        upper = np.array([features.max_gain_db + 20.0, features.freq_max_hz * 20.0, 50.0, -3.0])
    else:
        return None

    if not HAS_SCIPY_OPTIMIZE:
        return _grid_fit_one_model(family, order, freq_hz, measured_db, features)

    try:
        result = least_squares(
            lambda p: _model_db_response(family, order, freq_hz, p) - measured_db,
            x0=np.clip(x0, lower, upper),
            bounds=(lower, upper),
            max_nfev=500,
        )
    except Exception:
        return _grid_fit_one_model(family, order, freq_hz, measured_db, features)

    return _build_transfer_fit(family, order, result.x, freq_hz, measured_db, success=bool(result.success))


def fit_transfer_templates(omega, magnitude, phase_rad=None) -> list[TransferModelFit]:
    omega, magnitude, phase_rad, _ = _prepare_arrays(
        omega,
        magnitude,
        np.zeros_like(_as_float_array(omega)) if phase_rad is None else phase_rad,
        None,
    )
    if len(omega) < 8:
        return []
    features = extract_diagnosis_features(omega, magnitude, phase_rad)
    freq_hz = omega / (2.0 * math.pi)
    measured_db = _db(magnitude)
    fits: list[TransferModelFit] = []
    for family in ("lowpass", "highpass"):
        for order in (1, 2, 3):
            fit = _fit_one_model(family, order, freq_hz, measured_db, features)
            if fit is not None:
                fits.append(fit)
    for family in ("bandpass", "bandstop"):
        fit = _fit_one_model(family, 2, freq_hz, measured_db, features)
        if fit is not None:
            fits.append(fit)
    fits.sort(key=lambda item: (item.r_squared, -item.rmse_db), reverse=True)
    return fits


def _feature_score_for_type(features: DiagnosisFeatures, circuit_type: str, order: int) -> float:
    feature_type, feature_order, feature_confidence = _classify_from_features(features)
    if circuit_type == feature_type:
        score = feature_confidence
        if order == feature_order:
            score += 0.06
        elif circuit_type in ("lowpass", "highpass") and abs(order - feature_order) == 1:
            score -= 0.06
        return float(np.clip(score, 0.0, 1.0))
    if feature_type == "unknown":
        return 0.25
    if circuit_type in ("bandpass", "bandstop") and feature_type in ("lowpass", "highpass"):
        return 0.18
    return 0.12


def _fit_score(fit: TransferModelFit | None) -> float:
    if fit is None:
        return 0.0
    r2_score = np.clip((fit.r_squared - 0.55) / 0.42, 0.0, 1.0)
    rmse_score = np.clip(1.0 - fit.rmse_db / 12.0, 0.0, 1.0)
    return float(0.70 * r2_score + 0.30 * rmse_score)


def _analysis_result_state(analysis_result: Any | None) -> tuple[str | None, int, float, bool]:
    if analysis_result is None:
        return None, 0, 0.0, False
    result_type = getattr(analysis_result, "filter_type", None)
    if result_type in ("other", "unknown", ""):
        result_type = "unknown"
    result_order = int(getattr(analysis_result, "order_estimate", 0) or 0)
    result_conf = float(getattr(analysis_result, "identification_confidence", 0.0) or 0.0)
    order_reliable = bool(getattr(analysis_result, "order_reliable", False))
    anchor_reliable = (
        result_type in ("lowpass", "highpass", "bandpass", "bandstop")
        and result_order > 0
        and order_reliable
        and result_conf >= 0.75
    )
    return result_type, result_order, result_conf, anchor_reliable


def _decision_strategy_text(analysis_result: Any | None) -> str:
    result_type, result_order, result_conf, anchor_reliable = _analysis_result_state(analysis_result)
    if anchor_reliable:
        label = _system_label(result_type or "unknown", result_order)
        return (
            f"传统频响分析为主判据：filter_analysis 已可靠判定为{label}"
            f"（置信度 {result_conf:.2f}），AI用于传递函数拟合、故障证据链和主动补扫建议。"
        )
    if analysis_result is not None:
        return (
            "融合判据：传统频响分析置信度不足或阶次暂不可靠，"
            "由频响特征、物理模板拟合和故障知识库共同给出候选排名。"
        )
    return "融合判据：未提供传统分析结果，按频响特征、物理模板拟合和故障知识库综合判断。"


def rank_diagnosis_candidates(
    features: DiagnosisFeatures,
    model_fits: list[TransferModelFit],
    analysis_result: Any | None = None,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> list[DiagnosisCandidate]:
    combos = [
        ("lowpass", 1),
        ("lowpass", 2),
        ("lowpass", 3),
        ("highpass", 1),
        ("highpass", 2),
        ("highpass", 3),
        ("bandpass", 2),
        ("bandstop", 2),
        ("unknown", 1),
    ]
    by_combo = {(fit.model_family, fit.order): fit for fit in model_fits}
    result_type, result_order, result_conf, analysis_anchor_reliable = _analysis_result_state(analysis_result)
    if (
        result_type in ("lowpass", "highpass", "bandpass", "bandstop")
        and result_order > 0
        and (result_type, result_order) not in combos
    ):
        combos.insert(-1, (result_type, result_order))

    candidates: list[DiagnosisCandidate] = []
    for circuit_type, order in combos:
        fit = by_combo.get((circuit_type, order))
        feature_score = _feature_score_for_type(features, circuit_type, order)
        model_score = _fit_score(fit)
        analysis_score = 0.0
        if circuit_type == result_type:
            analysis_score = max(result_conf, 0.60)
            if order == result_order:
                analysis_score = min(1.0, analysis_score + 0.10)
            elif circuit_type in ("bandpass", "bandstop"):
                analysis_score = min(1.0, analysis_score + 0.04)
        elif circuit_type == "unknown" and result_type in (None, "other", "unknown"):
            analysis_score = 0.55

        if circuit_type == "unknown":
            confidence = max(0.05, 0.55 * feature_score + 0.45 * analysis_score)
        elif analysis_anchor_reliable:
            confidence = 0.20 * feature_score + 0.20 * model_score + 0.60 * analysis_score
            if circuit_type == result_type and order == result_order:
                confidence = max(confidence, min(0.97, 0.78 + 0.20 * result_conf))
            elif circuit_type == result_type:
                confidence *= 0.74
            else:
                confidence *= 0.45
        else:
            confidence = 0.38 * feature_score + 0.37 * model_score + 0.25 * analysis_score
        confidence = float(np.clip(confidence, 0.02, 0.99))

        key_frequency = None
        if fit is not None:
            key_frequency = fit.parameters.get("fc_hz") or fit.parameters.get("f0_hz")
        elif circuit_type in ("lowpass", "highpass"):
            key_frequency = _omega_attr_to_hz(analysis_result, "magnitude_cutoff_omega")
        elif circuit_type in ("bandpass", "bandstop"):
            key_frequency = _omega_attr_to_hz(analysis_result, "omega_c")

        evidence_parts = [
            f"特征分={feature_score:.2f}",
            f"拟合分={model_score:.2f}",
            f"传统分析分={analysis_score:.2f}",
        ]
        if analysis_anchor_reliable:
            evidence_parts.append("传统可靠锚定")
        if fit is not None:
            evidence_parts.append(f"R2={fit.r_squared:.3f}")
        candidates.append(
            DiagnosisCandidate(
                circuit_type=circuit_type,
                order_estimate=int(order),
                confidence=confidence,
                key_frequency_hz=key_frequency,
                evidence="; ".join(evidence_parts),
                model_fit=fit,
            )
        )

    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates[:5]


def load_fault_knowledge_base(path: str | Path | None = None) -> list[FaultRule]:
    kb_path = Path(path) if path is not None else DEFAULT_KB_PATH
    if not kb_path.exists():
        return []
    data = json.loads(kb_path.read_text(encoding="utf-8"))
    rules = []
    for item in data.get("rules", []):
        rules.append(
            FaultRule(
                rule_id=str(item["id"]),
                name=str(item["name"]),
                severity=str(item.get("severity", "info")),
                condition=str(item["condition"]),
                evidence_template=str(item.get("evidence", "")),
                suggestion=str(item.get("suggestion", "")),
                confidence_penalty=float(item.get("confidence_penalty", 0.0)),
            )
        )
    return rules


def _diagnostic_context(features: DiagnosisFeatures, circuit_type: str) -> dict[str, Any]:
    return {
        "point_count": features.point_count,
        "freq_min_hz": features.freq_min_hz,
        "freq_max_hz": features.freq_max_hz,
        "sweep_range_decades": features.sweep_range_decades,
        "low_gain_db": features.low_gain_db,
        "high_gain_db": features.high_gain_db,
        "max_gain_db": features.max_gain_db,
        "min_gain_db": features.min_gain_db,
        "pass_stop_span_db": features.pass_stop_span_db,
        "peak_margin_db": features.peak_margin_db,
        "notch_margin_db": features.notch_margin_db,
        "head_slope_db_dec": features.head_slope_db_dec,
        "tail_slope_db_dec": features.tail_slope_db_dec,
        "phase_span_deg": features.phase_span_deg,
        "input_rms_median_v": features.input_rms_median_v,
        "output_rms_median_v": features.output_rms_median_v,
        "input_dc_median_v": features.input_dc_median_v,
        "output_dc_median_v": features.output_dc_median_v,
        "input_clip_points": features.input_clip_points,
        "output_clip_points": features.output_clip_points,
        "clipped_points": features.clipped_points,
        "min_valid_capture_count": features.min_valid_capture_count,
        "circuit_type": circuit_type,
        "noise_floor_rms_v": NOISE_FLOOR_RMS_V,
        "low_input_rms_v": LOW_INPUT_RMS_V,
        "bias_center_v": ADC_BIAS_CENTER_V,
    }


def evaluate_fault_rules(
    features: DiagnosisFeatures,
    circuit_type: str,
    rules: list[FaultRule] | None = None,
) -> list[FaultFinding]:
    if rules is None:
        rules = load_fault_knowledge_base()
    context = _diagnostic_context(features, circuit_type)
    findings: list[FaultFinding] = []
    for rule in rules:
        try:
            triggered = bool(eval(rule.condition, {"__builtins__": {}}, context))
        except Exception:
            triggered = False
        if not triggered:
            continue
        evidence = rule.evidence_template.format(**context)
        findings.append(
            FaultFinding(
                rule_id=rule.rule_id,
                name=rule.name,
                severity=rule.severity,
                evidence=evidence,
                suggestion=rule.suggestion,
                confidence_penalty=rule.confidence_penalty,
            )
        )
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda item: (severity_order.get(item.severity, 3), -item.confidence_penalty))
    return findings


def _estimate_key_frequencies(
    omega,
    magnitude,
    phase_rad,
    circuit_type: str,
    analysis_result: Any | None,
    best_fit: TransferModelFit | None = None,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if best_fit is not None and best_fit.model_family == circuit_type:
        if circuit_type in ("lowpass", "highpass"):
            cutoff_hz = best_fit.parameters.get("fc_hz")
            return cutoff_hz, None, None, cutoff_hz, None
        if circuit_type in ("bandpass", "bandstop"):
            center_hz = best_fit.parameters.get("f0_hz")
            bandwidth_hz = best_fit.parameters.get("bandwidth_hz")
            q = best_fit.parameters.get("q")
            left_hz = right_hz = None
            if center_hz is not None and q is not None and q > 0.0:
                bandwidth_hz = center_hz / q
                left_hz = max(center_hz - bandwidth_hz / 2.0, 1.0)
                right_hz = center_hz + bandwidth_hz / 2.0
            return None, center_hz, bandwidth_hz, left_hz, right_hz

    if analysis_result is not None:
        cutoff_hz = _omega_attr_to_hz(analysis_result, "magnitude_cutoff_omega")
        center_hz = _omega_attr_to_hz(analysis_result, "omega_c")
        bandwidth_hz = _omega_attr_to_hz(analysis_result, "bandwidth_omega")
        right_hz = _omega_attr_to_hz(analysis_result, "secondary_cutoff_omega")
        if circuit_type in ("lowpass", "highpass"):
            return cutoff_hz, None, None, cutoff_hz, None
        if circuit_type in ("bandpass", "bandstop"):
            return None, center_hz, bandwidth_hz, cutoff_hz, right_hz

    omega, magnitude, phase_rad, _ = _prepare_arrays(omega, magnitude, phase_rad, None)
    if len(omega) < 3:
        return None, None, None, None, None

    freq_hz = omega / (2.0 * math.pi)
    mag_db = _db(magnitude)
    edge = _edge_count(len(mag_db))
    max_idx = int(np.argmax(mag_db))
    min_idx = int(np.argmin(mag_db))

    if circuit_type == "lowpass":
        target = float(np.median(mag_db[:edge])) - 3.0
        crossings = [item for item in _find_crossings(freq_hz, mag_db, target) if item[1] >= edge - 1]
        cutoff_hz = crossings[0][0] if crossings else None
        return cutoff_hz, None, None, cutoff_hz, None

    if circuit_type == "highpass":
        target = float(np.median(mag_db[-edge:])) - 3.0
        crossings = [item for item in _find_crossings(freq_hz, mag_db, target) if item[1] <= len(freq_hz) - edge]
        cutoff_hz = crossings[-1][0] if crossings else None
        return cutoff_hz, None, None, cutoff_hz, None

    if circuit_type == "bandpass":
        target = float(mag_db[max_idx]) - 3.0
        crossings = _find_crossings(freq_hz, mag_db, target)
        left = [item for item in crossings if item[1] < max_idx]
        right = [item for item in crossings if item[1] >= max_idx]
        left_hz = left[-1][0] if left else None
        right_hz = right[0][0] if right else None
        if left_hz is not None and right_hz is not None:
            return None, math.sqrt(left_hz * right_hz), max(right_hz - left_hz, 0.0), left_hz, right_hz
        return None, float(freq_hz[max_idx]), None, left_hz, right_hz

    if circuit_type == "bandstop":
        target = max(float(np.median(mag_db[:edge])), float(np.median(mag_db[-edge:]))) - 3.0
        crossings = _find_crossings(freq_hz, mag_db, target)
        left = [item for item in crossings if item[1] < min_idx]
        right = [item for item in crossings if item[1] >= min_idx]
        left_hz = left[-1][0] if left else None
        right_hz = right[0][0] if right else None
        bandwidth_hz = max(right_hz - left_hz, 0.0) if left_hz is not None and right_hz is not None else None
        return None, float(freq_hz[min_idx]), bandwidth_hz, left_hz, right_hz

    return None, None, None, None, None


def build_active_sweep_plan(diagnosis: IntelligentDiagnosis) -> list[str]:
    return [step.command for step in diagnosis.active_sweep_plan]


def _build_active_sweep_steps(
    features: DiagnosisFeatures,
    circuit_type: str,
    cutoff_hz: float | None,
    center_hz: float | None,
    left_hz: float | None,
    right_hz: float | None,
) -> tuple[list[str], list[ActiveSweepStep]]:
    suggestions: list[str] = []
    steps: list[ActiveSweepStep] = []

    if circuit_type in ("lowpass", "highpass") and cutoff_hz is not None:
        start = max(features.freq_min_hz * 0.8, cutoff_hz * 0.60, 1.0)
        stop = min(max(features.freq_max_hz * 1.15, cutoff_hz * 2.0), cutoff_hz * 2.5)
        if stop <= start:
            stop = cutoff_hz * 2.0
        step = _nice_step_hz((stop - start) / 36.0)
        command = _make_sweep_command(start, stop, step)
        reason = f"围绕{CIRCUIT_TYPE_LABELS[circuit_type]}截止频率 {cutoff_hz:.0f} Hz 加密扫频"
        suggestions.append(f"{reason}，建议命令 {command}。")
        steps.append(ActiveSweepStep(command=command, reason=reason))
    elif circuit_type in ("bandpass", "bandstop") and left_hz is not None and right_hz is not None:
        for label, boundary in (("左侧 -3 dB 边界", left_hz), ("右侧 -3 dB 边界", right_hz)):
            start = max(boundary * 0.70, 1.0)
            stop = boundary * 1.30
            step = _nice_step_hz((stop - start) / 32.0)
            command = _make_sweep_command(start, stop, step)
            reason = f"围绕{label} {boundary:.0f} Hz 加密补扫"
            suggestions.append(f"{reason}，确认带宽和边界斜率。")
            steps.append(ActiveSweepStep(command=command, reason=reason))
    else:
        start = max(features.freq_min_hz * 0.2, 1.0)
        stop = max(features.freq_max_hz * 5.0, start * 10.0)
        step = _nice_step_hz((stop - start) / 180.0)
        command = _make_sweep_command(start, stop, step)
        reason = "当前形态不够典型，扩大扫频范围并降低步进"
        suggestions.append(f"{reason}，建议命令 {command}。")
        steps.append(ActiveSweepStep(command=command, reason=reason))

    return suggestions, steps[:3]


def _quality_summary(features: DiagnosisFeatures, circuit_type: str) -> list[str]:
    quality = ["无削顶" if features.clipped_points == 0 else f"削顶频点 {features.clipped_points} 个"]
    if features.input_rms_median_v is not None:
        quality.append(f"输入RMS中位数 {features.input_rms_median_v:.4f} V")
    if features.output_rms_median_v is not None:
        quality.append(f"输出RMS中位数 {features.output_rms_median_v:.4f} V")
    if features.output_dc_median_v is not None:
        quality.append(f"PA1 DC约 {features.output_dc_median_v:.3f} V")
    if circuit_type == "lowpass" and features.output_rms_median_v is not None and features.high_gain_db < -30.0:
        quality.append("高频阻带可能接近噪声底")
    if circuit_type == "highpass" and features.output_rms_median_v is not None and features.low_gain_db < -30.0:
        quality.append("低频阻带可能接近噪声底")
    return quality


def run_intelligent_diagnosis(
    omega,
    magnitude,
    phase_rad,
    diagnostics: dict[str, np.ndarray] | None = None,
    analysis_result: Any | None = None,
) -> IntelligentDiagnosis:
    features = extract_diagnosis_features(omega, magnitude, phase_rad, diagnostics)
    model_fits = fit_transfer_templates(omega, magnitude, phase_rad)
    candidates = rank_diagnosis_candidates(features, model_fits, analysis_result, diagnostics)
    best_candidate = candidates[0] if candidates else DiagnosisCandidate("unknown", 1, 0.20, None, "无有效候选")
    decision_strategy = _decision_strategy_text(analysis_result)

    circuit_type = best_candidate.circuit_type
    order_estimate = best_candidate.order_estimate
    best_fit = best_candidate.model_fit
    fault_findings = evaluate_fault_rules(features, circuit_type)
    confidence_penalty = min(sum(item.confidence_penalty for item in fault_findings), 0.55)
    confidence = float(np.clip(best_candidate.confidence - confidence_penalty, 0.05, 0.99))
    needs_resweep = len(candidates) >= 2 and (candidates[0].confidence - candidates[1].confidence) < 0.12

    cutoff_hz, center_hz, bandwidth_hz, left_hz, right_hz = _estimate_key_frequencies(
        omega,
        magnitude,
        phase_rad,
        circuit_type,
        analysis_result,
        best_fit,
    )
    suggestions, sweep_steps = _build_active_sweep_steps(features, circuit_type, cutoff_hz, center_hz, left_hz, right_hz)
    if needs_resweep:
        suggestions.insert(0, "前两名候选置信度接近，建议执行智能补扫确认。")
    if features.point_count < 12:
        suggestions.insert(0, "有效频点偏少，建议至少采集 12 个以上频点再用于竞赛展示判据。")
        confidence = min(confidence, 0.45)

    possible_faults = [f"{item.name}: {item.suggestion}" for item in fault_findings]
    if not possible_faults:
        possible_faults = ["暂未发现明显硬件故障，优先按关键频率附近补扫确认。"]

    circuit_label = CIRCUIT_TYPE_LABELS.get(circuit_type, "未知")
    return IntelligentDiagnosis(
        circuit_type=circuit_type,
        circuit_label=circuit_label,
        order_estimate=int(order_estimate),
        system_label=_system_label(circuit_type, int(order_estimate)),
        cutoff_frequency_hz=cutoff_hz,
        center_frequency_hz=center_hz,
        bandwidth_hz=bandwidth_hz,
        left_cutoff_hz=left_hz,
        right_cutoff_hz=right_hz,
        confidence=confidence,
        measurement_quality=_quality_summary(features, circuit_type),
        possible_faults=possible_faults,
        next_test_suggestions=suggestions,
        adaptive_sweep_commands=[step.command for step in sweep_steps],
        features=features,
        candidates=candidates,
        best_fit=best_fit,
        model_fits=model_fits,
        fault_findings=fault_findings,
        active_sweep_plan=sweep_steps,
        equivalent_transfer_function=best_fit.transfer_function if best_fit is not None else None,
        needs_resweep_confirmation=needs_resweep,
        decision_strategy=decision_strategy,
    )


def _format_hz(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    if value >= 1000.0:
        return f"{value:.0f} Hz"
    return f"{value:.1f} Hz"


def _candidate_label(candidate: DiagnosisCandidate) -> str:
    return f"{_system_label(candidate.circuit_type, candidate.order_estimate)} {candidate.confidence:.2f}"


def _format_coefficients(values: list[float]) -> str:
    return "[" + ", ".join(_format_coeff(value) for value in values) + "]"


def format_ai_diagnosis_report_lines(diagnosis: IntelligentDiagnosis) -> list[str]:
    if diagnosis.circuit_type in ("lowpass", "highpass"):
        key_frequency = f"截止频率 {_format_hz(diagnosis.cutoff_frequency_hz)}"
    elif diagnosis.circuit_type == "bandpass":
        key_frequency = (
            f"中心频率 {_format_hz(diagnosis.center_frequency_hz)}，"
            f"带宽 {_format_hz(diagnosis.bandwidth_hz)}"
        )
    elif diagnosis.circuit_type == "bandstop":
        key_frequency = (
            f"陷波中心 {_format_hz(diagnosis.center_frequency_hz)}，"
            f"阻带宽度 {_format_hz(diagnosis.bandwidth_hz)}"
        )
    else:
        key_frequency = "关键频率无法稳定估计"

    lines = [
        "=== AI辅助电路识别与智能诊断报告 v2 ===",
        f"智能识别: {diagnosis.system_label}",
        f"判定策略: {diagnosis.decision_strategy}",
        f"置信度: {diagnosis.confidence:.2f}" + ("（需补扫确认）" if diagnosis.needs_resweep_confirmation else ""),
        f"关键频率: {key_frequency}",
        "候选排名: " + " / ".join(_candidate_label(item) for item in diagnosis.candidates[:3]),
    ]

    if diagnosis.best_fit is not None:
        fit = diagnosis.best_fit
        lines.append(
            f"模型拟合: {fit.evidence}；{fit.residual_summary}；"
            f"参数={{{', '.join(f'{k}={v:.3g}' for k, v in fit.parameters.items())}}}"
        )
        if diagnosis.equivalent_transfer_function is not None:
            tf = diagnosis.equivalent_transfer_function
            lines.append(
                "等效传递函数: "
                f"{tf.expression}；{tf.parameter_summary}；"
                f"num={_format_coefficients(tf.numerator)}；den={_format_coefficients(tf.denominator)}"
            )
    elif not HAS_SCIPY_OPTIMIZE:
        lines.append("模型拟合: 当前环境缺少 scipy.optimize，已退回到特征规则诊断。")
    else:
        lines.append("模型拟合: 数据点不足或模板未收敛，已退回到特征规则诊断。")

    lines.extend(
        [
            "测量质量: " + "；".join(diagnosis.measurement_quality),
            "故障证据链: "
            + (
                "；".join(f"{item.name}({item.severity}): {item.evidence}" for item in diagnosis.fault_findings)
                if diagnosis.fault_findings
                else "暂未触发故障知识库规则"
            ),
            "可能故障原因: " + "；".join(diagnosis.possible_faults),
            "下一步测试建议: " + "；".join(diagnosis.next_test_suggestions),
        ]
    )
    if diagnosis.active_sweep_plan:
        lines.append(
            "一键补扫计划: "
            + " | ".join(f"{step.command} ({step.reason})" for step in diagnosis.active_sweep_plan)
        )
        lines.append("补扫SWEEP建议: " + "；".join(step.command for step in diagnosis.active_sweep_plan))
    if diagnosis.features is not None:
        f = diagnosis.features
        lines.append(
            "关键特征: "
            f"低频{f.low_gain_db:.1f} dB，高频{f.high_gain_db:.1f} dB，"
            f"低频斜率{f.head_slope_db_dec:.1f} dB/dec，高频斜率{f.tail_slope_db_dec:.1f} dB/dec，"
            f"相位跨度{f.phase_span_deg:.1f} deg，Nyquist到(-1,0)最小距离{f.nyquist_min_distance:.3f}"
        )
    lines.append("")
    return lines
