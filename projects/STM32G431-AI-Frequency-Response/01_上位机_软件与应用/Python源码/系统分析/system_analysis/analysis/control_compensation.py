from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


MODE_AUTO = "auto"
MODE_GAIN = "gain"
MODE_LEAD = "lead"
MODE_LAG = "lag"
MODE_LEAD_LAG = "lead_lag"

MODE_LABELS = {
    MODE_AUTO: "自动",
    MODE_GAIN: "比例增益",
    MODE_LEAD: "超前校正",
    MODE_LAG: "滞后校正",
    MODE_LEAD_LAG: "滞后-超前校正",
}

UI_MODE_TO_KEY = {
    "自动": MODE_AUTO,
    "比例增益": MODE_GAIN,
    "比例增益校正": MODE_GAIN,
    "增益": MODE_GAIN,
    "超前": MODE_LEAD,
    "超前校正": MODE_LEAD,
    "滞后": MODE_LAG,
    "滞后校正": MODE_LAG,
    "滞后-超前": MODE_LEAD_LAG,
    "滞后-超前校正": MODE_LEAD_LAG,
    "lead": MODE_LEAD,
    "gain": MODE_GAIN,
    "lag": MODE_LAG,
    "lead_lag": MODE_LEAD_LAG,
    "auto": MODE_AUTO,
}

MIN_POINTS = 5
MIN_PHASE_BOOST_DEG = 3.0
MAX_SINGLE_LEAD_BOOST_DEG = 58.0


@dataclass(frozen=True)
class ControlCompensationSettings:
    enabled: bool = False
    mode: str = MODE_AUTO
    target_phase_margin_deg: float = 50.0
    target_gain_margin_db: float = 6.0
    safety_phase_deg: float = 8.0
    target_crossover_hz: float | None = None
    low_frequency_gain_boost_db: float = 0.0


@dataclass(frozen=True)
class FrequencyMargins:
    gain_crossover_rad_s: float | None
    phase_at_gain_crossover_deg: float | None
    phase_margin_deg: float | None
    phase_crossover_rad_s: float | None
    gain_at_phase_crossover: float | None
    gain_margin_db: float | None
    gain_crossovers_rad_s: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ControlCompensatorCandidate:
    kind: str
    label: str
    expression: str
    gain: float
    target_crossover_rad_s: float | None
    target_crossover_hz: float | None
    predicted_phase_margin_deg: float | None
    phase_boost_deg: float = 0.0
    alpha: float | None = None
    beta: float | None = None
    lead_t: float | None = None
    lag_t: float | None = None
    zero_rad_s: float | None = None
    pole_rad_s: float | None = None
    lag_zero_rad_s: float | None = None
    lag_pole_rad_s: float | None = None
    confidence: float = 0.0
    evidence: str = ""
    suggestion: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ControlCompensationReport:
    enabled: bool
    settings: ControlCompensationSettings
    current_margins: FrequencyMargins | None
    compensated_margins: FrequencyMargins | None
    summary: str
    notes: list[str]
    candidates: list[ControlCompensatorCandidate] = field(default_factory=list)

    @property
    def best_candidate(self) -> ControlCompensatorCandidate | None:
        return self.candidates[0] if self.candidates else None


def normalize_control_mode(mode: str | None) -> str:
    return UI_MODE_TO_KEY.get(str(mode or "").strip(), MODE_AUTO)


def settings_from_inputs(
    *,
    enabled: bool,
    mode: str,
    target_phase_margin: str | float,
    target_gain_margin_db: str | float = 6.0,
    safety_phase: str | float = 8.0,
    target_crossover_hz: str | float | None = None,
    low_frequency_gain_boost_db: str | float = 0.0,
) -> ControlCompensationSettings:
    target_pm = _parse_float(target_phase_margin, "目标相位裕度")
    target_gm = _parse_float(target_gain_margin_db, "最小增益裕度")
    safety = _parse_float(safety_phase, "安全补偿相位")
    boost_db = _parse_float(low_frequency_gain_boost_db, "低频增益提升")
    target_hz = _parse_optional_float(target_crossover_hz, "目标穿越频率")

    if target_pm <= 0.0 or target_pm >= 120.0:
        raise ValueError("目标相位裕度建议设置在 0 到 120 度之间。")
    if target_gm < 0.0 or target_gm > 80.0:
        raise ValueError("最小增益裕度建议设置在 0 到 80 dB 之间。")
    if safety < 0.0 or safety > 45.0:
        raise ValueError("安全补偿相位建议设置在 0 到 45 度之间。")
    if boost_db < 0.0 or boost_db > 80.0:
        raise ValueError("低频增益提升建议设置在 0 到 80 dB 之间。")
    if target_hz is not None and target_hz <= 0.0:
        raise ValueError("目标穿越频率必须为正数。")

    return ControlCompensationSettings(
        enabled=bool(enabled),
        mode=normalize_control_mode(mode),
        target_phase_margin_deg=float(target_pm),
        target_gain_margin_db=float(target_gm),
        safety_phase_deg=float(safety),
        target_crossover_hz=target_hz,
        low_frequency_gain_boost_db=float(boost_db),
    )


def compute_frequency_margins(omega: Any, magnitude: Any, phase_rad: Any) -> FrequencyMargins:
    omega_arr, mag_arr, phase_arr = _prepare_frequency_response(omega, magnitude, phase_rad)
    if len(omega_arr) < MIN_POINTS:
        return FrequencyMargins(
            None,
            None,
            None,
            None,
            None,
            None,
            notes=["有效频点太少，无法可靠计算稳定裕度。"],
        )

    logw = np.log(omega_arr)
    mag_db = 20.0 * np.log10(np.clip(mag_arr, 1.0e-12, None))
    phase_deg = np.degrees(np.unwrap(phase_arr))

    gain_crossings = _level_crossings(logw, mag_db, 0.0)
    gain_crossovers = [float(math.exp(x)) for x in gain_crossings]
    selected_wc = None
    selected_phase = None
    selected_pm = None
    notes: list[str] = []

    if gain_crossovers:
        candidates: list[tuple[float, float, float]] = []
        for wc in gain_crossovers:
            phase_at_wc = _interp_log_omega(omega_arr, phase_deg, wc)
            if phase_at_wc is None:
                continue
            phase_for_margin = _phase_for_margin(phase_at_wc)
            pm = 180.0 + phase_for_margin
            candidates.append((float(wc), float(phase_for_margin), float(pm)))
        if candidates:
            positive = [item for item in candidates if item[2] >= -180.0]
            selected_wc, selected_phase, selected_pm = min(positive or candidates, key=lambda item: item[2])
            if len(candidates) > 1:
                notes.append("检测到多个 0 dB 穿越点，报告采用相位裕度最小的穿越点。")
    else:
        if float(np.max(mag_db)) < 0.0:
            notes.append("全扫频范围内开环幅值低于 0 dB，未形成增益穿越。")
        elif float(np.min(mag_db)) > 0.0:
            notes.append("全扫频范围内开环幅值高于 0 dB，增益穿越可能在更高频率。")
        else:
            notes.append("未能稳定插值得到 0 dB 增益穿越点。")

    phase_crossing = None
    gain_at_phase = None
    gain_margin_db = None
    phase_crossings = _level_crossings(logw, phase_deg, -180.0)
    if phase_crossings:
        # Use the first -180 degree crossing as the classical phase crossover.
        phase_crossing = float(math.exp(phase_crossings[0]))
        gain_at_phase = _interp_log_omega(omega_arr, mag_arr, phase_crossing)
        if gain_at_phase is not None and gain_at_phase > 0.0:
            gain_margin_db = float(-20.0 * math.log10(max(gain_at_phase, 1.0e-12)))

    return FrequencyMargins(
        gain_crossover_rad_s=selected_wc,
        phase_at_gain_crossover_deg=selected_phase,
        phase_margin_deg=selected_pm,
        phase_crossover_rad_s=phase_crossing,
        gain_at_phase_crossover=None if gain_at_phase is None else float(gain_at_phase),
        gain_margin_db=gain_margin_db,
        gain_crossovers_rad_s=gain_crossovers,
        notes=notes,
    )


def build_control_compensation_report(
    omega: Any,
    magnitude: Any,
    phase_rad: Any,
    settings: ControlCompensationSettings | None = None,
    analysis_result: Any | None = None,
) -> ControlCompensationReport:
    settings = settings or ControlCompensationSettings(enabled=False)
    if not settings.enabled:
        return ControlCompensationReport(
            enabled=False,
            settings=settings,
            current_margins=None,
            compensated_margins=None,
            summary="自动控制校正未启用。",
            notes=[],
        )

    omega_arr, mag_arr, phase_arr = _prepare_frequency_response(omega, magnitude, phase_rad)
    if len(omega_arr) < MIN_POINTS:
        margins = compute_frequency_margins(omega_arr, mag_arr, phase_arr)
        return ControlCompensationReport(
            enabled=True,
            settings=settings,
            current_margins=margins,
            compensated_margins=None,
            summary="有效频点太少，无法设计控制校正器。",
            notes=margins.notes,
        )

    margins = compute_frequency_margins(omega_arr, mag_arr, phase_arr)
    notes = list(margins.notes)
    if analysis_result is not None:
        nyquist_stable = getattr(analysis_result, "nyquist_stable", None)
        if nyquist_stable is False:
            notes.append("奈奎斯特结果提示闭环可能不稳定，校正参数应先小信号仿真再上机。")

    candidate = _design_candidate(omega_arr, mag_arr, phase_arr, margins, settings, notes)
    candidates = [candidate] if candidate is not None else []

    compensated_margins = None
    if candidate is not None and candidate.kind != "none":
        response = compensator_response(omega_arr, candidate)
        compensated_margins = compute_frequency_margins(
            omega_arr,
            mag_arr * np.abs(response),
            phase_arr + np.angle(response),
        )

    summary = _build_summary(settings, margins, candidate, compensated_margins)
    return ControlCompensationReport(
        enabled=True,
        settings=settings,
        current_margins=margins,
        compensated_margins=compensated_margins,
        summary=summary,
        notes=notes,
        candidates=candidates,
    )


def compensator_response(omega: Any, candidate: ControlCompensatorCandidate) -> np.ndarray:
    omega_arr = np.asarray(omega, dtype=float).flatten()
    s = 1j * omega_arr
    response = np.ones_like(s, dtype=complex) * float(candidate.gain)

    if candidate.alpha is not None and candidate.lead_t is not None:
        alpha = float(candidate.alpha)
        t = float(candidate.lead_t)
        response *= (1.0 + s * t) / (1.0 + s * alpha * t)

    if candidate.beta is not None and candidate.lag_t is not None:
        beta = float(candidate.beta)
        t = float(candidate.lag_t)
        response *= beta * (1.0 + s * t) / (1.0 + s * beta * t)

    return response


def format_control_compensation_report_lines(report: ControlCompensationReport | None) -> list[str]:
    if report is None or not report.enabled:
        return []

    lines = ["", "--- 自动控制校正 ---", report.summary]
    margins = report.current_margins
    if margins is not None:
        lines.append(
            "自然增益穿越: "
            f"ωgc={_format_rad_s(margins.gain_crossover_rad_s)}, "
            f"PM={_format_deg(margins.phase_margin_deg)}, "
            f"GM={_format_db(margins.gain_margin_db)}"
        )

    candidate = report.best_candidate
    if candidate is not None:
        target_pm = float(report.settings.target_phase_margin_deg)
        margin_shortfall = (
            candidate.kind != "none"
            and candidate.predicted_phase_margin_deg is not None
            and candidate.predicted_phase_margin_deg < target_pm
        )
        heading = "候选校正器" if margin_shortfall else "推荐校正器"
        lines.append(f"{heading}: {candidate.label}")
        if margin_shortfall:
            lines.append(
                f"可行性提示: 预计 PM={candidate.predicted_phase_margin_deg:.1f}° "
                f"低于目标 {target_pm:.1f}°，不建议直接采用该目标穿越频率。"
            )
        if candidate.expression:
            lines.append(f"传递函数: {candidate.expression}")
        if candidate.kind != "none":
            lines.append(
                f"比例增益: K={_format_number(candidate.gain)} "
                f"({_format_db(20.0 * math.log10(max(candidate.gain, 1.0e-12)))})"
            )
        if candidate.target_crossover_rad_s is not None:
            lines.append(
                f"设计穿越频率: {_format_rad_s(candidate.target_crossover_rad_s)} "
                f"({_format_hz(candidate.target_crossover_hz)})"
            )
        if candidate.kind in ("lead", "lead_lag"):
            lines.append(
                "超前参数: "
                f"alpha={_format_number(candidate.alpha)}, "
                f"T={_format_seconds(candidate.lead_t)}, "
                f"ωz={_format_rad_s(candidate.zero_rad_s)}, "
                f"ωp={_format_rad_s(candidate.pole_rad_s)}, "
                f"相位提升≈{candidate.phase_boost_deg:.1f}°"
            )
        if candidate.kind in ("lag", "lead_lag"):
            lines.append(
                "滞后参数: "
                f"beta={_format_number(candidate.beta)}, "
                f"T={_format_seconds(candidate.lag_t)}, "
                f"ωz={_format_rad_s(candidate.lag_zero_rad_s)}, "
                f"ωp={_format_rad_s(candidate.lag_pole_rad_s)}"
            )
        if candidate.predicted_phase_margin_deg is not None:
            lines.append(f"预计校正后相位裕度: {candidate.predicted_phase_margin_deg:.1f}°")
        if candidate.evidence:
            lines.append(f"依据: {candidate.evidence}")
        if candidate.suggestion:
            lines.append(f"使用建议: {candidate.suggestion}")
        lines.extend(candidate.notes[:4])

    if report.compensated_margins is not None:
        cm = report.compensated_margins
        lines.append(
            "校正后估计: "
            f"ωgc={_format_rad_s(cm.gain_crossover_rad_s)}, "
            f"PM={_format_deg(cm.phase_margin_deg)}, "
            f"GM={_format_db(cm.gain_margin_db)}"
        )

    for note in report.notes[:4]:
        lines.append(f"提示: {note}")
    return lines


def _design_candidate(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_rad: np.ndarray,
    margins: FrequencyMargins,
    settings: ControlCompensationSettings,
    report_notes: list[str],
) -> ControlCompensatorCandidate | None:
    mode = normalize_control_mode(settings.mode)
    wants_lag = mode in (MODE_LAG, MODE_LEAD_LAG)
    wants_lead = mode in (MODE_LEAD, MODE_LEAD_LAG)
    boost_db = float(settings.low_frequency_gain_boost_db)
    gm_low = (
        settings.target_gain_margin_db > 0.0
        and margins.gain_margin_db is not None
        and margins.gain_margin_db < settings.target_gain_margin_db
    )
    target_point = _target_design_point(omega, magnitude, phase_rad, settings)
    target_out_of_range = settings.target_crossover_hz is not None and target_point is None

    current_pm = margins.phase_margin_deg
    if mode == MODE_AUTO:
        if target_point is not None:
            wants_lead = target_point["phase_margin_deg"] < settings.target_phase_margin_deg
        elif current_pm is None:
            wants_lead = False
        else:
            wants_lead = current_pm < settings.target_phase_margin_deg
        wants_lag = boost_db > 0.0

    lead_params = None
    if wants_lead:
        lead_params = _design_lead(omega, magnitude, phase_rad, margins, settings, report_notes)
        if lead_params is None and mode == MODE_LEAD:
            return _no_compensation_candidate(
                margins,
                "无法形成超前校正设计",
                "没有 0 dB 穿越点且未指定目标穿越频率；请扩大扫频范围或填写目标穿越频率。",
            )

    lag_params = None
    if wants_lag:
        lag_params = _design_lag(margins, settings, lead_params)
        if lag_params is None and mode == MODE_LAG:
            return _no_compensation_candidate(
                margins,
                "无法形成滞后校正设计",
                "滞后校正需要已有增益穿越频率，或先指定目标穿越频率。",
            )

    if lead_params is None and lag_params is None:
        if target_point is not None and mode in (MODE_AUTO, MODE_GAIN):
            return _target_gain_candidate(margins, settings, target_point)
        if target_out_of_range:
            return _no_compensation_candidate(
                margins,
                "目标穿越频率不在本次实测扫频范围内，无法按目标频点插值设计",
                (
                    f"请把目标穿越 Hz 设在 {omega[0] / (2.0 * math.pi):.3g} 到 "
                    f"{omega[-1] / (2.0 * math.pi):.3g} Hz 之间，或重新扫频覆盖该目标频率。"
                ),
            )
        if mode == MODE_GAIN:
            return _no_compensation_candidate(
                margins,
                "比例增益校正需要指定目标穿越频率",
                "请填写目标穿越 Hz，软件会按该频点的幅值计算 K，使该频率附近成为新的 0 dB 穿越点。",
            )
        if gm_low:
            return _gain_reduction_candidate(margins, settings)
        if current_pm is not None and current_pm >= settings.target_phase_margin_deg:
            if settings.target_gain_margin_db > 0.0 and margins.gain_margin_db is None:
                return _no_compensation_candidate(
                    margins,
                    "当前相位裕度已达到目标，增益裕度未在本次扫频范围内形成 -180° 相位穿越",
                    "暂不生成校正器；若实验要求增益裕度，请扩大扫频范围到相位接近 -180° 后重新测量。",
                )
            return _no_compensation_candidate(
                margins,
                "当前相位裕度和增益裕度已达到目标",
                "暂不需要超前校正；若要改善稳态误差，可选择滞后校正并填写低频增益提升。",
            )
        return _no_compensation_candidate(
            margins,
            "暂未生成校正器",
            "请确认测量的是开环频响；如果扫频全程低于 0 dB，请填写目标穿越 Hz，软件会按该频点计算比例增益或超前校正器。",
        )

    gain = 1.0
    label_parts: list[str] = []
    expression_parts: list[str] = []
    target_wc = None
    predicted_pm = None
    confidence = 0.65
    candidate_notes: list[str] = []

    alpha = lead_t = zero = pole = phase_boost = None
    if lead_params is not None:
        alpha = lead_params["alpha"]
        lead_t = lead_params["t"]
        zero = lead_params["zero"]
        pole = lead_params["pole"]
        phase_boost = lead_params["phase_boost"]
        gain *= lead_params["gain"]
        target_wc = lead_params["target_wc"]
        predicted_pm = lead_params["predicted_pm"]
        label_parts.append("超前校正")
        expression_parts.append(f"(1 + {lead_t:.6g}s)/(1 + {alpha * lead_t:.6g}s)")
        confidence += 0.15
        candidate_notes.extend(lead_params["notes"])

    beta = lag_t = lag_zero = lag_pole = None
    if lag_params is not None:
        beta = lag_params["beta"]
        lag_t = lag_params["t"]
        lag_zero = lag_params["zero"]
        lag_pole = lag_params["pole"]
        target_wc = target_wc or lag_params["target_wc"]
        label_parts.append("滞后校正")
        expression_parts.append(f"{beta:.6g}*(1 + {lag_t:.6g}s)/(1 + {beta * lag_t:.6g}s)")
        confidence += 0.08
        candidate_notes.extend(lag_params["notes"])

    if settings.target_crossover_hz is not None and target_wc is not None:
        target_gain = _gain_to_place_crossover_at(omega, magnitude, target_wc, alpha, lead_t, beta, lag_t)
        if target_gain is not None:
            gain = target_gain

    kind = MODE_LEAD_LAG if lead_params is not None and lag_params is not None else (
        MODE_LEAD if lead_params is not None else MODE_LAG
    )
    label = " + ".join(label_parts)
    expression = f"Gc(s) = {gain:.6g}"
    if expression_parts:
        expression += "*" + "*".join(expression_parts)
    target_hz = None if target_wc is None else target_wc / (2.0 * math.pi)
    if target_wc is not None:
        predicted_pm = _predict_phase_margin_at(omega, magnitude, phase_rad, target_wc, alpha, lead_t, beta, lag_t)
    if predicted_pm is not None and predicted_pm < settings.target_phase_margin_deg:
        candidate_notes.insert(
            0,
            (
                f"预计校正后 PM 只有 {predicted_pm:.1f}°，低于目标 "
                f"{settings.target_phase_margin_deg:.1f}°；请降低目标穿越 Hz，"
                "或改用两级超前/重新建模后再设计。"
            ),
        )
        confidence = min(confidence, 0.45)

    evidence_parts = []
    if current_pm is not None:
        evidence_parts.append(f"当前 PM={current_pm:.1f}°，目标 PM={settings.target_phase_margin_deg:.1f}°")
    elif target_point is not None:
        evidence_parts.append(
            f"目标频点 PM≈{target_point['phase_margin_deg']:.1f}°，目标 PM={settings.target_phase_margin_deg:.1f}°"
        )
    if margins.gain_crossover_rad_s is not None:
        evidence_parts.append(f"当前 ωgc={margins.gain_crossover_rad_s:.3g} rad/s")
    else:
        evidence_parts.append("当前扫频未发现自然 0 dB 穿越")
    if target_wc is not None:
        evidence_parts.append(f"设计 ωgc≈{target_wc:.3g} rad/s")

    suggestion = _candidate_suggestion(kind, alpha, beta, zero, pole, lag_zero, lag_pole)
    if abs(gain - 1.0) > 1.0e-6:
        suggestion = f"先将控制器比例增益设为 K={gain:.3g}，把目标频率处开环幅值拉到 0 dB。 {suggestion}"
    return ControlCompensatorCandidate(
        kind=kind,
        label=label,
        expression=expression,
        gain=float(gain),
        target_crossover_rad_s=target_wc,
        target_crossover_hz=target_hz,
        predicted_phase_margin_deg=predicted_pm,
        phase_boost_deg=0.0 if phase_boost is None else float(phase_boost),
        alpha=None if alpha is None else float(alpha),
        beta=None if beta is None else float(beta),
        lead_t=None if lead_t is None else float(lead_t),
        lag_t=None if lag_t is None else float(lag_t),
        zero_rad_s=None if zero is None else float(zero),
        pole_rad_s=None if pole is None else float(pole),
        lag_zero_rad_s=None if lag_zero is None else float(lag_zero),
        lag_pole_rad_s=None if lag_pole is None else float(lag_pole),
        confidence=float(min(confidence, 0.95)),
        evidence="；".join(evidence_parts),
        suggestion=suggestion,
        notes=candidate_notes,
    )


def _gain_reduction_candidate(
    margins: FrequencyMargins,
    settings: ControlCompensationSettings,
) -> ControlCompensatorCandidate:
    current_gm = float(margins.gain_margin_db or 0.0)
    reduction_db = max(settings.target_gain_margin_db - current_gm + 1.0, 0.0)
    gain = 10.0 ** (-reduction_db / 20.0)
    predicted_gm = current_gm + reduction_db
    evidence = (
        f"当前 GM={current_gm:.1f} dB，目标 GM={settings.target_gain_margin_db:.1f} dB，"
        f"建议先把开环增益降低 {reduction_db:.1f} dB"
    )
    return ControlCompensatorCandidate(
        kind="gain",
        label="开环增益校正",
        expression=f"Gc(s) = {gain:.6g}",
        gain=float(gain),
        target_crossover_rad_s=margins.gain_crossover_rad_s,
        target_crossover_hz=None if margins.gain_crossover_rad_s is None else margins.gain_crossover_rad_s / (2.0 * math.pi),
        predicted_phase_margin_deg=margins.phase_margin_deg,
        confidence=0.72,
        evidence=evidence,
        suggestion=(
            f"先将控制器比例增益乘以 {gain:.3g}，理论增益裕度约提高到 {predicted_gm:.1f} dB；"
            "随后重新扫频确认新的 0 dB 穿越频率和相位裕度。"
        ),
        notes=["增益校正会降低低频开环增益，稳态误差可能变大；若稳态指标变差，再叠加滞后校正。"],
    )


def _target_gain_candidate(
    margins: FrequencyMargins,
    settings: ControlCompensationSettings,
    target_point: dict[str, float],
) -> ControlCompensatorCandidate:
    target_wc = float(target_point["omega"])
    target_hz = target_wc / (2.0 * math.pi)
    plant_mag = max(float(target_point["magnitude"]), 1.0e-12)
    gain = 1.0 / plant_mag
    plant_mag_db = 20.0 * math.log10(plant_mag)
    gain_db = 20.0 * math.log10(max(gain, 1.0e-12))
    phase_deg = float(target_point["phase_deg"])
    predicted_pm = float(target_point["phase_margin_deg"])
    evidence_parts = [
        f"目标 {target_hz:.3g} Hz 处 |G|={plant_mag_db:.1f} dB、相位={phase_deg:.1f}°",
        f"乘以 K={gain:.3g} ({gain_db:.1f} dB) 后在该频率形成 0 dB 穿越",
        f"预计 PM={predicted_pm:.1f}°",
    ]
    if margins.gain_crossover_rad_s is None:
        evidence_parts.insert(0, "当前扫频未发现自然 0 dB 穿越")

    notes = ["目标频点相位裕度已满足要求，因此先给出比例增益校正，不额外加入超前网络。"]
    if predicted_pm < settings.target_phase_margin_deg + settings.safety_phase_deg:
        notes.append("预计相位裕度接近目标边界；上机后建议重新扫频确认，必要时再加入小幅超前校正。")

    return ControlCompensatorCandidate(
        kind="gain",
        label="比例增益校正",
        expression=f"Gc(s) = {gain:.6g}",
        gain=float(gain),
        target_crossover_rad_s=target_wc,
        target_crossover_hz=target_hz,
        predicted_phase_margin_deg=predicted_pm,
        confidence=0.78,
        evidence="；".join(evidence_parts),
        suggestion=(
            f"把控制器比例增益乘以 {gain:.3g}，目标是让 {target_hz:.3g} Hz 附近成为新的 0 dB 穿越点；"
            "随后重新导入实测数据，检查 PM/GM 是否仍满足指标。"
        ),
        notes=notes,
    )


def _design_lead(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_rad: np.ndarray,
    margins: FrequencyMargins,
    settings: ControlCompensationSettings,
    report_notes: list[str],
) -> dict[str, float | list[str]] | None:
    explicit_wc = None
    if settings.target_crossover_hz is not None:
        explicit_wc = float(settings.target_crossover_hz) * 2.0 * math.pi
        if explicit_wc < float(np.min(omega)) or explicit_wc > float(np.max(omega)):
            report_notes.append("目标穿越频率在当前扫频范围之外，预测裕度只能作为外推参考。")

    target_wc = explicit_wc or margins.gain_crossover_rad_s
    if target_wc is None:
        return None

    notes: list[str] = []
    phi_needed = _required_phase_boost_at(omega, phase_rad, target_wc, settings)
    for _ in range(3):
        phi_design = float(np.clip(phi_needed, MIN_PHASE_BOOST_DEG, MAX_SINGLE_LEAD_BOOST_DEG))
        alpha = _alpha_from_phase_boost(phi_design)
        if explicit_wc is None:
            target_mag = math.sqrt(alpha)
            selected = _find_magnitude_crossing(omega, magnitude, target_mag, prefer_above=margins.gain_crossover_rad_s)
            if selected is not None:
                target_wc = selected
        phi_needed = _required_phase_boost_at(omega, phase_rad, target_wc, settings)

    clipped = phi_needed > MAX_SINGLE_LEAD_BOOST_DEG
    phi_design = float(np.clip(phi_needed, MIN_PHASE_BOOST_DEG, MAX_SINGLE_LEAD_BOOST_DEG))
    alpha = _alpha_from_phase_boost(phi_design)
    if clipped:
        notes.append(
            f"所需相位提升约 {phi_needed:.1f}°，单级超前已按 {MAX_SINGLE_LEAD_BOOST_DEG:.0f}° 上限设计；"
            "若校正后裕度仍不足，建议采用两级超前或降低目标穿越频率。"
        )

    plant_mag = _interp_log_omega(omega, magnitude, target_wc)
    if plant_mag is None or plant_mag <= 0.0:
        return None
    lead_mag = 1.0 / math.sqrt(alpha)
    gain = 1.0 / max(plant_mag * lead_mag, 1.0e-12)
    lead_t = 1.0 / (float(target_wc) * math.sqrt(alpha))
    zero = 1.0 / lead_t
    pole = 1.0 / (alpha * lead_t)
    predicted_pm = _predict_phase_margin_at(omega, magnitude, phase_rad, target_wc, alpha, lead_t, None, None)

    if settings.target_crossover_hz is None:
        notes.append("未指定目标穿越频率，已按超前网络最大相位点自动选择新的 0 dB 穿越点。")
    else:
        notes.append("已按用户指定目标穿越频率设计超前网络。")

    return {
        "alpha": float(alpha),
        "t": float(lead_t),
        "zero": float(zero),
        "pole": float(pole),
        "gain": float(gain),
        "target_wc": float(target_wc),
        "phase_boost": float(phi_design),
        "predicted_pm": None if predicted_pm is None else float(predicted_pm),
        "notes": notes,
    }


def _design_lag(
    margins: FrequencyMargins,
    settings: ControlCompensationSettings,
    lead_params: dict[str, Any] | None,
) -> dict[str, float | list[str]] | None:
    boost_db = float(settings.low_frequency_gain_boost_db)
    if boost_db <= 0.0:
        if settings.mode == MODE_LAG:
            boost_db = 10.0
        else:
            return None

    target_wc = None
    if lead_params is not None:
        target_wc = float(lead_params["target_wc"])
    elif settings.target_crossover_hz is not None:
        target_wc = float(settings.target_crossover_hz) * 2.0 * math.pi
    else:
        target_wc = margins.gain_crossover_rad_s
    if target_wc is None or target_wc <= 0.0:
        return None

    beta = float(10.0 ** (boost_db / 20.0))
    beta = float(np.clip(beta, 1.05, 1000.0))
    zero = float(target_wc / 10.0)
    pole = float(zero / beta)
    lag_t = 1.0 / zero
    notes = [
        f"滞后网络按低频增益提升 {boost_db:.1f} dB 设计，零点放在目标穿越频率的约 1/10 处以减少相位损失。"
    ]
    return {
        "beta": beta,
        "t": lag_t,
        "zero": zero,
        "pole": pole,
        "target_wc": float(target_wc),
        "notes": notes,
    }


def _no_compensation_candidate(margins: FrequencyMargins, label: str, suggestion: str) -> ControlCompensatorCandidate:
    return ControlCompensatorCandidate(
        kind="none",
        label="暂不校正",
        expression="",
        gain=1.0,
        target_crossover_rad_s=margins.gain_crossover_rad_s,
        target_crossover_hz=None if margins.gain_crossover_rad_s is None else margins.gain_crossover_rad_s / (2.0 * math.pi),
        predicted_phase_margin_deg=margins.phase_margin_deg,
        confidence=0.7 if margins.phase_margin_deg is not None else 0.4,
        evidence=label,
        suggestion=suggestion,
    )


def _build_summary(
    settings: ControlCompensationSettings,
    margins: FrequencyMargins,
    candidate: ControlCompensatorCandidate | None,
    compensated_margins: FrequencyMargins | None,
) -> str:
    mode_label = MODE_LABELS.get(normalize_control_mode(settings.mode), "自动")
    if candidate is None:
        return f"{mode_label}: 未生成校正器。"
    if candidate.kind == "none":
        return f"{mode_label}: {candidate.evidence}。"
    current = "PM未知" if margins.phase_margin_deg is None else f"当前 PM {margins.phase_margin_deg:.1f}°"
    predicted = candidate.predicted_phase_margin_deg
    if compensated_margins is not None and compensated_margins.phase_margin_deg is not None:
        predicted = compensated_margins.phase_margin_deg
    predicted_text = "预计 PM未知" if predicted is None else f"预计校正后 PM {predicted:.1f}°"
    if predicted is not None and predicted < settings.target_phase_margin_deg:
        return (
            f"{mode_label}: 目标穿越频率偏高，单次校正预计 PM {predicted:.1f}°，"
            f"低于目标 {settings.target_phase_margin_deg:.1f}°；建议降低目标穿越 Hz 或采用两级超前。"
        )
    return f"{mode_label}: 推荐 {candidate.label}，{current}，{predicted_text}。"


def _candidate_suggestion(
    kind: str,
    alpha: float | None,
    beta: float | None,
    zero: float | None,
    pole: float | None,
    lag_zero: float | None,
    lag_pole: float | None,
) -> str:
    parts: list[str] = []
    if kind in (MODE_LEAD, MODE_LEAD_LAG):
        parts.append(
            "超前环节把零点放在 "
            f"{_format_rad_s(zero)}，极点放在 {_format_rad_s(pole)}；"
            "实际电路或数字控制器先按该传递函数仿真，再微调增益。"
        )
    if kind in (MODE_LAG, MODE_LEAD_LAG):
        parts.append(
            "滞后环节把极点放在 "
            f"{_format_rad_s(lag_pole)}，零点放在 {_format_rad_s(lag_zero)}；"
            f"低频增益约提升 {_format_number(beta)} 倍。"
        )
    if not parts:
        return "暂不改变控制器。"
    return " ".join(parts)


def _prepare_frequency_response(omega: Any, magnitude: Any, phase_rad: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega_arr = np.asarray(omega, dtype=float).flatten()
    mag_arr = np.asarray(magnitude, dtype=float).flatten()
    phase_arr = np.asarray(phase_rad, dtype=float).flatten()
    n = min(len(omega_arr), len(mag_arr), len(phase_arr))
    omega_arr = omega_arr[:n]
    mag_arr = mag_arr[:n]
    phase_arr = phase_arr[:n]
    mask = np.isfinite(omega_arr) & np.isfinite(mag_arr) & np.isfinite(phase_arr) & (omega_arr > 0.0) & (mag_arr > 0.0)
    omega_arr = omega_arr[mask]
    mag_arr = mag_arr[mask]
    phase_arr = phase_arr[mask]
    order = np.argsort(omega_arr)
    return omega_arr[order], mag_arr[order], phase_arr[order]


def _level_crossings(x: np.ndarray, y: np.ndarray, level: float) -> list[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float) - float(level)
    crossings: list[float] = []
    for index in range(len(x) - 1):
        y0 = float(y[index])
        y1 = float(y[index + 1])
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0:
            crossings.append(float(x[index]))
            continue
        if y0 * y1 > 0.0:
            continue
        denom = y0 - y1
        if abs(denom) < 1.0e-15:
            continue
        frac = y0 / denom
        crossings.append(float(x[index] + frac * (x[index + 1] - x[index])))
    if len(y) and float(y[-1]) == 0.0:
        crossings.append(float(x[-1]))
    return _dedupe_sorted(crossings)


def _dedupe_sorted(values: list[float]) -> list[float]:
    values = sorted(float(v) for v in values if np.isfinite(v))
    deduped: list[float] = []
    for value in values:
        if not deduped or abs(value - deduped[-1]) > 1.0e-9 * max(1.0, abs(value)):
            deduped.append(value)
    return deduped


def _interp_log_omega(omega: np.ndarray, values: np.ndarray, omega_target: float) -> float | None:
    if len(omega) == 0 or omega_target <= 0.0:
        return None
    logw = np.log(omega)
    target = math.log(float(omega_target))
    if target < float(logw[0]) or target > float(logw[-1]):
        return None
    return float(np.interp(target, logw, values))


def _target_design_point(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_rad: np.ndarray,
    settings: ControlCompensationSettings,
) -> dict[str, float] | None:
    if settings.target_crossover_hz is None:
        return None
    target_wc = float(settings.target_crossover_hz) * 2.0 * math.pi
    plant_mag = _interp_log_omega(omega, magnitude, target_wc)
    phase_deg_arr = np.degrees(np.unwrap(phase_rad))
    plant_phase = _interp_log_omega(omega, phase_deg_arr, target_wc)
    if plant_mag is None or plant_phase is None or plant_mag <= 0.0:
        return None
    phase_for_margin = _phase_for_margin(plant_phase)
    return {
        "omega": float(target_wc),
        "magnitude": float(plant_mag),
        "phase_deg": float(phase_for_margin),
        "phase_margin_deg": float(180.0 + phase_for_margin),
    }


def _gain_to_place_crossover_at(
    omega: np.ndarray,
    magnitude: np.ndarray,
    target_wc: float,
    alpha: float | None,
    lead_t: float | None,
    beta: float | None,
    lag_t: float | None,
) -> float | None:
    plant_mag = _interp_log_omega(omega, magnitude, target_wc)
    if plant_mag is None or plant_mag <= 0.0:
        return None

    s = 1j * float(target_wc)
    comp = 1.0 + 0j
    if alpha is not None and lead_t is not None:
        comp *= (1.0 + s * float(lead_t)) / (1.0 + s * float(alpha) * float(lead_t))
    if beta is not None and lag_t is not None:
        comp *= float(beta) * (1.0 + s * float(lag_t)) / (1.0 + s * float(beta) * float(lag_t))

    comp_mag = abs(comp)
    if comp_mag <= 0.0 or not np.isfinite(comp_mag):
        return None
    return float(1.0 / max(float(plant_mag) * comp_mag, 1.0e-12))


def _phase_for_margin(phase_deg: float) -> float:
    phase = float(phase_deg)
    # For phase margin we need the unwrapped branch around/below -180 deg.
    # Do not fold high-order lag such as -330 deg back to +30 deg; that would
    # turn a negative margin into an impossible "large positive" margin.
    while phase > 90.0:
        phase -= 360.0
    return phase


def _required_phase_boost_at(
    omega: np.ndarray,
    phase_rad: np.ndarray,
    target_wc: float,
    settings: ControlCompensationSettings,
) -> float:
    phase_deg = np.degrees(np.unwrap(phase_rad))
    phase_at_target = _interp_log_omega(omega, phase_deg, target_wc)
    if phase_at_target is None:
        return settings.target_phase_margin_deg + settings.safety_phase_deg
    current_pm_at_target = 180.0 + _phase_for_margin(phase_at_target)
    return float(settings.target_phase_margin_deg - current_pm_at_target + settings.safety_phase_deg)


def _alpha_from_phase_boost(phi_deg: float) -> float:
    phi_rad = math.radians(float(np.clip(phi_deg, 0.1, 89.0)))
    sin_phi = math.sin(phi_rad)
    return float((1.0 - sin_phi) / (1.0 + sin_phi))


def _find_magnitude_crossing(
    omega: np.ndarray,
    magnitude: np.ndarray,
    target_magnitude: float,
    prefer_above: float | None = None,
) -> float | None:
    logw = np.log(omega)
    mag_db = 20.0 * np.log10(np.clip(magnitude, 1.0e-12, None))
    target_db = 20.0 * math.log10(max(float(target_magnitude), 1.0e-12))
    crossings = [float(math.exp(item)) for item in _level_crossings(logw, mag_db, target_db)]
    if not crossings:
        return None
    if prefer_above is not None:
        higher = [item for item in crossings if item >= prefer_above]
        if higher:
            return min(higher, key=lambda item: abs(item - prefer_above))
    return min(crossings, key=lambda item: abs(mag_db[np.argmin(np.abs(omega - item))] - target_db))


def _predict_phase_margin_at(
    omega: np.ndarray,
    magnitude: np.ndarray,
    phase_rad: np.ndarray,
    target_wc: float,
    alpha: float | None,
    lead_t: float | None,
    beta: float | None,
    lag_t: float | None,
) -> float | None:
    phase_deg = np.degrees(np.unwrap(phase_rad))
    plant_phase = _interp_log_omega(omega, phase_deg, target_wc)
    if plant_phase is None:
        return None
    s = 1j * float(target_wc)
    comp = 1.0 + 0j
    if alpha is not None and lead_t is not None:
        comp *= (1.0 + s * lead_t) / (1.0 + s * alpha * lead_t)
    if beta is not None and lag_t is not None:
        comp *= beta * (1.0 + s * lag_t) / (1.0 + s * beta * lag_t)
    phase = _phase_for_margin(float(plant_phase) + math.degrees(math.atan2(comp.imag, comp.real)))
    return float(180.0 + phase)


def _parse_float(value: str | float | int, label: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{label}必须是数字。") from None
    if not np.isfinite(parsed):
        raise ValueError(f"{label}必须是有限数字。")
    return float(parsed)


def _parse_optional_float(value: str | float | int | None, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _parse_float(value, label)


def _format_rad_s(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    return f"{float(value):.3g} rad/s"


def _format_hz(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    return f"{float(value):.3g} Hz"


def _format_deg(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    return f"{float(value):.1f}°"


def _format_db(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    return f"{float(value):.1f} dB"


def _format_seconds(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    return f"{float(value):.3g} s"


def _format_number(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    return f"{float(value):.3g}"
