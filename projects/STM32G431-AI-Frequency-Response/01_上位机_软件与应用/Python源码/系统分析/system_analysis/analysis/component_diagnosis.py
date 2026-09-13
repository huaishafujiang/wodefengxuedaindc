from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

import numpy as np


CIRCUIT_LABEL_TO_TYPE_ORDER = {
    "一阶低通": ("lowpass", 1),
    "二阶低通": ("lowpass", 2),
    "三阶低通": ("lowpass", 3),
    "一阶高通": ("highpass", 1),
    "二阶高通": ("highpass", 2),
    "三阶高通": ("highpass", 3),
    "带通": ("bandpass", 2),
    "带阻": ("bandstop", 2),
}

TYPE_LABELS = {
    "lowpass": "低通",
    "highpass": "高通",
    "bandpass": "带通",
    "bandstop": "带阻",
}

UNIT_FACTORS = {
    "": 1.0,
    "r": 1.0,
    "ohm": 1.0,
    "k": 1.0e3,
    "kohm": 1.0e3,
    "m": 1.0e-3,
    "meg": 1.0e6,
    "g": 1.0e9,
    "p": 1.0e-12,
    "n": 1.0e-9,
    "u": 1.0e-6,
    "μ": 1.0e-6,
    "f": 1.0,
}

MIN_POINTS = 6
MIN_SIGNIFICANT_DEVIATION = 0.08


@dataclass(frozen=True)
class ComponentValue:
    name: str
    nominal: float
    kind: str


@dataclass(frozen=True)
class ComponentDiagnosisProfile:
    enabled: bool = True
    circuit_label: str = "Auto"
    circuit_type: str = "lowpass"
    order: int = 1
    resistors: list[ComponentValue] = field(default_factory=list)
    capacitors: list[ComponentValue] = field(default_factory=list)
    calibrated: bool = False


@dataclass(frozen=True)
class ComponentDeviationCandidate:
    component: str
    kind: str
    nominal_value: float
    fitted_value: float
    deviation_percent: float
    confidence: float
    rmse_db: float
    baseline_rmse_db: float
    evidence: str
    suggestion: str


@dataclass(frozen=True)
class ComponentDeviationReport:
    enabled: bool
    circuit_label: str
    circuit_type: str
    order: int
    baseline_rmse_db: float | None
    best_rmse_db: float | None
    summary: str
    notes: list[str]
    candidates: list[ComponentDeviationCandidate] = field(default_factory=list)

    @property
    def has_significant_deviation(self) -> bool:
        return bool(self.candidates and self.candidates[0].confidence >= 0.55)


def parse_component_value(text: str | float | int) -> float:
    if isinstance(text, (int, float)):
        value = float(text)
        if value <= 0.0 or not np.isfinite(value):
            raise ValueError("元件值必须为正数。")
        return value

    raw_original = str(text or "").strip().replace("Ω", "ohm").replace("ω", "ohm")
    raw = raw_original.lower()
    if not raw:
        raise ValueError("元件值不能为空。")
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?|\.\d+)\s*([a-zA-Zμ]*)", raw_original)
    if not match:
        raise ValueError(f"无法解析元件值: {text}")
    number = float(match.group(1))
    unit_raw = match.group(2)
    unit = "meg" if unit_raw in ("M", "Mohm", "MOhm") else unit_raw.lower()
    if unit not in UNIT_FACTORS:
        raise ValueError(f"不支持的元件单位: {unit}")
    value = number * UNIT_FACTORS[unit]
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError("元件值必须为正数。")
    return float(value)


def format_component_value(value: float, kind: str) -> str:
    value = float(value)
    if kind == "R":
        if abs(value) >= 1.0e6:
            return f"{value / 1.0e6:.3g} Mohm"
        if abs(value) >= 1.0e3:
            return f"{value / 1.0e3:.3g} kohm"
        return f"{value:.3g} ohm"
    if abs(value) >= 1.0e-6:
        return f"{value / 1.0e-6:.3g} uF"
    if abs(value) >= 1.0e-9:
        return f"{value / 1.0e-9:.3g} nF"
    if abs(value) >= 1.0e-12:
        return f"{value / 1.0e-12:.3g} pF"
    return f"{value:.3g} F"


def parse_component_list(text: str, kind: str, minimum_count: int = 1) -> list[ComponentValue]:
    pieces = [item.strip() for item in re.split(r"[,，;\s]+", str(text or "").strip()) if item.strip()]
    if not pieces:
        raise ValueError(f"{kind} 标称列表不能为空。")
    values = [
        ComponentValue(name=f"{kind}{index}", nominal=parse_component_value(piece), kind=kind)
        for index, piece in enumerate(pieces, start=1)
    ]
    if len(values) < minimum_count:
        raise ValueError(f"{kind} 至少需要 {minimum_count} 个标称值。")
    return values


def profile_from_inputs(
    circuit_label: str,
    resistor_text: str,
    capacitor_text: str,
    resistor2_text: str = "",
    capacitor2_text: str = "",
    *,
    enabled: bool = True,
    calibrated: bool = False,
) -> ComponentDiagnosisProfile:
    circuit_label = circuit_label if circuit_label in CIRCUIT_LABEL_TO_TYPE_ORDER else "Auto"
    if circuit_label == "Auto":
        raise ValueError("请先在“期望电路”中选择具体电路类型，再启用元件偏差诊断。")
    circuit_type, order = CIRCUIT_LABEL_TO_TYPE_ORDER[circuit_label]

    if circuit_type in ("lowpass", "highpass"):
        resistors = parse_component_list(resistor_text, "R", minimum_count=1)
        capacitors = parse_component_list(capacitor_text, "C", minimum_count=1)
        resistors = _expand_components(resistors, order)
        capacitors = _expand_components(capacitors, order)
    else:
        resistors = parse_component_list(resistor_text, "R", minimum_count=1)
        capacitors = parse_component_list(capacitor_text, "C", minimum_count=1)
        second_r = parse_component_list(resistor2_text or resistor_text, "R", minimum_count=1)
        second_c = parse_component_list(capacitor2_text or capacitor_text, "C", minimum_count=1)
        resistors = [ComponentValue("R1", resistors[0].nominal, "R"), ComponentValue("R2", second_r[0].nominal, "R")]
        capacitors = [ComponentValue("C1", capacitors[0].nominal, "C"), ComponentValue("C2", second_c[0].nominal, "C")]

    return ComponentDiagnosisProfile(
        enabled=enabled,
        circuit_label=circuit_label,
        circuit_type=circuit_type,
        order=order,
        resistors=resistors,
        capacitors=capacitors,
        calibrated=calibrated,
    )


def _expand_components(values: list[ComponentValue], count: int) -> list[ComponentValue]:
    count = max(1, int(count))
    source = list(values)
    if len(source) >= count:
        return [
            ComponentValue(name=f"{source[index].kind}{index + 1}", nominal=source[index].nominal, kind=source[index].kind)
            for index in range(count)
        ]
    expanded = source[:]
    while len(expanded) < count:
        last = expanded[-1]
        expanded.append(ComponentValue(name=f"{last.kind}{len(expanded) + 1}", nominal=last.nominal, kind=last.kind))
    return expanded


def diagnose_component_deviation(
    omega: Any,
    magnitude: Any,
    phase_rad: Any | None,
    profile: ComponentDiagnosisProfile | None,
) -> ComponentDeviationReport:
    if profile is None or not profile.enabled:
        return ComponentDeviationReport(False, "Auto", "unknown", 0, None, None, "元件偏差诊断未启用。", [])

    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    n = min(len(omega), len(magnitude))
    if n < MIN_POINTS:
        return ComponentDeviationReport(
            True,
            profile.circuit_label,
            profile.circuit_type,
            profile.order,
            None,
            None,
            "有效频点太少，无法进行元件偏差诊断。",
            ["建议至少采集 6 个以上频点。"],
        )
    omega = omega[:n]
    magnitude = magnitude[:n]
    valid = np.isfinite(omega) & (omega > 0.0) & np.isfinite(magnitude) & (magnitude > 0.0)
    if int(np.count_nonzero(valid)) < MIN_POINTS:
        return ComponentDeviationReport(
            True,
            profile.circuit_label,
            profile.circuit_type,
            profile.order,
            None,
            None,
            "有效频点不足，无法进行元件偏差诊断。",
            ["请检查实测幅值是否为 0、NaN 或削顶。"],
        )

    freq_hz = omega[valid] / (2.0 * math.pi)
    measured_db = 20.0 * np.log10(np.clip(magnitude[valid], 1.0e-12, None))
    nominal_db = response_db(freq_hz, profile)
    gain_offset_db = _best_gain_offset(measured_db, nominal_db)
    baseline_residual = gain_offset_db + nominal_db - measured_db
    baseline_rmse = _rmse(baseline_residual)

    candidates: list[ComponentDeviationCandidate] = []
    for component in [*profile.resistors, *profile.capacitors]:
        best_value, best_rmse = _fit_single_component(freq_hz, measured_db, profile, component)
        deviation = (best_value / component.nominal) - 1.0
        confidence = _candidate_confidence(abs(deviation), baseline_rmse, best_rmse)
        direction = "偏大" if deviation > 0.0 else "偏小"
        evidence = (
            f"标称模型 RMSE={baseline_rmse:.2f} dB，调整 {component.name} 后 RMSE={best_rmse:.2f} dB；"
            f"{component.name} 估计{direction} {abs(deviation) * 100.0:.1f}%。"
        )
        suggestion = (
            f"检查 {component.name}，标称 {format_component_value(component.nominal, component.kind)}，"
            f"拟合约 {format_component_value(best_value, component.kind)}；建议换回标称值或按设计重新核对该元件。"
        )
        candidates.append(
            ComponentDeviationCandidate(
                component=component.name,
                kind=component.kind,
                nominal_value=float(component.nominal),
                fitted_value=float(best_value),
                deviation_percent=float(deviation * 100.0),
                confidence=float(confidence),
                rmse_db=float(best_rmse),
                baseline_rmse_db=float(baseline_rmse),
                evidence=evidence,
                suggestion=suggestion,
            )
        )

    candidates.sort(key=lambda item: _candidate_sort_key(profile, item), reverse=True)
    significant = [item for item in candidates if abs(item.deviation_percent) >= MIN_SIGNIFICANT_DEVIATION * 100.0]
    top = significant[:3]
    best_rmse = top[0].rmse_db if top else (candidates[0].rmse_db if candidates else baseline_rmse)

    notes: list[str] = []
    if not profile.calibrated:
        notes.append("本次未检测到参考校正标记；元件偏差诊断置信度会偏保守，建议先做直通参考校正。")
    if profile.circuit_type == "bandstop":
        notes.append("带阻 V1 使用等效中心频率模型，只判断 R/C 偏移方向，不保证唯一定位 Twin-T 每个支路。")
    if not top:
        summary = (
            f"{profile.circuit_label} 标称模板与实测基本一致；标称 RMSE={baseline_rmse:.2f} dB，"
            "未发现显著单元件偏差。"
        )
    elif top[0].confidence < 0.55:
        summary = (
            f"疑似存在元件偏差，但单元件解释置信度偏低；Top1 为 {top[0].component}，"
            f"估计偏差 {top[0].deviation_percent:+.1f}%。"
        )
        notes.append("多个元件同时偏差、运放/面包板寄生或测量链误差都可能造成类似频响。")
    else:
        summary = (
            f"最可能偏差元件: {top[0].component}，估计值 "
            f"{format_component_value(top[0].fitted_value, top[0].kind)}，"
            f"相对标称 {top[0].deviation_percent:+.1f}%，置信度 {top[0].confidence:.2f}。"
        )
    if len(top) >= 2 and (top[0].confidence - top[1].confidence) < 0.12:
        notes.append("Top1/Top2 置信度接近，建议人工测量候选元件或补扫拐点附近频段。")
    if top and _has_rc_product_ambiguity(profile, top):
        notes.append("同一 RC 级的 R 与 C 对截止频率呈乘积等效，V1 按固定偏好排序；请用万用表/LCR 表确认候选元件。")

    return ComponentDeviationReport(
        enabled=True,
        circuit_label=profile.circuit_label,
        circuit_type=profile.circuit_type,
        order=profile.order,
        baseline_rmse_db=float(baseline_rmse),
        best_rmse_db=float(best_rmse),
        summary=summary,
        notes=notes,
        candidates=top,
    )


def response_db(freq_hz: Any, profile: ComponentDiagnosisProfile) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=float)
    h = response_complex(freq_hz, profile)
    return 20.0 * np.log10(np.clip(np.abs(h), 1.0e-12, None))


def response_complex(freq_hz: Any, profile: ComponentDiagnosisProfile) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=float)
    s = 1j * 2.0 * math.pi * np.clip(freq_hz, 1.0e-12, None)
    if profile.circuit_type == "lowpass":
        h = np.ones_like(s, dtype=complex)
        for resistor, capacitor in zip(profile.resistors, profile.capacitors):
            wc = _stage_wc(resistor.nominal, capacitor.nominal)
            h = h * (wc / (s + wc))
        return h
    if profile.circuit_type == "highpass":
        h = np.ones_like(s, dtype=complex)
        for resistor, capacitor in zip(profile.resistors, profile.capacitors):
            wc = _stage_wc(resistor.nominal, capacitor.nominal)
            h = h * (s / (s + wc))
        return h
    if profile.circuit_type == "bandpass":
        low_edge = _stage_wc(profile.resistors[0].nominal, profile.capacitors[0].nominal)
        high_edge = _stage_wc(profile.resistors[1].nominal, profile.capacitors[1].nominal)
        return (s / (s + low_edge)) * (high_edge / (s + high_edge))
    if profile.circuit_type == "bandstop":
        wc1 = _stage_wc(profile.resistors[0].nominal, profile.capacitors[0].nominal)
        wc2 = _stage_wc(profile.resistors[1].nominal, profile.capacitors[1].nominal)
        w0 = math.sqrt(max(wc1 * wc2, 1.0e-18))
        q = float(np.clip(w0 / max(abs(wc2 - wc1), w0 * 0.18), 0.25, 12.0))
        return (s**2 + w0**2) / (s**2 + (w0 / q) * s + w0**2)
    return np.ones_like(s, dtype=complex)


def format_component_report_lines(report: ComponentDeviationReport | None) -> list[str]:
    if report is None or not report.enabled:
        return []
    lines = [
        "=== 元件偏差诊断 V1 ===",
        f"模板: {report.circuit_label}",
        f"结论: {report.summary}",
    ]
    if report.baseline_rmse_db is not None:
        lines.append(f"标称模板误差: RMSE={report.baseline_rmse_db:.2f} dB")
    if report.best_rmse_db is not None:
        lines.append(f"最佳单元件解释误差: RMSE={report.best_rmse_db:.2f} dB")
    if report.candidates:
        lines.append(
            "Top候选: "
            + " / ".join(
                f"{item.component} {item.deviation_percent:+.1f}% conf={item.confidence:.2f}"
                for item in report.candidates
            )
        )
        for item in report.candidates:
            lines.append(f"- {item.evidence} 建议: {item.suggestion}")
    else:
        lines.append("Top候选: 暂无显著偏差。")
    if report.notes:
        lines.append("提示: " + "；".join(report.notes))
    lines.append("")
    return lines


def _stage_wc(resistance_ohm: float, capacitance_f: float) -> float:
    return 1.0 / max(float(resistance_ohm) * float(capacitance_f), 1.0e-18)


def _best_gain_offset(measured_db: np.ndarray, model_db: np.ndarray) -> float:
    finite = np.isfinite(measured_db) & np.isfinite(model_db)
    if not np.any(finite):
        return 0.0
    return float(np.mean(measured_db[finite] - model_db[finite]))


def _rmse(residual: np.ndarray) -> float:
    finite = np.asarray(residual, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("inf")
    return float(np.sqrt(np.mean(finite**2)))


def _fit_single_component(
    freq_hz: np.ndarray,
    measured_db: np.ndarray,
    profile: ComponentDiagnosisProfile,
    component: ComponentValue,
) -> tuple[float, float]:
    factors = np.geomspace(0.25, 4.0, 121)
    best_value = component.nominal
    best_rmse = float("inf")
    for factor in factors:
        trial_value = component.nominal * float(factor)
        trial_profile = _profile_with_component_value(profile, component.name, trial_value)
        model_db = response_db(freq_hz, trial_profile)
        gain = _best_gain_offset(measured_db, model_db)
        rmse = _rmse(gain + model_db - measured_db)
        if rmse < best_rmse:
            best_rmse = rmse
            best_value = trial_value

    for span in (1.18, 1.035):
        center = best_value
        low = max(center / span, component.nominal * 0.2)
        high = min(center * span, component.nominal * 5.0)
        for trial_value in np.geomspace(low, high, 61):
            trial_profile = _profile_with_component_value(profile, component.name, float(trial_value))
            model_db = response_db(freq_hz, trial_profile)
            gain = _best_gain_offset(measured_db, model_db)
            rmse = _rmse(gain + model_db - measured_db)
            if rmse < best_rmse:
                best_rmse = rmse
                best_value = float(trial_value)
    return float(best_value), float(best_rmse)


def _profile_with_component_value(
    profile: ComponentDiagnosisProfile,
    component_name: str,
    value: float,
) -> ComponentDiagnosisProfile:
    resistors = [
        ComponentValue(item.name, float(value), item.kind) if item.name == component_name else item
        for item in profile.resistors
    ]
    capacitors = [
        ComponentValue(item.name, float(value), item.kind) if item.name == component_name else item
        for item in profile.capacitors
    ]
    return ComponentDiagnosisProfile(
        enabled=profile.enabled,
        circuit_label=profile.circuit_label,
        circuit_type=profile.circuit_type,
        order=profile.order,
        resistors=resistors,
        capacitors=capacitors,
        calibrated=profile.calibrated,
    )


def _candidate_confidence(deviation_abs: float, baseline_rmse: float, best_rmse: float) -> float:
    if not np.isfinite(best_rmse) or not np.isfinite(baseline_rmse):
        return 0.0
    improvement = max(0.0, baseline_rmse - best_rmse)
    improvement_score = np.clip(improvement / max(baseline_rmse, 0.8), 0.0, 1.0)
    deviation_score = np.clip((deviation_abs - 0.03) / 0.22, 0.0, 1.0)
    fit_score = np.clip(1.0 - best_rmse / 8.0, 0.0, 1.0)
    return float(np.clip(0.50 * improvement_score + 0.30 * deviation_score + 0.20 * fit_score, 0.0, 0.99))


def _candidate_sort_key(
    profile: ComponentDiagnosisProfile,
    candidate: ComponentDeviationCandidate,
) -> tuple[float, float, int]:
    return (
        round(float(candidate.confidence), 6),
        -round(float(candidate.rmse_db), 6),
        _component_tie_priority(profile, candidate.component),
    )


def _component_tie_priority(profile: ComponentDiagnosisProfile, component_name: str) -> int:
    kind = component_name[:1]
    try:
        index = int(component_name[1:] or "1")
    except ValueError:
        index = 1

    if profile.circuit_type == "lowpass":
        if profile.order <= 1:
            kind_priority = 2 if kind == "C" else 1
        else:
            kind_priority = 2 if kind == "R" else 1
    elif profile.circuit_type == "highpass":
        kind_priority = 2 if kind == "R" else 1
    else:
        kind_priority = 2 if kind == "R" else 1
    return index * 10 + kind_priority


def _has_rc_product_ambiguity(
    profile: ComponentDiagnosisProfile,
    candidates: list[ComponentDeviationCandidate],
) -> bool:
    if len(candidates) < 2:
        return False
    best = candidates[0]
    for other in candidates[1:]:
        if abs(best.confidence - other.confidence) > 0.08:
            continue
        if abs(best.rmse_db - other.rmse_db) > 0.08:
            continue
        if best.component[1:] == other.component[1:] and best.kind != other.kind:
            return True
    return False
