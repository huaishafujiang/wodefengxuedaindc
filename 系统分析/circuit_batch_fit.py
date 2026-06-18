from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from component_diagnosis import (
    ComponentDiagnosisProfile,
    ComponentDeviationReport,
    ComponentValue,
    diagnose_component_deviation,
    format_component_value,
    response_db,
)
from schematic_reader import CircuitLibrary, build_default_circuit_library


DEFAULT_SINGLE_COMPONENT_FACTORS = tuple(float(x) for x in np.geomspace(0.35, 2.85, 49))


@dataclass(frozen=True)
class SingleComponentChange:
    component: str
    kind: str
    nominal_value: float
    trial_value: float
    factor: float

    def text(self) -> str:
        return (
            f"{self.component}: {format_component_value(self.nominal_value, self.kind)} -> "
            f"{format_component_value(self.trial_value, self.kind)} ({self.factor:.2f}x)"
        )


@dataclass(frozen=True)
class CircuitFitCandidate:
    circuit_label: str
    circuit_type: str
    order: int
    rmse_db: float
    r_squared: float
    gain_offset_db: float
    change: SingleComponentChange | None = None
    evidence: str = ""


@dataclass(frozen=True)
class BatchFitReport:
    source_summary: str
    candidates_tested: int
    top_candidates: tuple[CircuitFitCandidate, ...]
    component_report: ComponentDeviationReport | None
    notes: tuple[str, ...] = ()

    @property
    def best(self) -> CircuitFitCandidate | None:
        return self.top_candidates[0] if self.top_candidates else None

    @property
    def has_single_component_shift(self) -> bool:
        best = self.best
        return bool(best is not None and best.change is not None and abs(best.change.factor - 1.0) >= 0.03)


def fit_measurement_to_schematic_library(
    omega,
    magnitude,
    phase_rad=None,
    *,
    library: CircuitLibrary | None = None,
    expected_label: str | None = None,
    calibrated: bool = False,
    factors: Iterable[float] | None = None,
    top_n: int = 8,
) -> BatchFitReport:
    library = library or build_default_circuit_library()
    if factors is None:
        factors = DEFAULT_SINGLE_COMPONENT_FACTORS
    factors = tuple(float(item) for item in factors)
    freq_hz, measured_db = _prepare_measurement(omega, magnitude)
    if len(freq_hz) < 6:
        return BatchFitReport(
            source_summary=library.source_summary(),
            candidates_tested=0,
            top_candidates=(),
            component_report=None,
            notes=("有效频点太少, 原理图批量拟合未执行。",),
        )

    candidates: list[CircuitFitCandidate] = []
    tested = 0
    ordered_labels = _ordered_labels(library, expected_label)
    for label in ordered_labels:
        nominal_profile = library.profile_for_label(label, calibrated=calibrated)
        if nominal_profile is None:
            continue
        for profile, change in _candidate_profiles(nominal_profile, factors):
            rmse_db, r_squared, gain_offset_db = _fit_profile(freq_hz, measured_db, profile)
            tested += 1
            evidence = _candidate_evidence(label, rmse_db, r_squared, change)
            candidates.append(
                CircuitFitCandidate(
                    circuit_label=label,
                    circuit_type=profile.circuit_type,
                    order=profile.order,
                    rmse_db=float(rmse_db),
                    r_squared=float(r_squared),
                    gain_offset_db=float(gain_offset_db),
                    change=change,
                    evidence=evidence,
                )
            )

    candidates.sort(key=lambda item: (round(item.rmse_db, 8), -round(item.r_squared, 8), -_change_priority(item)))
    top = tuple(candidates[: max(1, int(top_n))])
    component_report = None
    if top:
        nominal_best = library.profile_for_label(top[0].circuit_label, calibrated=calibrated)
        if nominal_best is not None:
            component_report = diagnose_component_deviation(omega, magnitude, phase_rad, nominal_best)

    notes = [
        "当前 V2 批量仿真只枚举单个阻值或容值变化；多个元件同时改变留到后续优化。",
        "每个候选会自动补偿一个整体增益偏移, 避免把测量链增益误差误判成元件误差。",
    ]
    if expected_label and top and top[0].circuit_label != expected_label:
        notes.append(f"期望电路为 {expected_label}, 但实测更接近 {top[0].circuit_label}, 请检查跳线和输出节点。")
    return BatchFitReport(
        source_summary=library.source_summary(),
        candidates_tested=tested,
        top_candidates=top,
        component_report=component_report,
        notes=tuple(notes),
    )


def apply_single_component_change(
    profile: ComponentDiagnosisProfile,
    component_name: str,
    new_value: float,
) -> ComponentDiagnosisProfile:
    component_names = {item.name for item in [*profile.resistors, *profile.capacitors]}
    if component_name not in component_names:
        raise ValueError(f"未在 {profile.circuit_label} 中找到元件 {component_name}。")
    new_value = float(new_value)
    if not np.isfinite(new_value) or new_value <= 0.0:
        raise ValueError("新的阻值/容值必须为正数。")
    return _profile_with_component_value(profile, component_name, new_value)


def format_batch_fit_report_lines(report: BatchFitReport | None) -> list[str]:
    if report is None:
        return []
    lines = [
        "=== 原理图候选参数批量拟合 V2 ===",
        f"原理图来源: {report.source_summary}",
        f"流程: 读取原理图 -> 生成候选参数 -> 批量仿真 -> 和实测拟合",
        f"候选数量: {report.candidates_tested}",
    ]
    best = report.best
    if best is None:
        lines.append("结论: 有效数据不足, 未生成候选拟合。")
    else:
        change_text = "标称参数" if best.change is None else best.change.text()
        lines.append(
            f"最佳匹配: {best.circuit_label}, RMSE={best.rmse_db:.2f} dB, "
            f"R2={best.r_squared:.3f}, 增益偏移={best.gain_offset_db:+.2f} dB"
        )
        lines.append(f"最佳参数候选: {change_text}")
        lines.append(
            "Top候选: "
            + " / ".join(
                f"{item.circuit_label}"
                f"{'' if item.change is None else ':' + item.change.component}"
                f" RMSE={item.rmse_db:.2f}"
                for item in report.top_candidates[:5]
            )
        )
    if report.notes:
        lines.append("说明: " + "；".join(report.notes))
    lines.append("")
    return lines


def _prepare_measurement(omega, magnitude) -> tuple[np.ndarray, np.ndarray]:
    omega = np.asarray(omega, dtype=float).flatten()
    magnitude = np.asarray(magnitude, dtype=float).flatten()
    n = min(len(omega), len(magnitude))
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    omega = omega[:n]
    magnitude = magnitude[:n]
    valid = np.isfinite(omega) & (omega > 0.0) & np.isfinite(magnitude) & (magnitude > 0.0)
    if not np.any(valid):
        return np.array([], dtype=float), np.array([], dtype=float)
    freq_hz = omega[valid] / (2.0 * math.pi)
    measured_db = 20.0 * np.log10(np.clip(magnitude[valid], 1.0e-12, None))
    order = np.argsort(freq_hz)
    return freq_hz[order], measured_db[order]


def _ordered_labels(library: CircuitLibrary, expected_label: str | None) -> list[str]:
    labels = library.labels()
    if expected_label in labels:
        return [expected_label, *[label for label in labels if label != expected_label]]
    return labels


def _candidate_profiles(
    profile: ComponentDiagnosisProfile,
    factors: Iterable[float],
) -> Iterable[tuple[ComponentDiagnosisProfile, SingleComponentChange | None]]:
    yield profile, None
    for component in [*profile.resistors, *profile.capacitors]:
        for factor in factors:
            if abs(float(factor) - 1.0) < 1.0e-9:
                continue
            trial_value = component.nominal * float(factor)
            change = SingleComponentChange(
                component=component.name,
                kind=component.kind,
                nominal_value=float(component.nominal),
                trial_value=float(trial_value),
                factor=float(factor),
            )
            yield _profile_with_component_value(profile, component.name, trial_value), change


def _fit_profile(
    freq_hz: np.ndarray,
    measured_db: np.ndarray,
    profile: ComponentDiagnosisProfile,
) -> tuple[float, float, float]:
    model_db = response_db(freq_hz, profile)
    finite = np.isfinite(measured_db) & np.isfinite(model_db)
    if not np.any(finite):
        return float("inf"), 0.0, 0.0
    measured = measured_db[finite]
    model = model_db[finite]
    gain_offset = float(np.mean(measured - model))
    residual = gain_offset + model - measured
    rmse = float(np.sqrt(np.mean(residual**2)))
    total = float(np.sum((measured - float(np.mean(measured))) ** 2))
    r_squared = 0.0 if total <= 1.0e-12 else 1.0 - float(np.sum(residual**2)) / total
    return rmse, float(np.clip(r_squared, -1.0, 1.0)), gain_offset


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


def _candidate_evidence(
    label: str,
    rmse_db: float,
    r_squared: float,
    change: SingleComponentChange | None,
) -> str:
    change_text = "标称参数" if change is None else change.text()
    return f"{label} {change_text}: RMSE={rmse_db:.2f} dB, R2={r_squared:.3f}"


def _change_priority(candidate: CircuitFitCandidate) -> int:
    change = candidate.change
    if change is None:
        return 0
    try:
        index = int(change.component[1:] or "1")
    except ValueError:
        index = 1
    kind_priority = 2 if change.kind == "R" else 1
    return index * 10 + kind_priority
