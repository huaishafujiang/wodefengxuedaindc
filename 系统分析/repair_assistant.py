from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from circuit_batch_fit import BatchFitReport
from component_diagnosis import ComponentDeviationReport, format_component_value


@dataclass(frozen=True)
class RepairAssistantReport:
    summary: str
    fault_explanation: tuple[str, ...]
    sweep_suggestions: tuple[str, ...]
    repair_steps: tuple[str, ...]


def build_repair_assistant_report(
    diagnosis: Any | None,
    batch_report: BatchFitReport | None,
    *,
    point_count: int = 0,
) -> RepairAssistantReport:
    summary = _summary(diagnosis, batch_report, point_count)
    fault_explanation = tuple(_fault_explanations(diagnosis, batch_report))
    sweep_suggestions = tuple(_sweep_suggestions(diagnosis))
    repair_steps = tuple(_repair_steps(diagnosis, batch_report, sweep_suggestions))
    return RepairAssistantReport(
        summary=summary,
        fault_explanation=fault_explanation,
        sweep_suggestions=sweep_suggestions,
        repair_steps=repair_steps,
    )


def format_repair_assistant_report_lines(report: RepairAssistantReport | None) -> list[str]:
    if report is None:
        return []
    lines = [
        "=== AI 电路诊断助手 V3 ===",
        f"综合结论: {report.summary}",
        "故障解释: " + ("；".join(report.fault_explanation) if report.fault_explanation else "暂未发现明确硬件故障。"),
        "建议补扫频段: " + ("；".join(report.sweep_suggestions) if report.sweep_suggestions else "当前数据已覆盖主要特征频段。"),
        "维修步骤:",
    ]
    for index, step in enumerate(report.repair_steps, start=1):
        lines.append(f"{index}. {step}")
    lines.append("")
    return lines


def _summary(diagnosis: Any | None, batch_report: BatchFitReport | None, point_count: int) -> str:
    if diagnosis is None:
        return "尚未完成智能诊断。"
    system_label = str(getattr(diagnosis, "system_label", "未知系统"))
    confidence = float(getattr(diagnosis, "confidence", 0.0) or 0.0)
    best = batch_report.best if batch_report is not None else None
    if best is not None:
        return (
            f"实测 {point_count} 点, AI 判定为 {system_label} (置信度 {confidence:.2f}); "
            f"原理图批量拟合最接近 {best.circuit_label}, RMSE={best.rmse_db:.2f} dB。"
        )
    return f"实测 {point_count} 点, AI 判定为 {system_label} (置信度 {confidence:.2f})。"


def _fault_explanations(diagnosis: Any | None, batch_report: BatchFitReport | None) -> list[str]:
    explanations: list[str] = []
    if diagnosis is None:
        return explanations
    for finding in getattr(diagnosis, "fault_findings", []) or []:
        explanations.append(
            f"{getattr(finding, 'name', '故障规则')}[{getattr(finding, 'severity', '')}]: "
            f"{getattr(finding, 'evidence', '')}"
        )

    component_report = getattr(diagnosis, "component_deviation_report", None)
    explanations.extend(_component_explanations(component_report))

    if batch_report is not None and batch_report.best is not None:
        best = batch_report.best
        if best.change is not None:
            explanations.append(f"批量仿真显示单独改变 {best.change.text()} 时最接近实测。")
        else:
            explanations.append("原理图标称参数已经能较好解释实测曲线, 优先检查接线和测量链。")
    return explanations


def _component_explanations(report: ComponentDeviationReport | None) -> list[str]:
    if report is None or not report.enabled:
        return []
    if report.candidates:
        top = report.candidates[0]
        return [
            f"元件偏差拟合 Top1: {top.component} 估计为 "
            f"{format_component_value(top.fitted_value, top.kind)}, "
            f"相对标称 {top.deviation_percent:+.1f}%, 置信度 {top.confidence:.2f}。"
        ]
    return [report.summary]


def _sweep_suggestions(diagnosis: Any | None) -> list[str]:
    if diagnosis is None:
        return []
    commands = list(getattr(diagnosis, "adaptive_sweep_commands", []) or [])
    suggestions = list(getattr(diagnosis, "next_test_suggestions", []) or [])
    out: list[str] = []
    if commands:
        out.append("执行 " + " | ".join(commands))
    out.extend(suggestions[:2])
    return out


def _repair_steps(
    diagnosis: Any | None,
    batch_report: BatchFitReport | None,
    sweep_suggestions: tuple[str, ...],
) -> list[str]:
    steps: list[str] = [
        "断电检查跳线选择, 确认 Vin/VOUT、OUT1/OUT2/OUT3、BP_OUT/BR_OUT 接到当前期望电路对应节点。",
    ]
    candidate = _top_component_candidate(diagnosis, batch_report)
    if candidate is not None:
        name, nominal, fitted, kind = candidate
        steps.append(
            f"单独检查 {name}: 标称 {format_component_value(nominal, kind)}, "
            f"拟合约 {format_component_value(fitted, kind)}；先只改这一颗元件后复测。"
        )
    else:
        steps.append("原理图标称参数未显示明显单元件偏差, 先检查供电、VREF、面包板接触和输入输出通道。")

    for finding in getattr(diagnosis, "fault_findings", [])[:2] if diagnosis is not None else []:
        suggestion = str(getattr(finding, "suggestion", "")).strip()
        if suggestion:
            steps.append(suggestion)

    if sweep_suggestions:
        steps.append("按建议补扫频段执行一次加密扫频, 用补扫后的合并数据再生成诊断报告。")
    steps.append("修复或改值后保留同一激励幅值和扫频范围复测, 对比 RMSE、截止/中心频率和 Top候选是否回到标称。")
    return _dedupe_steps(steps)


def _top_component_candidate(
    diagnosis: Any | None,
    batch_report: BatchFitReport | None,
) -> tuple[str, float, float, str] | None:
    component_report = getattr(diagnosis, "component_deviation_report", None) if diagnosis is not None else None
    if component_report is not None and getattr(component_report, "candidates", None):
        top = component_report.candidates[0]
        return top.component, float(top.nominal_value), float(top.fitted_value), str(top.kind)

    best = batch_report.best if batch_report is not None else None
    if best is not None and best.change is not None and np.isfinite(best.rmse_db):
        change = best.change
        return change.component, change.nominal_value, change.trial_value, change.kind
    return None


def _dedupe_steps(steps: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for step in steps:
        normalized = " ".join(str(step).split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out
