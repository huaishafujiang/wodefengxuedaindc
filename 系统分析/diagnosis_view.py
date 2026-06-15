from __future__ import annotations

from typing import Any

import numpy as np


CIRCUIT_LABELS = {
    "lowpass": "低通",
    "highpass": "高通",
    "bandpass": "带通",
    "bandstop": "带阻",
    "unknown": "未知",
    "other": "未知",
}


def format_hz(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    if abs(value) >= 1000.0:
        return f"{value:.0f} Hz"
    return f"{value:.1f} Hz"


def top_candidates_text(diagnosis: Any | None) -> str:
    if diagnosis is None:
        return "暂无"
    items = []
    for candidate in getattr(diagnosis, "candidates", [])[:3]:
        raw_label = getattr(candidate, "circuit_type", "unknown")
        label = CIRCUIT_LABELS.get(raw_label, raw_label)
        order = getattr(candidate, "order_estimate", "")
        confidence = float(getattr(candidate, "confidence", 0.0))
        items.append(f"{label}/{order}阶 {confidence:.2f}")
    return " / ".join(items) if items else "暂无"


def key_frequency_text(diagnosis: Any | None) -> str:
    if diagnosis is None:
        return "暂无"
    circuit_type = getattr(diagnosis, "circuit_type", "")
    if circuit_type in ("lowpass", "highpass"):
        return f"截止频率 {format_hz(getattr(diagnosis, 'cutoff_frequency_hz', None))}"
    if circuit_type == "bandpass":
        return (
            f"中心 {format_hz(getattr(diagnosis, 'center_frequency_hz', None))}, "
            f"带宽 {format_hz(getattr(diagnosis, 'bandwidth_hz', None))}"
        )
    if circuit_type == "bandstop":
        return (
            f"陷波中心 {format_hz(getattr(diagnosis, 'center_frequency_hz', None))}, "
            f"阻带宽度 {format_hz(getattr(diagnosis, 'bandwidth_hz', None))}"
        )
    return "关键频率无法稳定估计"


def structured_diagnosis_sections(session: Any | None) -> dict[str, str]:
    if session is None:
        return {
            "判定": "暂无测量",
            "置信度": "-",
            "Top3": "暂无",
            "关键频率": "暂无",
            "测量健康": "暂无",
            "测量质量": "暂无",
            "故障证据": "暂无",
            "下一步建议": "暂无",
        }
    if getattr(session, "error", None):
        return {
            "判定": "无效测量",
            "置信度": "-",
            "Top3": "暂无",
            "关键频率": "暂无",
            "测量健康": "无效",
            "测量质量": str(session.error),
            "故障证据": "分析未完成",
            "下一步建议": "检查采集数据后重新测量",
        }
    diagnosis = getattr(session, "ai_diagnosis", None)
    if diagnosis is None:
        return {
            "判定": "分析中",
            "置信度": "-",
            "Top3": "暂无",
            "关键频率": "暂无",
            "测量健康": "等待分析线程返回",
            "测量质量": "等待分析线程返回",
            "故障证据": "暂无",
            "下一步建议": "暂无",
        }

    findings = getattr(diagnosis, "fault_findings", []) or []
    evidence = "；".join(
        f"{getattr(item, 'name', '')}({getattr(item, 'severity', '')}): {getattr(item, 'evidence', '')}"
        for item in findings
    )
    if not evidence:
        evidence = "暂未触发故障知识库规则"

    return {
        "判定": str(getattr(diagnosis, "system_label", "未知")),
        "置信度": f"{float(getattr(diagnosis, 'confidence', 0.0)):.2f}",
        "Top3": top_candidates_text(diagnosis),
        "关键频率": key_frequency_text(diagnosis),
        "测量健康": str(getattr(diagnosis, "measurement_health", "") or "暂无"),
        "测量质量": "；".join(getattr(diagnosis, "measurement_quality", []) or ["暂无"]),
        "故障证据": evidence,
        "下一步建议": "；".join(getattr(diagnosis, "next_test_suggestions", []) or ["暂无"]),
    }


def diagnosis_clipboard_text(session: Any | None) -> str:
    sections = structured_diagnosis_sections(session)
    lines = [f"{key}: {value}" for key, value in sections.items()]
    calibration_label = getattr(session, "calibration_label", "") if session is not None else ""
    if calibration_label:
        lines.append(f"参考校正: {calibration_label}")
    remark = getattr(session, "remark", "") if session is not None else ""
    if remark:
        lines.append(f"备注: {remark}")
    return "\n".join(lines)
