from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

import numpy as np

from ai_diagnosis import transfer_fit_reliability_text
from diagnosis_view import structured_diagnosis_sections
from filter_analysis import build_filter_report_lines
from transfer_formula import build_transfer_formula_view, formula_css, formula_html, save_formula_png


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return ""
    return f"{f:.{digits}g}"


def _hz_from_omega(value: float | None) -> str:
    if value is None:
        return ""
    try:
        return _fmt(float(value) / (2.0 * np.pi), 6)
    except Exception:
        return ""


def _write_raw_data_csv(path: Path, session: Any) -> None:
    result = session.result
    diagnostics = getattr(session, "diagnostics", {}) or {}
    diag_keys = sorted(
        key
        for key, values in diagnostics.items()
        if len(np.asarray(values).flatten()) == session.point_count
    )
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "omega_rad_s",
                "freq_hz",
                "Magnitude_raw",
                "Magnitude_smooth",
                "Phase_raw_rad",
                "Phase_smooth_rad",
                "Magnitude_dB",
                "Phase_deg",
                "Real_part",
                "Imag_part",
                *diag_keys,
            ]
        )
        for index, row in enumerate(
            zip(
                result.omega,
                result.omega / (2.0 * np.pi),
                result.mag_raw,
                result.mag_smooth,
                result.phase_raw_rad,
                result.phase_smooth_rad,
                result.mag_db,
                result.phase_deg,
                result.real_part,
                result.imag_part,
            )
        ):
            writer.writerow(
                [
                    *row,
                    *[np.asarray(diagnostics[key], dtype=float).flatten()[index] for key in diag_keys],
                ]
            )


def _write_diagnostics_csv(path: Path, session: Any) -> None:
    diagnosis = session.ai_diagnosis
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "name", "value", "evidence", "suggestion"])
        for key, value in structured_diagnosis_sections(session).items():
            writer.writerow(["summary", key, value, "", ""])
        if diagnosis is None:
            return
        for candidate in getattr(diagnosis, "candidates", [])[:5]:
            writer.writerow(
                [
                    "candidate",
                    f"{getattr(candidate, 'circuit_type', '')}/{getattr(candidate, 'order_estimate', '')}",
                    _fmt(getattr(candidate, "confidence", None), 4),
                    getattr(candidate, "evidence", ""),
                    "",
                ]
            )
        for finding in getattr(diagnosis, "fault_findings", []) or []:
            writer.writerow(
                [
                    "fault",
                    getattr(finding, "name", ""),
                    getattr(finding, "severity", ""),
                    getattr(finding, "evidence", ""),
                    getattr(finding, "suggestion", ""),
                ]
            )
        control_report = getattr(session, "control_compensation_report", None)
        if control_report is None:
            control_report = getattr(diagnosis, "control_compensation_report", None)
        if control_report is not None and getattr(control_report, "enabled", False):
            writer.writerow(["control_compensation", "summary", getattr(control_report, "summary", ""), "", ""])
            for candidate in getattr(control_report, "candidates", []) or []:
                writer.writerow(
                    [
                        "control_compensation",
                        getattr(candidate, "label", ""),
                        _fmt(getattr(candidate, "confidence", None), 4),
                        getattr(candidate, "evidence", ""),
                        getattr(candidate, "suggestion", ""),
                    ]
                )
        conclusion = getattr(diagnosis, "practical_conclusion", "")
        if conclusion:
            writer.writerow(["ai_practical", "conclusion", conclusion, "", ""])
        for suggestion in getattr(diagnosis, "experiment_recommendations", []) or []:
            writer.writerow(["experiment_recommendation", "", suggestion, "", ""])
        for suggestion in getattr(diagnosis, "measurement_suggestions", []) or []:
            writer.writerow(["measurement_suggestion", "", suggestion, "", ""])
        for suggestion in getattr(diagnosis, "control_suggestions", []) or []:
            writer.writerow(["control_suggestion", "", suggestion, "", ""])
        if not getattr(diagnosis, "measurement_suggestions", None) and not getattr(diagnosis, "control_suggestions", None):
            for suggestion in getattr(diagnosis, "next_test_suggestions", []) or []:
                writer.writerow(["suggestion", "", suggestion, "", ""])


def _kv_rows(items: list[tuple[str, Any]]) -> str:
    return "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in items
        if value not in (None, "")
    )


def export_html_report(session: Any, figure: Any, output_dir: str | Path) -> Path:
    if session is None or session.result is None:
        raise ValueError("当前没有可导出的分析结果")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_path = out_dir / "bode_nyquist.png"
    raw_path = out_dir / "raw_data.csv"
    diagnostics_path = out_dir / "diagnostics.csv"
    html_path = out_dir / "report.html"

    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    _write_raw_data_csv(raw_path, session)
    _write_diagnostics_csv(diagnostics_path, session)

    result = session.result
    diagnosis = session.ai_diagnosis
    sections = structured_diagnosis_sections(session)
    report_lines = session.report_lines or build_filter_report_lines(result)
    tf = getattr(diagnosis, "equivalent_transfer_function", None) if diagnosis is not None else None
    best_fit = getattr(diagnosis, "best_fit", None) if diagnosis is not None else None
    formula_view = build_transfer_formula_view(session)
    formula_path = out_dir / "transfer_formula.png"
    formula_image_saved = save_formula_png(formula_view, formula_path)

    sweep_items = [
        ("时间", session.created_at.strftime("%Y-%m-%d %H:%M:%S")),
        ("来源", session.source),
        ("串口", session.serial_port),
        ("波特率", session.baudrate or ""),
        ("扫频命令", session.sweep_command),
        ("期望电路", session.expected_circuit),
        ("点数", session.point_count),
        ("起始频率 Hz", _fmt(result.omega[0] / (2.0 * np.pi))),
        ("终止频率 Hz", _fmt(result.omega[-1] / (2.0 * np.pi))),
    ]
    result_items = [
        ("识别结果", getattr(result, "system_order", "")),
        ("滤波类型", getattr(result, "filter_type", "")),
        ("阶次", getattr(result, "order_estimate", "")),
        ("特征角频率 rad/s", _fmt(getattr(result, "omega_c", None))),
        ("特征频率 Hz", _hz_from_omega(getattr(result, "omega_c", None))),
        ("左截止 Hz", _hz_from_omega(getattr(result, "magnitude_cutoff_omega", None))),
        ("右截止 Hz", _hz_from_omega(getattr(result, "secondary_cutoff_omega", None))),
        ("带宽 Hz", _hz_from_omega(getattr(result, "bandwidth_omega", None))),
        ("稳定性", getattr(result, "stability_text", "")),
    ]
    tf_items = []
    if tf is not None:
        tf_items = [
            ("表达式", getattr(tf, "expression", "")),
            ("参数", getattr(tf, "parameter_summary", "")),
            ("拟合可信度", transfer_fit_reliability_text(best_fit)),
            ("分子系数", ", ".join(_fmt(item) for item in getattr(tf, "numerator", []))),
            ("分母系数", ", ".join(_fmt(item) for item in getattr(tf, "denominator", []))),
        ]

    diagnosis_rows = _kv_rows(list(sections.items()))
    notes_html = html.escape(session.remark or "无").replace("\n", "<br>")
    report_html = "<br>".join(html.escape(line) for line in report_lines)
    formula_block = formula_html(formula_view)
    if formula_image_saved:
        formula_block = formula_block.replace(
            '<div class="formula-display">',
            '<div><img class="formula-image" src="transfer_formula.png" alt="Equivalent transfer function formula"></div>'
            '<div class="formula-display">',
            1,
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>测量报告 - {html.escape(session.created_at.strftime('%Y-%m-%d %H:%M:%S'))}</title>
  <style>
    body {{ font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif; margin: 28px; color: #111827; }}
    h1 {{ font-size: 24px; margin: 0 0 8px; }}
    h2 {{ font-size: 17px; margin: 26px 0 10px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 8px 0 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ width: 180px; background: #f8fafc; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; border: 1px solid #e5e7eb; padding: 12px; }}
    .muted {{ color: #64748b; }}
    .module-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .module-card {{ border: 1px solid #d7dee8; border-radius: 10px; padding: 14px; background: #ffffff; }}
    .module-card h3 {{ font-size: 14px; margin: 0 0 8px; }}
    .module-card p {{ margin: 0; color: #334155; line-height: 1.58; }}
    @media (max-width: 900px) {{ .module-grid {{ grid-template-columns: 1fr; }} }}
    {formula_css()}
  </style>
</head>
<body>
  <h1>频响测量报告</h1>
  <div class="muted">导出时间：{html.escape(session.created_at.strftime('%Y-%m-%d %H:%M:%S'))}</div>

  <div class="module-grid">
    <section class="module-card">
      <h3>模块 1：扫频参数</h3>
      <p>记录本次串口、命令、频率范围和点数，方便复现实测条件。</p>
    </section>
    <section class="module-card">
      <h3>模块 2：识别结论</h3>
      <p>汇总滤波器族、阶次、关键频率、置信度和期望电路匹配状态。</p>
    </section>
    <section class="module-card">
      <h3>模块 3：等效传函</h3>
      <p>把 AI 拟合模型拆成可阅读公式、参数和可信度说明，不再只给文本串。</p>
    </section>
  </div>

  <h2>模块 1：扫频参数</h2>
  <table>{_kv_rows(sweep_items)}</table>

  <h2>模块 2：识别结果与结构化诊断</h2>
  <table>{_kv_rows(result_items)}</table>
  <table>{diagnosis_rows}</table>

  <h2>模块 3：等效传递函数公式</h2>
  {formula_block}
  <table>{_kv_rows(tf_items) if tf_items else '<tr><td>暂无等效传递函数</td></tr>'}</table>

  <h2>测量质量、故障建议与原始报告行</h2>
  <pre>{report_html}</pre>

  <h2>备注</h2>
  <p>{notes_html}</p>

  <h2>Bode / Nyquist 图</h2>
  <img src="bode_nyquist.png" alt="Bode and Nyquist plot">

  <h2>附件</h2>
  <ul>
    <li><a href="raw_data.csv">raw_data.csv</a></li>
    <li><a href="diagnostics.csv">diagnostics.csv</a></li>
    <li><a href="bode_nyquist.png">bode_nyquist.png</a></li>
    {"<li><a href=\"transfer_formula.png\">transfer_formula.png</a></li>" if formula_image_saved else ""}
  </ul>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return html_path
