from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from math import isclose, pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np


SCHEMATIC_RC_R_OHM = 10_000.0
SCHEMATIC_RC_C_F = 10e-9


@dataclass(frozen=True)
class TransferFormulaView:
    title: str
    subtitle: str
    latex: str
    plain: str
    numerator_html: str = ""
    denominator_html: str = ""
    details: list[str] = field(default_factory=list)
    schematic_notes: list[str] = field(default_factory=list)

    def copy_text(self) -> str:
        lines = [self.title, self.plain]
        if self.subtitle:
            lines.append(self.subtitle)
        lines.extend(self.details)
        lines.extend(self.schematic_notes)
        return "\n".join(line for line in lines if line)


def _format_number(value: float, digits: int = 4) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1.0e4 or (0.0 < abs(value) < 1.0e-3):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def _latex_number(value: float, digits: int = 3) -> str:
    text = _format_number(value, digits)
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e", 1)
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _format_hz(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    value = float(value)
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.3g} kHz"
    return f"{value:.3g} Hz"


def _format_rad_s(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "无法计算"
    value = float(value)
    if abs(value) >= 1.0e4:
        return f"{value:.3e} rad/s"
    return f"{value:.4g} rad/s"


def _coefficient_latex(value: float, power: int) -> str:
    coeff = abs(float(value))
    if power > 0 and isclose(coeff, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12):
        coeff_text = ""
    else:
        coeff_text = _latex_number(coeff)
    if power == 0:
        return coeff_text or "1"
    if power == 1:
        return rf"{coeff_text}s"
    return rf"{coeff_text}s^{{{power}}}"


def _polynomial_latex(coefficients: list[float]) -> str:
    degree = len(coefficients) - 1
    terms: list[str] = []
    for index, raw_coeff in enumerate(coefficients):
        coeff = float(raw_coeff)
        if abs(coeff) < 1.0e-12:
            continue
        power = degree - index
        term = _coefficient_latex(coeff, power)
        if not terms:
            terms.append(term if coeff >= 0 else "-" + term)
        else:
            terms.append(("+" if coeff >= 0 else "-") + term)
    return " ".join(terms) if terms else "0"


def _polynomial_plain(coefficients: list[float]) -> str:
    degree = len(coefficients) - 1
    terms: list[str] = []
    for index, raw_coeff in enumerate(coefficients):
        coeff = float(raw_coeff)
        if abs(coeff) < 1.0e-12:
            continue
        power = degree - index
        coeff_text = _format_number(abs(coeff), 4)
        if power == 0:
            term = coeff_text
        elif power == 1:
            term = "s" if isclose(abs(coeff), 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12) else f"{coeff_text}s"
        else:
            term = f"s^{power}" if isclose(abs(coeff), 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12) else f"{coeff_text}s^{power}"
        if not terms:
            terms.append(term if coeff >= 0 else "-" + term)
        else:
            terms.append(("+ " if coeff >= 0 else "- ") + term)
    return " ".join(terms) if terms else "0"


def _compact_low_high_view(tf: Any, fit: Any | None, family_label: str) -> TransferFormulaView | None:
    params = getattr(fit, "parameters", {}) if fit is not None else {}
    model_shape = str(getattr(fit, "model_shape", "") or "")
    order = int(getattr(tf, "order", 0) or 0)
    gain = float(getattr(tf, "gain", 1.0) or 1.0)
    pole_fc_hz = params.get("pole_fc_hz", params.get("fc_hz"))
    if model_shape != "cascade" or order <= 0 or pole_fc_hz is None or pole_fc_hz <= 0:
        return None

    omega_p = 2.0 * pi * float(pole_fc_hz)
    fc_hz = params.get("fc_hz", pole_fc_hz)
    gain_text = _format_number(gain, 4)
    omega_latex = _latex_number(omega_p, 3)
    order_latex = f"^{{{order}}}" if order != 1 else ""

    if getattr(tf, "model_family", "") == "lowpass":
        numerator = gain_text
        denominator = f"(1 + s/{_format_rad_s(omega_p).replace(' rad/s', '')})^{order}"
        latex = rf"H(s)=\frac{{{_latex_number(gain, 3)}}}{{\left(1+\frac{{s}}{{{omega_latex}}}\right){order_latex}}}"
    else:
        numerator = f"{gain_text}(s/{_format_rad_s(omega_p).replace(' rad/s', '')})^{order}"
        denominator = f"(1 + s/{_format_rad_s(omega_p).replace(' rad/s', '')})^{order}"
        latex = (
            rf"H(s)=\frac{{{_latex_number(gain, 3)}\left(\frac{{s}}{{{omega_latex}}}\right){order_latex}}}"
            rf"{{\left(1+\frac{{s}}{{{omega_latex}}}\right){order_latex}}}"
        )

    details = [
        f"模型形态: {family_label}{order}阶一阶级联模板",
        f"增益 K = {gain_text}",
        f"单级极点 fp = {_format_hz(float(pole_fc_hz))}，ωp = {_format_rad_s(omega_p)}",
        f"总 -3 dB 频率 = {_format_hz(float(fc_hz))}",
    ]
    if fit is not None:
        details.append(
            f"拟合质量: R2={float(getattr(fit, 'r_squared', 0.0)):.3f}, "
            f"RMSE={float(getattr(fit, 'rmse_db', 0.0)):.2f} dB"
        )

    return TransferFormulaView(
        title="等效传递函数",
        subtitle=f"{family_label}{order}阶级联模型",
        latex=latex,
        plain=f"H(s) = {numerator} / {denominator}",
        numerator_html=escape(numerator),
        denominator_html=escape(denominator),
        details=details,
    )


def _expanded_view(tf: Any, fit: Any | None, family_label: str) -> TransferFormulaView:
    numerator = [float(item) for item in getattr(tf, "numerator", [])]
    denominator = [float(item) for item in getattr(tf, "denominator", [])]
    if not numerator or not denominator:
        expression = str(getattr(tf, "expression", "") or "暂无等效传递函数")
        return TransferFormulaView("等效传递函数", family_label, r"H(s)", expression, details=[])

    latex_num = _polynomial_latex(numerator)
    latex_den = _polynomial_latex(denominator)
    plain_num = _polynomial_plain(numerator)
    plain_den = _polynomial_plain(denominator)
    details = [str(getattr(tf, "parameter_summary", "") or "")]
    if fit is not None:
        details.append(
            f"拟合质量: R2={float(getattr(fit, 'r_squared', 0.0)):.3f}, "
            f"RMSE={float(getattr(fit, 'rmse_db', 0.0)):.2f} dB"
        )
    return TransferFormulaView(
        title="等效传递函数",
        subtitle=family_label,
        latex=rf"H(s)=\frac{{{latex_num}}}{{{latex_den}}}",
        plain=f"H(s) = ({plain_num}) / ({plain_den})",
        numerator_html=escape(plain_num),
        denominator_html=escape(plain_den),
        details=[line for line in details if line],
    )


def _schematic_notes_for_lowpass3(view: TransferFormulaView, diagnosis: Any | None) -> list[str]:
    circuit_type = str(getattr(diagnosis, "circuit_type", "") or "")
    order = int(getattr(diagnosis, "order_estimate", 0) or 0)
    if circuit_type != "lowpass" or order != 3:
        return []

    tau = SCHEMATIC_RC_R_OHM * SCHEMATIC_RC_C_F
    omega_p = 1.0 / tau
    pole_fc = omega_p / (2.0 * pi)
    total_fc = pole_fc * sqrt(max(2.0 ** (1.0 / 3.0) - 1.0, 1.0e-12))
    return [
        "实物电路参考: 三个 10 kΩ / 10 nF RC 低通级串联，每级后由 LM358 电压跟随器隔离。",
        f"理论单级: τ=RC={tau * 1e6:.1f} us，fp={_format_hz(pole_fc)}，ωp={_format_rad_s(omega_p)}。",
        f"理论总 -3 dB: f≈{_format_hz(total_fc)}；拓扑公式为 H(s)=1/(1+sRC)^3。",
        "LM358 带宽、面包板寄生和高频噪声底会改变高频相位，最终报告以实测拟合参数为准。",
    ]


def build_transfer_formula_view(session_or_diagnosis: Any | None) -> TransferFormulaView:
    if session_or_diagnosis is None:
        return TransferFormulaView("等效传递函数", "", "H(s)", "暂无等效传递函数")

    diagnosis = getattr(session_or_diagnosis, "ai_diagnosis", session_or_diagnosis)
    tf = getattr(diagnosis, "equivalent_transfer_function", None)
    if tf is None:
        return TransferFormulaView("等效传递函数", "", "H(s)", "暂无等效传递函数")

    fit = getattr(diagnosis, "best_fit", None)
    family = str(getattr(tf, "model_family", "") or getattr(diagnosis, "circuit_type", ""))
    family_labels = {
        "lowpass": "低通",
        "highpass": "高通",
        "bandpass": "带通",
        "bandstop": "带阻",
    }
    family_label = family_labels.get(family, "等效模型")

    view = None
    if family in ("lowpass", "highpass"):
        view = _compact_low_high_view(tf, fit, family_label)
    if view is None:
        view = _expanded_view(tf, fit, family_label)
    notes = _schematic_notes_for_lowpass3(view, diagnosis)
    if notes:
        view = TransferFormulaView(
            title=view.title,
            subtitle=view.subtitle,
            latex=view.latex,
            plain=view.plain,
            numerator_html=view.numerator_html,
            denominator_html=view.denominator_html,
            details=view.details,
            schematic_notes=notes,
        )
    return view


def formula_css() -> str:
    return """
    .formula-card { border: 1px solid #d7dee8; border-radius: 12px; padding: 18px; background: #ffffff; }
    .formula-title { font-size: 15px; font-weight: 700; color: #172033; margin-bottom: 10px; }
    .formula-display { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; font-size: 26px; font-weight: 700; color: #111827; margin: 12px 0; }
    .formula-frac { display: inline-flex; flex-direction: column; align-items: stretch; min-width: 280px; text-align: center; line-height: 1.22; }
    .formula-num { border-bottom: 2px solid #172033; padding: 0 10px 5px; }
    .formula-den { padding: 5px 10px 0; }
    .formula-details { margin: 10px 0 0; color: #334155; line-height: 1.65; }
    .formula-details li { margin: 2px 0; }
    .formula-image { max-width: 100%; border: 0; background: #ffffff; }
    """


def formula_html(view: TransferFormulaView) -> str:
    details = "".join(f"<li>{escape(line)}</li>" for line in (view.details + view.schematic_notes) if line)
    if view.numerator_html and view.denominator_html:
        display = (
            '<div class="formula-display">'
            '<span>H(s)=</span>'
            '<span class="formula-frac">'
            f'<span class="formula-num">{view.numerator_html}</span>'
            f'<span class="formula-den">{view.denominator_html}</span>'
            "</span></div>"
        )
    else:
        display = f'<div class="formula-display"><span>{escape(view.plain)}</span></div>'
    return (
        '<div class="formula-card">'
        f'<div class="formula-title">{escape(view.title)}'
        + (f' · {escape(view.subtitle)}' if view.subtitle else "")
        + "</div>"
        + display
        + (f'<ul class="formula-details">{details}</ul>' if details else "")
        + "</div>"
    )


def save_formula_png(view: TransferFormulaView, path: str | Path) -> bool:
    if not view.latex or view.latex == "H(s)":
        return False
    try:
        from matplotlib.figure import Figure
    except Exception:
        return False

    path = Path(path)
    fig = Figure(figsize=(7.4, 1.65), dpi=180, facecolor="#ffffff")
    ax = fig.add_subplot(111)
    ax.axis("off")
    try:
        ax.text(0.5, 0.52, f"${view.latex}$", ha="center", va="center", fontsize=16, color="#111827")
        fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
        return True
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return False
