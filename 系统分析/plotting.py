from __future__ import annotations

from typing import Any

import numpy as np

from filter_analysis import (
    normalize_phase_for_display,
    phase_array_for_display,
)


PLOT_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#f97316",
    "#0891b2",
    "#4b5563",
)


def create_bode_nyquist_axes(fig):
    fig.clear()
    fig.set_facecolor("#f8fafc")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.9, 0.85], height_ratios=[1.0, 1.0])
    ax_mag = fig.add_subplot(grid[0, 0])
    ax_phase = fig.add_subplot(grid[1, 0], sharex=ax_mag)
    ax_ny = fig.add_subplot(grid[:, 1])
    return ax_mag, ax_phase, ax_ny


def style_axes(axes):
    for ax in axes:
        ax.clear()
        ax.set_facecolor("#ffffff")
        ax.grid(True, which="major", color="#cbd5e1", alpha=0.65, linewidth=0.8)
        ax.grid(True, which="minor", color="#e2e8f0", alpha=0.45, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")


def _marker_omegas(result: Any) -> list[tuple[float, str, str]]:
    filter_type = getattr(result, "filter_type", "other")
    marker_omegas: list[tuple[float, str, str]] = []
    if filter_type in ("lowpass", "highpass"):
        marker_omegas.append((float(result.magnitude_cutoff_omega), "ωc", "#dc2626"))
    elif filter_type in ("bandpass", "bandstop"):
        marker_omegas.append((float(result.magnitude_cutoff_omega), "ωL", "#f97316"))
        if getattr(result, "secondary_cutoff_omega", None) is not None:
            marker_omegas.append((float(result.secondary_cutoff_omega), "ωH", "#f97316"))
        marker_omegas.append((float(result.omega_c), "ω0", "#dc2626"))
    else:
        marker_omegas.append((float(result.omega_c), "ω*", "#dc2626"))
    return marker_omegas


def _session_label(session: Any, fallback: str) -> str:
    created_at = getattr(session, "created_at", None)
    if created_at is not None:
        return created_at.strftime("%H:%M:%S")
    return fallback


def _result_for(session: Any):
    return getattr(session, "result", None)


def _plot_line(ax, x, y, *, color, lw, alpha, label=None, semilogx=False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return
    x = x[valid]
    y = y[valid]
    if len(y) < 2:
        if semilogx:
            ax.semilogx(x, y, color=color, lw=lw, alpha=alpha, label=label)
        else:
            ax.plot(x, y, color=color, lw=lw, alpha=alpha, label=label)
        return

    jumps = np.where(np.abs(np.diff(y)) > 175.0)[0]
    starts = [0] + [int(idx + 1) for idx in jumps]
    stops = [int(idx + 1) for idx in jumps] + [len(y)]
    label_used = False
    for start, stop in zip(starts, stops):
        if stop <= start:
            continue
        segment_label = label if not label_used else None
        label_used = True
        if semilogx:
            ax.semilogx(x[start:stop], y[start:stop], color=color, lw=lw, alpha=alpha, label=segment_label)
        else:
            ax.plot(x[start:stop], y[start:stop], color=color, lw=lw, alpha=alpha, label=segment_label)


def plot_sessions(fig, axes, sessions: list[Any], primary: Any | None = None) -> None:
    ax_mag, ax_phase, ax_ny = axes
    style_axes(axes)

    ready = [session for session in sessions if _result_for(session) is not None]
    if not ready:
        ax_mag.text(0.5, 0.5, "等待测量分析", transform=ax_mag.transAxes, ha="center", va="center", color="#64748b")
        ax_phase.text(0.5, 0.5, "暂无相频数据", transform=ax_phase.transAxes, ha="center", va="center", color="#64748b")
        ax_ny.text(0.5, 0.5, "暂无 Nyquist 数据", transform=ax_ny.transAxes, ha="center", va="center", color="#64748b")
        fig.canvas.draw_idle()
        return

    if primary is None or _result_for(primary) is None:
        primary = ready[0]

    for index, session in enumerate(ready):
        result = session.result
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        is_primary = session is primary
        alpha = 1.0 if is_primary else 0.45
        linewidth = 2.1 if is_primary else 1.25
        label = _session_label(session, f"session {getattr(session, 'id', index + 1)}")

        raw_mag_db = 20.0 * np.log10(np.clip(result.mag_raw, 1e-12, None))
        phase_display = phase_array_for_display(result.phase_deg, result.filter_type)
        _plot_line(ax_mag, result.omega, result.mag_db, color=color, lw=linewidth, alpha=alpha, label=label, semilogx=True)
        if is_primary:
            sample_step = max(1, len(result.omega) // 80)
            ax_mag.semilogx(
                result.omega[::sample_step],
                raw_mag_db[::sample_step],
                ".",
                color="#94a3b8",
                ms=3.3,
                alpha=0.5,
                label="原始点",
            )
            ax_mag.axhline(result.cutoff_mag_db, linestyle=":", color="#334155", lw=1.0)
            cutoff_phase_display = normalize_phase_for_display(result.cutoff_phase_deg, result.filter_type)
            ax_phase.axhline(cutoff_phase_display, linestyle=":", color="#334155", lw=1.0)
            for omega_value, marker_label, marker_color in _marker_omegas(result):
                ax_mag.axvline(omega_value, linestyle="--", color=marker_color, lw=1.1, alpha=0.9)
                ax_phase.axvline(omega_value, linestyle="--", color=marker_color, lw=1.1, alpha=0.9)
                ax_mag.annotate(
                    marker_label,
                    xy=(omega_value, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -16),
                    textcoords="offset points",
                    color=marker_color,
                    fontsize=9,
                    ha="left",
                    va="top",
                )

        _plot_line(ax_phase, result.omega, phase_display, color=color, lw=linewidth, alpha=alpha, label=label, semilogx=True)

        if is_primary:
            ax_ny.plot(result.real_part[0], result.imag_part[0], "o", color="#16a34a", ms=6, label="低频")
            ax_ny.plot(result.real_part[-1], result.imag_part[-1], "x", color="#dc2626", ms=7, label="高频")
            ax_ny.plot(result.real_part[result.cutoff_index], result.imag_part[result.cutoff_index], "*", color="#111827", ms=11, label="特征点")
        ax_ny.plot(result.real_part, result.imag_part, color=color, lw=linewidth, alpha=alpha, label=label)

    primary_result = primary.result
    ax_mag.set_title(f"幅频响应  {primary_result.system_order}", loc="left", fontsize=11, color="#0f172a")
    ax_mag.set_ylabel("幅值 (dB)")
    ax_phase.set_title("相频响应", loc="left", fontsize=11, color="#0f172a")
    ax_phase.set_xlabel("角频率 ω (rad/s)")
    ax_phase.set_ylabel("相位 (°)")
    ax_ny.plot(-1, 0, "s", color="#475569", ms=6)
    ax_ny.axhline(0, color="#94a3b8", linestyle=":", lw=0.8)
    ax_ny.axvline(-1, color="#94a3b8", linestyle=":", lw=0.8)
    ax_ny.set_aspect("equal", adjustable="box")
    ax_ny.set_title("Nyquist 图", loc="left", fontsize=11, color="#0f172a")
    ax_ny.set_xlabel("实部 Re{H(jω)}")
    ax_ny.set_ylabel("虚部 Im{H(jω)}")

    ny_x = [np.array([-1.0])]
    ny_y = [np.array([0.0])]
    for session in ready:
        result = session.result
        ny_x.append(np.asarray(result.real_part, dtype=float))
        ny_y.append(np.asarray(result.imag_part, dtype=float))
    all_x = np.concatenate(ny_x)
    all_y = np.concatenate(ny_y)
    x_mid = float((np.max(all_x) + np.min(all_x)) / 2.0)
    y_mid = float((np.max(all_y) + np.min(all_y)) / 2.0)
    span = max(float(np.max(all_x) - np.min(all_x)), float(np.max(all_y) - np.min(all_y)), 0.25) * 0.62
    ax_ny.set_xlim(x_mid - span, x_mid + span)
    ax_ny.set_ylim(y_mid - span, y_mid + span)

    ax_mag.legend(loc="best", fontsize=8, frameon=False)
    ax_phase.legend(loc="best", fontsize=8, frameon=False)
    ax_ny.legend(loc="upper right", fontsize=8, frameon=False)
    fig.canvas.draw_idle()
